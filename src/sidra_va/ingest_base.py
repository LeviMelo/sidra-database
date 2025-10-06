"""Metadata ingestion pipeline."""
from __future__ import annotations

import asyncio
import hashlib
from array import array
from datetime import datetime, timezone
from typing import Any, NamedTuple, Sequence

import orjson

from .api_client import SidraApiClient
from .config import get_settings
from .db import sqlite_session
from .embedding_client import EmbeddingClient

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MUNICIPALITY_LEVEL_CODE = "N6"


class _EmbeddingTarget(NamedTuple):
    entity_type: str
    entity_id: str
    agregado_id: int
    text: str
    text_hash: str


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _hash_text(*parts: str) -> str:
    joined = "||".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _json_dumps(obj: Any) -> bytes:
    return orjson.dumps(obj)


def _json_dump_text(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def _line_or_none(prefix: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return f"{prefix}{text}"


def _canonical_agregado_text(metadata: dict[str, Any]) -> str:
    periodicidade = metadata.get("periodicidade") or {}
    freq = periodicidade.get("frequencia")
    inicio = periodicidade.get("inicio")
    fim = periodicidade.get("fim")

    if inicio and fim:
        period_line = f"Period: {inicio} - {fim}" if inicio != fim else f"Period: {inicio}"
    elif inicio or fim:
        period_line = f"Period: {inicio or fim}"
    else:
        period_line = None

    nivel = metadata.get("nivelTerritorial") or {}
    level_parts: list[str] = []
    for level_type in sorted(nivel):
        codes = nivel.get(level_type) or []
        if codes:
            level_parts.append(f"{level_type}: {', '.join(codes)}")

    lines = [
        f"Table {metadata.get('id')}: {str(metadata.get('nome') or '').strip()}".strip(),
        _line_or_none("Survey: ", metadata.get("pesquisa")),
        _line_or_none("Subject: ", metadata.get("assunto")),
        _line_or_none("Frequency: ", freq),
        period_line,
        f"Territorial levels: {'; '.join(level_parts)}" if level_parts else None,
        _line_or_none("URL: ", metadata.get("URL")),
    ]
    return "\n".join(line for line in lines if line)


def _canonical_variable_text(metadata: dict[str, Any], variable: dict[str, Any]) -> str:
    lines = [
        f"Table {metadata.get('id')}: {str(metadata.get('nome') or '').strip()}".strip(),
        f"Variable {variable.get('id')}: {str(variable.get('nome') or '').strip()}".strip(),
    ]
    unit_line = _line_or_none("Unit: ", variable.get("unidade"))
    if unit_line:
        lines.append(unit_line)
    subject_line = _line_or_none("Subject: ", metadata.get("assunto"))
    if subject_line:
        lines.append(subject_line)
    survey_line = _line_or_none("Survey: ", metadata.get("pesquisa"))
    if survey_line:
        lines.append(survey_line)
    summary_data = variable.get("sumarizacao")
    if summary_data:
        lines.append(f"Summarization: {_json_dump_text(summary_data)}")
    return "\n".join(line for line in lines if line)


def _canonical_classification_text(metadata: dict[str, Any], classificacao: dict[str, Any]) -> str:
    lines = [
        f"Table {metadata.get('id')}: {str(metadata.get('nome') or '').strip()}".strip(),
        f"Classification {classificacao.get('id')}: {str(classificacao.get('nome') or '').strip()}".strip(),
    ]
    summary_payload = classificacao.get("sumarizacao")
    if isinstance(summary_payload, dict):
        status = summary_payload.get("status")
        if status is not None:
            lines.append(f"Summarization enabled: {bool(status)}")
        excecao = summary_payload.get("excecao")
        if excecao:
            lines.append(f"Exceptions: {_json_dump_text(excecao)}")
    return "\n".join(line for line in lines if line)


def _canonical_category_text(
    metadata: dict[str, Any],
    classificacao: dict[str, Any],
    categoria: dict[str, Any],
) -> str:
    lines = [
        f"Table {metadata.get('id')}: {str(metadata.get('nome') or '').strip()}".strip(),
        f"Classification {classificacao.get('id')}: {str(classificacao.get('nome') or '').strip()}".strip(),
        f"Category {categoria.get('id')}: {str(categoria.get('nome') or '').strip()}".strip(),
    ]
    unit_line = _line_or_none("Unit: ", categoria.get("unidade"))
    if unit_line:
        lines.append(unit_line)
    level = categoria.get("nivel")
    if level is not None:
        lines.append(f"Level: {level}")
    return "\n".join(line for line in lines if line)

def _normalize_period_id_to_ord_kind(periodo_id: Any) -> tuple[int | None, str]:
    """
    Map raw period id to a sortable ordinal and a kind:
      YYYY    -> ord=YYYY00, kind='Y'
      YYYYMM  -> ord=YYYYMM, kind='YM'
      YYYYMMDD-> ord=YYYYMMDD, kind='YMD'
      otherwise -> (None, 'UNK')
    We only strip digits; we don't guess beyond obvious lengths.
    """
    s = str(periodo_id or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) == 4:   # YYYY
        try:
            return int(digits + "00"), "Y"
        except ValueError:
            return None, "UNK"
    if len(digits) == 6:   # YYYYMM
        try:
            return int(digits), "YM"
        except ValueError:
            return None, "UNK"
    if len(digits) == 8:   # YYYYMMDD
        try:
            return int(digits), "YMD"
        except ValueError:
            return None, "UNK"

    return None, "UNK"

def _build_embedding_targets(agregado_id: int, metadata: dict[str, Any]) -> list[_EmbeddingTarget]:
    table_text = _canonical_agregado_text(metadata)
    if not table_text:
        return []
    return [
        _EmbeddingTarget(
            "agregado",
            str(agregado_id),
            agregado_id,
            table_text,
            _hash_text("agregado", str(agregado_id), table_text),
        )
    ]


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    arr = array("f", (float(value) for value in vector))
    return arr.tobytes()


async def _persist_embeddings(
    conn,
    targets: list[_EmbeddingTarget],
    embedding_client: EmbeddingClient,
    timestamp: str,
) -> None:
    if not targets:
        return

    model_name = embedding_client.model
    for target in targets:
        if not target.text.strip():
            continue
        existing = conn.execute(
            "SELECT text_hash FROM embeddings WHERE entity_type = ? AND entity_id = ? AND model = ?",
            (target.entity_type, target.entity_id, model_name),
        ).fetchone()
        if existing and existing["text_hash"] == target.text_hash:
            continue

        vector = await asyncio.to_thread(embedding_client.embed_text, target.text, model=model_name)
        dimension = len(vector)
        if dimension == 0:
            continue

        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target.entity_type,
                target.entity_id,
                target.agregado_id,
                target.text_hash,
                model_name,
                dimension,
                _vector_to_blob(vector),
                timestamp,
            ),
        )


async def ingest_agregado(
    agregado_id: int,
    *,
    client: SidraApiClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    generate_embeddings: bool = True,
    db_lock: asyncio.Lock | None = None,
    prefetched_localities: dict[str, list[dict[str, Any]]] | None = None,
) -> None:

    """Fetch and persist metadata for a single agregados table."""

    own_client = False
    settings = get_settings()
    if client is None:
        client = SidraApiClient()
        own_client = True
    if generate_embeddings:
        if embedding_client is None:
            embedding_client = EmbeddingClient()
    else:
        embedding_client = None
    try:
        try:
            raw_metadata = await client.fetch_metadata(agregado_id)
        except Exception:
            _log_ingestion_status(agregado_id, "error", "api")
            raise

        try:
            metadata = _validate_metadata_payload(raw_metadata)
        except IngestionValidationError:
            _log_ingestion_status(agregado_id, "error", "validation")
            raise

        try:
            periods = await client.fetch_periods(agregado_id)
        except Exception:
            _log_ingestion_status(agregado_id, "error", "api")
            raise

        # --- derive territorial levels from metadata
        nivel_groups_raw = metadata.get("nivelTerritorial") or {}
        nivel_groups = nivel_groups_raw if isinstance(nivel_groups_raw, dict) else {}

        municipality_locality_count: int = 0
        level_rows: list[tuple[int, str, str | None, str, int]] = []
        locality_rows: list[tuple[int, str, str | None, str | None]] = []

        for level_type, codes in nivel_groups.items():
            if not codes:
                continue
            for level_code in codes:
                # Prefer prefetched localities for this level if provided
                payload_list = None
                if prefetched_localities and isinstance(prefetched_localities, dict):
                    payload_list = (
                        prefetched_localities.get(str(level_code))
                        or prefetched_localities.get(str(level_code).upper())
                    )

                if payload_list is None:
                    try:
                        localities = await client.fetch_localities(agregado_id, level_code)
                    except Exception:
                        _log_ingestion_status(agregado_id, "error", "api")
                        raise
                    raw_payload = localities or []
                else:
                    raw_payload = payload_list or []

                # Normalize to list of dicts
                if isinstance(raw_payload, list):
                    payload = raw_payload
                else:
                    try:
                        payload = list(raw_payload)
                    except TypeError:
                        payload = []

                count = len(payload)
                if str(level_code).upper() == MUNICIPALITY_LEVEL_CODE and count > municipality_locality_count:
                    municipality_locality_count = count

                level_name = None
                if payload:
                    level_name = (payload[0].get("nivel", {}) or {}).get("nome")

                level_rows.append(
                    (
                        agregado_id,
                        str(level_code),
                        level_name,
                        str(level_type),
                        count,
                    )
                )
                for loc in payload:
                    locality_rows.append(
                        (
                            agregado_id,
                            str(level_code),
                            loc.get("id"),
                            loc.get("nome"),
                        )
                    )

        raw_threshold = getattr(settings, "municipality_national_threshold", 0)
        try:
            threshold_setting = max(0, int(raw_threshold))
        except (TypeError, ValueError):
            threshold_setting = 0
        if threshold_setting == 0:
            covers_national_munis = 1 if municipality_locality_count > 0 else 0
        else:
            covers_national_munis = 1 if municipality_locality_count >= threshold_setting else 0

        variables_payload = metadata.get("variaveis") or []
        variable_rows: list[tuple[Any, int, Any, Any, str, str]] = []
        for variable in variables_payload:
            variable_rows.append(
                (
                    variable.get("id"),
                    agregado_id,
                    variable.get("nome"),
                    variable.get("unidade"),
                    _json_dump_text(variable.get("sumarizacao", [])),
                    _hash_text(
                        str(variable.get("id")),
                        variable.get("nome", ""),
                        variable.get("unidade", ""),
                    ),
                )
            )

        classification_rows: list[tuple[int, int, str | None, int, str]] = []
        category_rows: list[tuple[int, int, int, str | None, str | None, Any, str]] = []
        for classificacao in metadata.get("classificacoes", []) or []:
            cid = classificacao.get("id")
            classification_rows.append(
                (
                    cid,
                    agregado_id,
                    classificacao.get("nome"),
                    1 if (classificacao.get("sumarizacao", {}).get("status")) else 0,
                    _json_dump_text(classificacao.get("sumarizacao", {}).get("excecao", [])),
                )
            )
            for categoria in classificacao.get("categorias", []) or []:
                category_rows.append(
                    (
                        agregado_id,
                        cid,
                        categoria.get("id"),
                        categoria.get("nome"),
                        categoria.get("unidade"),
                        categoria.get("nivel"),
                        _hash_text(
                            str(cid),
                            str(categoria.get("id")),
                            categoria.get("nome", ""),
                            categoria.get("unidade", ""),
                        ),
                    )
                )

        period_rows: list[tuple[int, str, str, Any, int | None, str]] = []
        for period in periods or []:
            pid = period.get("id") if isinstance(period, dict) else period
            literals = period.get("literals", [pid]) if isinstance(period, dict) else [period]
            modificacao = period.get("modificacao") if isinstance(period, dict) else None
            ord_val, kind = _normalize_period_id_to_ord_kind(pid)
            period_rows.append(
                (agregado_id, str(pid), _json_dump_text(literals), modificacao, ord_val, kind)
            )

        embedding_targets: list[_EmbeddingTarget] = []
        if generate_embeddings:
            embedding_targets = _build_embedding_targets(agregado_id, metadata)

        fetched_at = _utcnow()

        async def _write_to_database() -> None:
            with sqlite_session() as conn:
                conn.execute("BEGIN")
                try:
                    # 0) Upsert agregados header row first (no children depend on it)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO agregados (
                            id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at,
                            municipality_locality_count, covers_national_municipalities
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            metadata.get("id"),
                            metadata.get("nome"),
                            metadata.get("pesquisa"),
                            metadata.get("assunto"),
                            metadata.get("URL"),
                            metadata.get("periodicidade", {}).get("frequencia"),
                            metadata.get("periodicidade", {}).get("inicio"),
                            metadata.get("periodicidade", {}).get("fim"),
                            _json_dumps(metadata),
                            fetched_at,
                            municipality_locality_count,
                            covers_national_munis,
                        ),
                    )

                    # 1) DELETE children first (FK-safe), then parents
                    #    localities -> agregados_levels
                    conn.execute("DELETE FROM localities WHERE agregado_id = ?", (agregado_id,))
                    conn.execute("DELETE FROM agregados_levels WHERE agregado_id = ?", (agregado_id,))

                    #    categories -> classifications
                    conn.execute("DELETE FROM categories WHERE agregado_id = ?", (agregado_id,))
                    conn.execute("DELETE FROM classifications WHERE agregado_id = ?", (agregado_id,))

                    #    variables have no children we manage here, but purge before reinsert
                    conn.execute("DELETE FROM variables WHERE agregado_id = ?", (agregado_id,))

                    #    periods: only child of agregados (we keep agregados), so purge them safely
                    conn.execute("DELETE FROM periods WHERE agregado_id = ?", (agregado_id,))

                    # 2) INSERT parents first, then children

                    # agregados_levels (parent of localities)
                    if level_rows:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO agregados_levels (
                                agregado_id, level_id, level_name, level_type, locality_count
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            level_rows,
                        )

                    # variables (standalone; used by VA index later)
                    if variable_rows:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO variables (
                                id, agregado_id, nome, unidade, sumarizacao, text_hash
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            variable_rows,
                        )

                    # classifications (parent of categories)
                    if classification_rows:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO classifications (
                                id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            classification_rows,
                        )

                    # categories (child of classifications)
                    if category_rows:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO categories (
                                agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            category_rows,
                        )

                    # periods (standalone child of agregados)
                    if period_rows:
                        def _iter_period_rows():
                            for row in period_rows:
                                # Accept [ag, pid, literals_json, modificacao, ord, kind]  (new)
                                # or    [ag, pid, literals_json, modificacao]             (old)
                                if len(row) >= 6:
                                    ag, pid, literals_json, modificacao, ord_val, kind = row[:6]
                                    # backfill if someone built with None ord/kind
                                    if ord_val is None or kind is None:
                                        ord_val, kind = _normalize_period_id_to_ord_kind(pid)
                                elif len(row) == 4:
                                    ag, pid, literals_json, modificacao = row
                                    ord_val, kind = _normalize_period_id_to_ord_kind(pid)
                                else:
                                    # very defensive fallback
                                    ag = row[0]
                                    pid = row[1]
                                    literals_json = row[2] if len(row) > 2 else "[]"
                                    modificacao = row[3] if len(row) > 3 else None
                                    ord_val, kind = _normalize_period_id_to_ord_kind(pid)
                                yield (ag, pid, literals_json, modificacao, ord_val, kind)

                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO periods (
                                agregado_id, periodo_id, literals, modificacao, periodo_ord, periodo_kind
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            list(_iter_period_rows()),
                        )

                    # localities (child of agregados_levels)
                    if locality_rows:
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO localities (
                                agregado_id, level_id, locality_id, nome
                            ) VALUES (?, ?, ?, ?)
                            """,
                            locality_rows,
                        )

                    # 3) (optional) embeddings for table header
                    if generate_embeddings and embedding_targets and embedding_client is not None:
                        await _persist_embeddings(conn, embedding_targets, embedding_client, fetched_at)

                    # 4) log success
                    conn.execute(
                        """
                        INSERT INTO ingestion_log (agregado_id, stage, status, detail, run_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (agregado_id, "metadata", "success", None, fetched_at),
                    )

                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        try:
            if db_lock is not None:
                async with db_lock:
                    await _write_to_database()
            else:
                await _write_to_database()
        except Exception:
            _log_ingestion_status(agregado_id, "error", "db")
            raise

    finally:
        if own_client:
            await client.close()


async def generate_embeddings_for_agregado(
    agregado_id: int,
    *,
    embedding_client: EmbeddingClient | None = None,
    db_lock: asyncio.Lock | None = None,
) -> None:
    """Regenerate embeddings for an already ingested agregado."""

    if embedding_client is None:
        embedding_client = EmbeddingClient()

    with sqlite_session() as conn:
        row = conn.execute(
            "SELECT raw_json FROM agregados WHERE id = ?",
            (agregado_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Agregado {agregado_id} not found in database")

    metadata = orjson.loads(row["raw_json"])
    if not metadata or not metadata.get("nome"):
        # Synthetic fixtures may not provide the full metadata payload. Skip
        # embedding generation if the required context is missing to avoid
        # spurious network calls during tests.
        return
    embedding_targets = _build_embedding_targets(agregado_id, metadata)
    if not embedding_targets:
        return

    fetched_at = _utcnow()

    async def _write() -> None:
        with sqlite_session() as conn:
            conn.execute("BEGIN")
            await _persist_embeddings(conn, embedding_targets, embedding_client, fetched_at)
            conn.commit()

    if db_lock is not None:
        async with db_lock:
            await _write()
    else:
        await _write()


def ingest_agregado_sync(agregado_id: int) -> None:
    asyncio.run(ingest_agregado(agregado_id))


__all__ = ["ingest_agregado", "ingest_agregado_sync", "generate_embeddings_for_agregado"]
class IngestionValidationError(RuntimeError):
    """Raised when the SIDRA API returns an incomplete metadata payload."""


def _validate_metadata_payload(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise IngestionValidationError("metadata payload is not a JSON object")

    nome = metadata.get("nome")
    if not isinstance(nome, str) or not nome.strip():
        raise IngestionValidationError("metadata payload missing 'nome'")

    variaveis = metadata.get("variaveis")
    if variaveis is not None and not isinstance(variaveis, list):
        raise IngestionValidationError("metadata field 'variaveis' is not a list")
    classificacoes = metadata.get("classificacoes")
    if classificacoes is not None and not isinstance(classificacoes, list):
        raise IngestionValidationError("metadata field 'classificacoes' is not a list")
    has_variables = isinstance(variaveis, list) and any(variaveis)
    has_classifications = isinstance(classificacoes, list) and any(classificacoes)
    if not has_variables and not has_classifications:
        raise IngestionValidationError("metadata payload missing both 'variaveis' and 'classificacoes'")

    return metadata


def _log_ingestion_status(agregado_id: int, status: str, detail: str) -> None:
    timestamp = _utcnow()

    with sqlite_session() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (agregado_id, stage, status, detail, run_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agregado_id, "metadata", status, detail, timestamp),
            )


