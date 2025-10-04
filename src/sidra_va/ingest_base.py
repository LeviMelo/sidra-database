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

        nivel_groups_raw = metadata.get("nivelTerritorial", {}) or {}
        nivel_groups = nivel_groups_raw if isinstance(nivel_groups_raw, dict) else {}

        municipality_locality_count: int = 0
        level_rows: list[tuple[int, str, str | None, str, int]] = []
        locality_rows: list[tuple[int, str, str | None, str | None]] = []
        for level_type, codes in nivel_groups.items():
            if not codes:
                continue
            for level_code in codes:
                try:
                    localities = await client.fetch_localities(agregado_id, level_code)
                except Exception:
                    _log_ingestion_status(agregado_id, "error", "api")
                    raise
                raw_payload = localities or []
                if isinstance(raw_payload, list):
                    payload = raw_payload
                else:
                    try:
                        payload = list(raw_payload)
                    except TypeError:
                        payload = []
                count = len(payload)
                if level_code.upper() == MUNICIPALITY_LEVEL_CODE and count > municipality_locality_count:
                    municipality_locality_count = count
                level_name = None
                if payload:
                    level_name = (payload[0].get("nivel", {}) or {}).get("nome")
                level_rows.append(
                    (
                        agregado_id,
                        level_code,
                        level_name,
                        level_type,
                        count,
                    )
                )
                for loc in payload:
                    locality_rows.append(
                        (
                            agregado_id,
                            level_code,
                            loc.get("id"),
                            loc.get("nome"),
                        )
                    )

        threshold_setting = max(0, int(getattr(settings, "municipality_national_threshold", 0) or 0))
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

        period_rows: list[tuple[int, str, str, Any]] = []
        for period in periods or []:
            pid = period.get("id") if isinstance(period, dict) else period
            literals = period.get("literals", [pid]) if isinstance(period, dict) else [period]
            modificacao = period.get("modificacao") if isinstance(period, dict) else None
            period_rows.append((agregado_id, str(pid), _json_dump_text(literals), modificacao))

        embedding_targets: list[_EmbeddingTarget] = []
        if generate_embeddings:
            embedding_targets = _build_embedding_targets(agregado_id, metadata)

        fetched_at = _utcnow()

        async def _write_to_database() -> None:
            with sqlite_session() as conn:
                conn.execute("BEGIN")
                try:
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

                    if level_rows:
                        conn.execute("DELETE FROM agregados_levels WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO agregados_levels (
                                agregado_id, level_id, level_name, level_type, locality_count
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            level_rows,
                        )

                    if variable_rows:
                        conn.execute("DELETE FROM variables WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO variables (
                                id, agregado_id, nome, unidade, sumarizacao, text_hash
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            variable_rows,
                        )

                    if classification_rows:
                        conn.execute("DELETE FROM classifications WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO classifications (
                                id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            classification_rows,
                        )

                    if category_rows:
                        conn.execute("DELETE FROM categories WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO categories (
                                agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            category_rows,
                        )

                    if period_rows:
                        conn.execute("DELETE FROM periods WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO periods (
                                agregado_id, periodo_id, literals, modificacao
                            ) VALUES (?, ?, ?, ?)
                            """,
                            period_rows,
                        )

                    if locality_rows:
                        conn.execute("DELETE FROM localities WHERE agregado_id = ?", (agregado_id,))
                        conn.executemany(
                            """
                            INSERT OR REPLACE INTO localities (
                                agregado_id, level_id, locality_id, nome
                            ) VALUES (?, ?, ?, ?)
                            """,
                            locality_rows,
                        )

                    if generate_embeddings and embedding_targets and embedding_client is not None:
                        await _persist_embeddings(conn, embedding_targets, embedding_client, fetched_at)

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


