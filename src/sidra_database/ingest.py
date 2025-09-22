"""Metadata ingestion pipeline."""
from __future__ import annotations

import asyncio
import hashlib
from array import array
from datetime import datetime, timezone
from typing import Any, NamedTuple, Sequence

import orjson

from .api_client import SidraApiClient
from .db import ensure_schema, sqlite_session
from .embedding import EmbeddingClient

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


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
    summarizacao = classificacao.get("sumarizacao")
    if isinstance(sumarizacao, dict):
        status = summarizacao.get("status")
        if status is not None:
            lines.append(f"Summarization enabled: {bool(status)}")
        excecao = summarizacao.get("excecao")
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
    targets: list[_EmbeddingTarget] = []
    table_text = _canonical_agregado_text(metadata)
    if table_text:
        targets.append(
            _EmbeddingTarget(
                "agregado",
                str(agregado_id),
                agregado_id,
                table_text,
                _hash_text("agregado", str(agregado_id), table_text),
            )
        )

    for variable in metadata.get("variaveis", []) or []:
        vid = variable.get("id")
        if vid is None:
            continue
        variable_text = _canonical_variable_text(metadata, variable)
        if not variable_text:
            continue
        targets.append(
            _EmbeddingTarget(
                "variable",
                f"{agregado_id}:{vid}",
                agregado_id,
                variable_text,
                _hash_text("variable", str(agregado_id), str(vid), variable_text),
            )
        )

    for classificacao in metadata.get("classificacoes", []) or []:
        cid = classificacao.get("id")
        if cid is None:
            continue
        classification_text = _canonical_classification_text(metadata, classificacao)
        if classification_text:
            targets.append(
                _EmbeddingTarget(
                    "classification",
                    f"{agregado_id}:{cid}",
                    agregado_id,
                    classification_text,
                    _hash_text("classification", str(agregado_id), str(cid), classification_text),
                )
            )
        for categoria in classificacao.get("categorias", []) or []:
            cat_id = categoria.get("id")
            if cat_id is None:
                continue
            category_text = _canonical_category_text(metadata, classificacao, categoria)
            if not category_text:
                continue
            targets.append(
                _EmbeddingTarget(
                    "category",
                    f"{agregado_id}:{cid}:{cat_id}",
                    agregado_id,
                    category_text,
                    _hash_text(
                        "category",
                        str(agregado_id),
                        str(cid),
                        str(cat_id),
                        category_text,
                    ),
                )
            )

    return targets


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
) -> None:
    """Fetch and persist metadata for a single agregados table."""

    ensure_schema()
    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True
    if embedding_client is None:
        embedding_client = EmbeddingClient()
    try:
        metadata = await client.fetch_metadata(agregado_id)
        periods = await client.fetch_periods(agregado_id)
        nivel_groups: dict[str, list[str]] = metadata.get("nivelTerritorial", {}) or {}

        locality_payloads: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for level_type, codes in nivel_groups.items():
            if not codes:
                continue
            for level_code in codes:
                localities = await client.fetch_localities(agregado_id, level_code)
                locality_payloads[(level_type, level_code)] = list(localities)

        embedding_targets = _build_embedding_targets(agregado_id, metadata)
        fetched_at = _utcnow()
        with sqlite_session() as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT OR REPLACE INTO agregados (
                    id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

            conn.execute("DELETE FROM agregados_levels WHERE agregado_id = ?", (agregado_id,))
            for (level_type, level_code), locs in locality_payloads.items():
                level_name = None
                if locs:
                    level_name = locs[0].get("nivel", {}).get("nome")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agregados_levels (agregado_id, level_id, level_name, level_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (agregado_id, level_code, level_name, level_type),
                )

            conn.execute("DELETE FROM variables WHERE agregado_id = ?", (agregado_id,))
            for variable in metadata.get("variaveis", []) or []:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO variables (id, agregado_id, nome, unidade, sumarizacao, text_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
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
                    ),
                )

            conn.execute("DELETE FROM classifications WHERE agregado_id = ?", (agregado_id,))
            conn.execute("DELETE FROM categories WHERE agregado_id = ?", (agregado_id,))
            for classificacao in metadata.get("classificacoes", []) or []:
                cid = classificacao.get("id")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO classifications (
                        id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        agregado_id,
                        classificacao.get("nome"),
                        1 if (classificacao.get("sumarizacao", {}).get("status")) else 0,
                        _json_dump_text(classificacao.get("sumarizacao", {}).get("excecao", [])),
                    ),
                )
                for categoria in classificacao.get("categorias", []) or []:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO categories (
                            agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
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
                        ),
                    )

            conn.execute("DELETE FROM periods WHERE agregado_id = ?", (agregado_id,))
            for period in periods:
                pid = period.get("id") if isinstance(period, dict) else period
                literals = period.get("literals", [pid]) if isinstance(period, dict) else [period]
                modificacao = period.get("modificacao") if isinstance(period, dict) else None
                conn.execute(
                    """
                    INSERT OR REPLACE INTO periods (agregado_id, periodo_id, literals, modificacao)
                    VALUES (?, ?, ?, ?)
                    """,
                    (agregado_id, str(pid), _json_dump_text(literals), modificacao),
                )

            conn.execute("DELETE FROM localities WHERE agregado_id = ?", (agregado_id,))
            for (_, level_code), locs in locality_payloads.items():
                for loc in locs:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO localities (agregado_id, level_id, locality_id, nome)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            agregado_id,
                            level_code,
                            loc.get("id"),
                            loc.get("nome"),
                        ),
                    )

            await _persist_embeddings(conn, embedding_targets, embedding_client, fetched_at)

            conn.execute(
                """
                INSERT INTO ingestion_log (agregado_id, stage, status, detail, run_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agregado_id, "metadata", "success", None, fetched_at),
            )
            conn.commit()
    finally:
        if own_client:
            await client.close()


def ingest_agregado_sync(agregado_id: int) -> None:
    asyncio.run(ingest_agregado(agregado_id))


__all__ = ["ingest_agregado", "ingest_agregado_sync"]
