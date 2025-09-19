"""Metadata ingestion pipeline."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any, Iterable

import orjson

from .api_client import SidraApiClient
from .db import ensure_schema, sqlite_session

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _hash_text(*parts: str) -> str:
    joined = "||".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _json_dumps(obj: Any) -> bytes:
    return orjson.dumps(obj)


def _json_dump_text(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


async def ingest_agregado(agregado_id: int, *, client: SidraApiClient | None = None) -> None:
    """Fetch and persist metadata for a single agregados table."""

    ensure_schema()
    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True
    try:
        metadata = await client.fetch_metadata(agregado_id)
        periods = await client.fetch_periods(agregado_id)
        nivel_groups: dict[str, list[str]] = metadata.get("nivelTerritorial", {}) or {}

        locality_payloads: dict[tuple[str, str], Iterable[dict[str, Any]]] = {}
        for level_type, codes in nivel_groups.items():
            if not codes:
                continue
            for level_code in codes:
                localities = await client.fetch_localities(agregado_id, level_code)
                locality_payloads[(level_type, level_code)] = localities

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
                locs_list = list(locs)
                if locs_list:
                    level_name = locs_list[0].get("nivel", {}).get("nome")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agregados_levels (agregado_id, level_id, level_name, level_type)
                    VALUES (?, ?, ?, ?)
                    """,
                    (agregado_id, level_code, level_name, level_type),
                )

            conn.execute("DELETE FROM variables WHERE agregado_id = ?", (agregado_id,))
            for variable in metadata.get("variaveis", []):
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
                        _hash_text(str(variable.get("id")), variable.get("nome", ""), variable.get("unidade", "")),
                    ),
                )

            conn.execute("DELETE FROM classifications WHERE agregado_id = ?", (agregado_id,))
            conn.execute("DELETE FROM categories WHERE agregado_id = ?", (agregado_id,))
            for classificacao in metadata.get("classificacoes", []):
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
