"""Diagnostics and repair helpers for SIDRA metadata ingestion."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .api_client import SidraApiClient
from .db import sqlite_session
from .ingest import ingest_agregado


def _fetch_scalar(conn, query: str, params: Sequence[Any] | None = None) -> int:
    cursor = conn.execute(query, params or ())
    row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _get_index_presence(conn, table: str, expected: Iterable[str]) -> dict[str, bool]:
    cursor = conn.execute(f"PRAGMA index_list({table})")
    available = {row[1] for row in cursor.fetchall()}
    return {name: name in available for name in expected}


def collect_missing_variable_ids(conn, *, limit: int | None = None) -> list[int]:
    query = (
        "SELECT a.id FROM agregados AS a "
        "LEFT JOIN (SELECT DISTINCT agregado_id FROM variables) AS v ON v.agregado_id = a.id "
        "WHERE v.agregado_id IS NULL ORDER BY a.id"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    cursor = conn.execute(query)
    return [int(row[0]) for row in cursor.fetchall()]


def global_health_report(conn, *, sample_limit: int = 50) -> dict[str, Any]:
    total_agregados = _fetch_scalar(conn, "SELECT COUNT(*) FROM agregados")
    agregados_with_variables = _fetch_scalar(
        conn, "SELECT COUNT(DISTINCT agregado_id) FROM variables"
    )
    agregados_with_classifications = _fetch_scalar(
        conn, "SELECT COUNT(DISTINCT agregado_id) FROM classifications"
    )
    agregados_with_categories = _fetch_scalar(
        conn, "SELECT COUNT(DISTINCT agregado_id) FROM categories"
    )

    missing_ids = collect_missing_variable_ids(conn)
    sample_missing = missing_ids[:sample_limit]

    index_status = {
        "variables": _get_index_presence(conn, "variables", ["idx_variables_agregado"]),
        "categories": _get_index_presence(conn, "categories", ["idx_categories_agregado"]),
        "localities": _get_index_presence(conn, "localities", ["idx_localities_agregado"]),
        "embeddings": _get_index_presence(conn, "embeddings", ["idx_embeddings_agregado"]),
    }

    journal_row = conn.execute("PRAGMA journal_mode").fetchone()
    journal_mode = str(journal_row[0]).upper() if journal_row else "UNKNOWN"

    return {
        "counts": {
            "agregados": total_agregados,
            "agregados_with_variables": agregados_with_variables,
            "agregados_with_classifications": agregados_with_classifications,
            "agregados_with_categories": agregados_with_categories,
            "agregados_missing_variables": len(missing_ids),
        },
        "sample_missing_variables": sample_missing,
        "indexes": index_status,
        "journal_mode": journal_mode,
    }


async def api_vs_db_spot_check(
    conn,
    *,
    sample_size: int = 10,
    client: SidraApiClient | None = None,
) -> dict[str, Any]:
    missing_ids = collect_missing_variable_ids(conn, limit=sample_size)
    if not missing_ids:
        return {
            "sampled": 0,
            "api_nonempty": 0,
            "api_empty": 0,
            "errors": 0,
            "details": [],
        }

    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True

    results: list[dict[str, Any]] = []
    try:
        for agregado_id in missing_ids:
            try:
                metadata = await client.fetch_metadata(agregado_id)
                variaveis = metadata.get("variaveis") if isinstance(metadata, dict) else None
                if isinstance(variaveis, list):
                    count = len(variaveis)
                else:
                    count = 0
                results.append(
                    {
                        "agregado_id": agregado_id,
                        "variables_in_api": count,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "agregado_id": agregado_id,
                        "variables_in_api": None,
                        "error": str(exc)[:200],
                    }
                )
    finally:
        if own_client:
            await client.close()

    api_nonempty = sum(1 for item in results if (item["variables_in_api"] or 0) > 0)
    api_empty = sum(1 for item in results if item["variables_in_api"] == 0)
    errors = sum(1 for item in results if item["error"])

    return {
        "sampled": len(results),
        "api_nonempty": api_nonempty,
        "api_empty": api_empty,
        "errors": errors,
        "details": results,
    }


@dataclass
class RepairResult:
    attempted: int
    succeeded: int
    failed: int
    failures: list[tuple[int, str]]


async def _ingest_chunk(
    ids: Sequence[int],
    *,
    concurrency: int,
    client: SidraApiClient,
) -> list[tuple[int, bool, str | None]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[tuple[int, bool, str | None]] = []

    async def worker(agregado_id: int) -> None:
        async with semaphore:
            try:
                await ingest_agregado(
                    agregado_id,
                    client=client,
                    generate_embeddings=False,
                )
                results.append((agregado_id, True, None))
            except Exception as exc:  # noqa: BLE001
                results.append((agregado_id, False, str(exc)[:200]))

    await asyncio.gather(*(worker(ag_id) for ag_id in ids))
    return results


async def repair_missing_variables(
    *,
    chunk_size: int = 50,
    concurrency: int = 6,
    limit: int | None = None,
    max_retries: int = 3,
) -> RepairResult:
    with sqlite_session() as conn:
        missing_ids = collect_missing_variable_ids(conn, limit=limit)

    if not missing_ids:
        return RepairResult(attempted=0, succeeded=0, failed=0, failures=[])

    attempted = len(missing_ids)
    succeeded = 0
    failures: list[tuple[int, str]] = []

    async with SidraApiClient() as client:
        for start in range(0, len(missing_ids), max(1, chunk_size)):
            chunk = missing_ids[start : start + max(1, chunk_size)]
            remaining = list(chunk)
            attempt = 0
            partial_failures: list[tuple[int, str]] = []
            while remaining and attempt < max(1, max_retries):
                attempt += 1
                results = await _ingest_chunk(
                    remaining,
                    concurrency=concurrency,
                    client=client,
                )
                remaining = [ag_id for ag_id, ok, _ in results if not ok]
                succeeded += sum(1 for _, ok, _ in results if ok)
                partial_failures = [
                    (ag_id, error or "unknown error")
                    for ag_id, ok, error in results
                    if not ok
                ]

            failures.extend(partial_failures)

    failed = len(failures)
    return RepairResult(
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        failures=failures,
    )


__all__ = [
    "RepairResult",
    "api_vs_db_spot_check",
    "collect_missing_variable_ids",
    "global_health_report",
    "repair_missing_variables",
]

