"""Bulk ingestion helpers that discover and ingest SIDRA agregados."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence
from typing import Any

from .api_client import SidraApiClient
from .db import ensure_schema, sqlite_session
from .discovery import CatalogEntry, fetch_catalog_entries, filter_catalog_entries
from .embedding import EmbeddingClient
from .ingest import ingest_agregado


@dataclass(slots=True)
class BulkIngestionReport:
    """Outcome of a bulk ingestion run."""

    discovered_ids: list[int] = field(default_factory=list)
    scheduled_ids: list[int] = field(default_factory=list)
    skipped_existing: list[int] = field(default_factory=list)
    ingested_ids: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable summary of the ingestion run."""

        return {
            "discovered": self.discovered_ids,
            "scheduled": self.scheduled_ids,
            "skipped_existing": self.skipped_existing,
            "ingested": self.ingested_ids,
            "failed": self.failed,
        }


async def discover_agregados_by_coverage(
    *,
    client: SidraApiClient | None = None,
    require_any_levels: Iterable[str] | None = None,
    require_all_levels: Iterable[str] | None = None,
    exclude_levels: Iterable[str] | None = None,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
    limit: int | None = None,
) -> list[CatalogEntry]:
    """Return catalog entries matching the requested territorial filters."""

    level_filter: list[str] | None = None
    if require_any_levels or require_all_levels:
        combined = [
            *(code.upper() for code in (require_any_levels or []) if code),
            *(code.upper() for code in (require_all_levels or []) if code),
        ]
        if combined:
            level_filter = list(dict.fromkeys(combined))

    entries = await fetch_catalog_entries(
        client=client,
        levels=level_filter,
    )
    filtered = filter_catalog_entries(
        entries,
        require_any_levels=require_any_levels,
        require_all_levels=require_all_levels,
        exclude_levels=exclude_levels,
        subject_contains=subject_contains,
        survey_contains=survey_contains,
    )
    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


async def ingest_by_coverage(
    *,
    require_any_levels: Iterable[str] | None = None,
    require_all_levels: Iterable[str] | None = None,
    exclude_levels: Iterable[str] | None = None,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
    limit: int | None = None,
    concurrency: int = 8,
    skip_existing: bool = True,
    dry_run: bool = False,
    client: SidraApiClient | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> BulkIngestionReport:
    """Discover agregados using coverage filters and ingest them."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    ensure_schema()

    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True

    report = BulkIngestionReport()

    try:
        candidates = await discover_agregados_by_coverage(
            client=client,
            require_any_levels=require_any_levels,
            require_all_levels=require_all_levels,
            exclude_levels=exclude_levels,
            subject_contains=subject_contains,
            survey_contains=survey_contains,
            limit=limit,
        )
        report.discovered_ids = [entry.id for entry in candidates]

        existing_ids: set[int] = set()
        if skip_existing:
            with sqlite_session() as conn:
                rows = conn.execute("SELECT id FROM agregados")
                existing_ids = {int(row["id"]) for row in rows}

        to_schedule: list[int] = []
        for entry in candidates:
            if entry.id in existing_ids:
                report.skipped_existing.append(entry.id)
                continue
            to_schedule.append(entry.id)
        report.scheduled_ids = list(to_schedule)

        if dry_run or not to_schedule:
            return report

        semaphore = asyncio.Semaphore(concurrency)

        async def _run(agregado_id: int) -> None:
            async with semaphore:
                try:
                    await ingest_agregado(
                        agregado_id,
                        client=client,
                        embedding_client=embedding_client,
                    )
                except Exception as exc:  # noqa: BLE001
                    report.failed.append((agregado_id, str(exc)))
                else:
                    report.ingested_ids.append(agregado_id)

        await asyncio.gather(*(_run(agregado_id) for agregado_id in to_schedule))
        return report
    finally:
        if own_client:
            await client.close()


__all__ = ["BulkIngestionReport", "discover_agregados_by_coverage", "ingest_by_coverage"]
