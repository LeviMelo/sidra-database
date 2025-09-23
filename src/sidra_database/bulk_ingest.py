"""Bulk ingestion helpers that discover and ingest SIDRA agregados."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence
from typing import Any, Callable

from .api_client import SidraApiClient
from .db import ensure_schema, sqlite_session
from .discovery import CatalogEntry, fetch_catalog_entries, filter_catalog_entries
from .embedding import EmbeddingClient
from .ingest import ingest_agregado, generate_embeddings_for_agregado


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
    progress_callback: Callable[[str], None] | None = None,
    generate_embeddings: bool = True,
) -> BulkIngestionReport:
    """Discover agregados using coverage filters and ingest them."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    ensure_schema()

    def _emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

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

        total_to_schedule = len(to_schedule)
        if total_to_schedule:
            _emit(
                f"Scheduling {total_to_schedule} agregados for ingestion "
                f"with concurrency={concurrency}"
            )
        else:
            _emit("No new agregados matched the requested filters.")

        if dry_run or not to_schedule:
            return report

        semaphore = asyncio.Semaphore(concurrency)
        db_lock = asyncio.Lock() if concurrency > 1 else None

        progress_step = max(1, total_to_schedule // 100) if total_to_schedule > 0 else 1
        progress_time_budget = 15.0
        last_progress_at = time.monotonic()
        completed = 0

        async def _run(agregado_id: int) -> None:
            nonlocal completed
            nonlocal last_progress_at
            async with semaphore:
                try:
                    await ingest_agregado(
                        agregado_id,
                        client=client,
                        embedding_client=embedding_client,
                        generate_embeddings=False,
                        db_lock=db_lock,
                    )
                except Exception as exc:  # noqa: BLE001
                    report.failed.append((agregado_id, str(exc)))
                    _emit(f"Failed agregado {agregado_id}: {exc}")
                else:
                    report.ingested_ids.append(agregado_id)
                finally:
                    completed += 1
                    now = time.monotonic()
                    should_emit = (
                        completed <= 10
                        or completed == total_to_schedule
                        or completed % progress_step == 0
                        or (now - last_progress_at) >= progress_time_budget
                    )
                    if should_emit:
                        last_progress_at = now
                        ingested = len(report.ingested_ids)
                        failed = len(report.failed)
                        _emit(
                            f"Progress {completed}/{total_to_schedule}: "
                            f"ingested={ingested}, failed={failed}"
                        )

        await asyncio.gather(*(_run(agregado_id) for agregado_id in to_schedule))

        if generate_embeddings and report.ingested_ids:
            embed_client = embedding_client or EmbeddingClient()
            embed_lock = asyncio.Lock() if concurrency > 1 else None
            embed_semaphore = asyncio.Semaphore(concurrency)
            embed_total = len(report.ingested_ids)
            embed_step = max(1, embed_total // 100) if embed_total > 0 else 1
            embed_last_progress = time.monotonic()
            completed_embeds = 0

            async def _embed(agregado_id: int) -> None:
                nonlocal completed_embeds
                nonlocal embed_last_progress
                async with embed_semaphore:
                    try:
                        await generate_embeddings_for_agregado(
                            agregado_id,
                            embedding_client=embed_client,
                            db_lock=embed_lock,
                        )
                    except Exception as exc:  # noqa: BLE001
                        report.failed.append((agregado_id, f"embedding: {exc}"))
                        _emit(f"Embedding failed for {agregado_id}: {exc}")
                    finally:
                        completed_embeds += 1
                        now = time.monotonic()
                        if (
                            completed_embeds <= 10
                            or completed_embeds == embed_total
                            or completed_embeds % embed_step == 0
                            or (now - embed_last_progress) >= progress_time_budget
                        ):
                            embed_last_progress = now
                            _emit(
                                f"Embedding progress {completed_embeds}/{embed_total}"
                            )

            _emit(
                f"Generating embeddings for {len(report.ingested_ids)} agregados"
            )
            await asyncio.gather(
                *(_embed(agregado_id) for agregado_id in report.ingested_ids)
            )
        return report
    finally:
        if own_client:
            await client.close()


__all__ = ["BulkIngestionReport", "discover_agregados_by_coverage", "ingest_by_coverage"]
