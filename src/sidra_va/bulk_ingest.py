"""Bulk ingestion helpers that discover and ingest SIDRA agregados."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence
from typing import Any, Callable

from .api_client import SidraApiClient
from .db import ensure_full_schema, sqlite_session
from .discovery import CatalogEntry, fetch_catalog_entries, filter_catalog_entries
from .embedding_client import EmbeddingClient
from .ingest_base import generate_embeddings_for_agregado, ingest_agregado
from .coverage import parse_coverage_expr, extract_levels, eval_coverage
from .utils import utcnow_iso


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

async def _probe_counts_and_prefetch(
    entries: Sequence[CatalogEntry],
    levels_to_probe: set[str],
    *,
    client: SidraApiClient,
    concurrency: int = 16,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[int, dict[str, int]], dict[int, dict[str, list[dict]]]]:
    """
    For each catalog entry, fetch localities for the requested levels and compute counts.
    We do NOT trust the catalog to enumerate supported levels; we try the levels
    mentioned in the coverage expression. Failures/404s are treated as empty lists.
    Returns:
      (counts_map, prefetch_map)
        counts_map[agregado_id][level_code] -> int
        prefetch_map[agregado_id][level_code] -> list[locality_dict]
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    counts_map: dict[int, dict[str, int]] = {}
    prefetch_map: dict[int, dict[str, list[dict]]] = {}

    total = len(entries)
    processed = 0
    emit_every = max(50, total // 20)  # ~5% or at least every 50
    last_emit = time.monotonic()

    async def _one(entry: CatalogEntry) -> None:
        nonlocal processed, last_emit
        entry_counts: dict[str, int] = {}
        entry_prefetch: dict[str, list[dict]] = {}
        needed = [lvl.upper() for lvl in levels_to_probe]
        for lvl in needed:
            async with sem:
                try:
                    payload = await client.fetch_localities(entry.id, lvl)
                    if not isinstance(payload, list):
                        try:
                            payload = list(payload)
                        except Exception:
                            payload = []
                except Exception:
                    payload = []
                entry_counts[lvl] = len(payload)
                entry_prefetch[lvl] = payload
        counts_map[entry.id] = entry_counts
        prefetch_map[entry.id] = entry_prefetch

        processed += 1
        if progress_callback:
            now = time.monotonic()
            if (
                processed <= 10
                or processed == total
                or processed % emit_every == 0
                or (now - last_emit) >= 5.0
            ):
                last_emit = now
                progress_callback(f"Probing progress {processed}/{total} ({processed*100//max(1,total)}%)")

    await asyncio.gather(*(_one(e) for e in entries))
    if progress_callback:
        progress_callback(f"Probed locality counts for {total} agregados across {len(levels_to_probe)} levels.")
    return counts_map, prefetch_map


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
    coverage_expr: str | None = None,
    probe_concurrency: int = 16,
) -> BulkIngestionReport:
    """Discover agregados using coverage filters and ingest them."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    ensure_full_schema()

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
        # Optional: client-time coverage filtering using a boolean expression
        prefetch_map: dict[int, dict[str, list[dict]]] = {}
        if coverage_expr:
            try:
                ast = parse_coverage_expr(coverage_expr)
            except Exception as exc:
                _emit(f"Invalid --coverage expression: {exc}")
                return report

            needed_levels = extract_levels(ast)
            if not needed_levels:
                _emit("Coverage expression has no level identifiers; skipping coverage probe.")
            else:
                _emit(f"Probing coverage levels {sorted(needed_levels)} for {len(candidates)} candidates...")
                counts_map, prefetch_map = await _probe_counts_and_prefetch(
                    candidates,
                    needed_levels,
                    client=client,
                    concurrency=max(1, probe_concurrency),
                    progress_callback=_emit,
                )

                # Keep only entries that satisfy the expression
                filtered: list[CatalogEntry] = []
                for entry in candidates:
                    counts = counts_map.get(entry.id, {})
                    if eval_coverage(ast, counts):
                        filtered.append(entry)
                candidates = filtered
                report.discovered_ids = [e.id for e in candidates]
                _emit(f"Coverage expression kept {len(candidates)} agregados.")

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
                        prefetched_localities=prefetch_map.get(agregado_id),
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
                        # Log success
                        ts = utcnow_iso()
                        with sqlite_session() as conn:
                            with conn:
                                conn.execute(
                                    """
                                    INSERT INTO ingestion_log (agregado_id, stage, status, detail, run_at)
                                    VALUES (?, ?, ?, ?, ?)
                                    """,
                                    (agregado_id, "embedding", "success", None, ts),
                                )
                    except Exception as exc:  # noqa: BLE001
                        report.failed.append((agregado_id, f"embedding: {exc}"))
                        _emit(f"Embedding failed for {agregado_id}: {exc}")
                        # Log failure
                        ts = utcnow_iso()
                        with sqlite_session() as conn:
                            with conn:
                                conn.execute(
                                    """
                                    INSERT INTO ingestion_log (agregado_id, stage, status, detail, run_at)
                                    VALUES (?, ?, ?, ?, ?)
                                    """,
                                    (agregado_id, "embedding", "error", str(exc)[:200], ts),
                                )
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
                            _emit(f"Embedding progress {completed_embeds}/{embed_total}")


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
