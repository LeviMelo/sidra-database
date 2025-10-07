from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..net.api_client import SidraApiClient
from ..db.session import sqlite_session, ensure_full_schema
from .ingest_table import ingest_table
from ..search.coverage import parse_coverage_expr, extract_levels, eval_coverage
from ..config import get_settings

@dataclass(frozen=True)
class CatalogEntry:
    id: int
    nome: str | None
    pesquisa: str | None
    pesquisa_id: int | None
    assunto: str | None
    assunto_id: int | None
    periodicidade: Any
    nivel_territorial: dict[str, list[str]]
    level_hints: frozenset[str] = frozenset()

    @property
    def level_codes(self) -> set[str]:
        codes: set[str] = set()
        for vals in self.nivel_territorial.values():
            for v in vals:
                codes.add(str(v).upper())
        codes.update(self.level_hints)
        return codes

def _normalize_levels(payload: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(payload, dict):
        for k, vals in payload.items():
            arr = vals if isinstance(vals, list) else [vals]
            out = []
            for it in arr:
                if isinstance(it, str):
                    out.append(it.upper())
                elif isinstance(it, dict):
                    code = it.get("codigo") or it.get("nivel") or it.get("id")
                    if isinstance(code, str): out.append(code.upper())
            if out: result[str(k)] = out
    return result

async def fetch_catalog_entries(
    *, client: SidraApiClient | None = None, subject_id: int | None = None,
    periodicity: str | None = None, levels: Sequence[str] | None = None
) -> list[CatalogEntry]:
    own = False
    if client is None:
        client = SidraApiClient()
        own = True
    normalized_levels = [c.upper() for c in levels or [] if c]
    try:
        catalog = await client.fetch_catalog(subject_id=subject_id, periodicity=periodicity, levels=normalized_levels or None)
    finally:
        if own:
            await client.close()

    out: list[CatalogEntry] = []
    if not isinstance(catalog, list): return out
    for survey in catalog:
        ags = survey.get("agregados") if isinstance(survey, dict) else None
        if not isinstance(ags, list): continue
        for ag in ags:
            if not isinstance(ag, dict): continue
            entry = CatalogEntry(
                id = int(ag.get("id")),
                nome = ag.get("nome") or ag.get("tabela"),
                pesquisa = survey.get("pesquisa") or survey.get("nome"),
                pesquisa_id = survey.get("idPesquisa") or survey.get("id"),
                assunto = (survey.get("assunto") or {}).get("nome") if isinstance(survey.get("assunto"), dict) else survey.get("assunto"),
                assunto_id = (survey.get("assunto") or {}).get("id") if isinstance(survey.get("assunto"), dict) else survey.get("idAssunto"),
                periodicidade = survey.get("periodicidade"),
                nivel_territorial = _normalize_levels(ag.get("nivelTerritorial")),
                level_hints = frozenset(c.upper() for c in normalized_levels),
            )
            out.append(entry)
    return out

def filter_catalog_entries(
    entries: Sequence[CatalogEntry], *,
    require_any_levels: Iterable[str] | None = None,
    require_all_levels: Iterable[str] | None = None,
    exclude_levels: Iterable[str] | None = None,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
) -> list[CatalogEntry]:
    any_levels = {c.upper() for c in require_any_levels or ()}
    all_levels = {c.upper() for c in require_all_levels or ()}
    excluded  = {c.upper() for c in exclude_levels or ()}
    subj_q = subject_contains.lower() if subject_contains else None
    surv_q = survey_contains.lower() if survey_contains else None

    out: list[CatalogEntry] = []
    for e in entries:
        codes = e.level_codes
        if any_levels and not (codes & any_levels): continue
        if all_levels and not all_levels.issubset(codes): continue
        if excluded and (codes & excluded): continue
        if subj_q and (e.assunto or "").lower().find(subj_q) == -1: continue
        if surv_q and (e.pesquisa or "").lower().find(surv_q) == -1: continue
        out.append(e)
    return out


@dataclass
class BulkReport:
    discovered_ids: list[int] = field(default_factory=list)
    scheduled_ids: list[int] = field(default_factory=list)
    skipped_existing: list[int] = field(default_factory=list)
    ingested_ids: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

async def _probe_counts_for_levels(
    client: SidraApiClient,
    table_id: int,
    levels: Iterable[str],
) -> dict[str, int]:
    """
    Fetch locality lists per level for a table and return {LEVEL: count}.
    Levels are normalized to UPPERCASE strings. Non-list responses count as 0.
    """
    counts: dict[str, int] = {}
    for lvl in {str(l).upper() for l in levels if l}:
        try:
            payload = await client.fetch_localities(table_id, lvl)
            if isinstance(payload, list):
                counts[lvl] = len(payload)
            else:
                # defensive: sometimes API oddities; treat non-list as 0
                counts[lvl] = 0
        except Exception:
            counts[lvl] = 0
    return counts


async def ingest_by_coverage(
    *,
    coverage: str,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
    limit: int | None = None,
    concurrency: int = 8,
    probe_concurrent: int | None = None,
) -> BulkReport:
    """
    Discover tables that satisfy the coverage expression and ingest them.

    Fixes:
      - Probes are bounded (no huge gather).
      - Early-stop probing once we have enough matches.
      - Per-table probe timeout.
      - Progress ticks so it doesn't look frozen.
    """
    ensure_full_schema()
    report = BulkReport()

    # Parse coverage and extract hinted levels (e.g., {"N3","N6"})
    try:
        cov_ast = parse_coverage_expr(coverage)
    except Exception as exc:
        raise RuntimeError(f"Invalid --coverage expression: {coverage!r}: {exc}") from exc
    level_hints = extract_levels(cov_ast)

    # How many matches we actually need to keep
    need = max(0, int(limit)) if (limit is not None and limit >= 0) else None
    par_probe = max(1, int(probe_concurrent or concurrency))

    async with SidraApiClient() as client:
        # 1) Catalog fetch with hinted levels (cheap server-side pruning)
        entries = await fetch_catalog_entries(client=client, levels=sorted(level_hints) or None)

        # 2) Optional subject/survey narrowing (kept)
        entries = filter_catalog_entries(
            entries,
            require_any_levels=None,
            require_all_levels=None,
            exclude_levels=None,
            subject_contains=subject_contains,
            survey_contains=survey_contains,
        )

        total = len(entries)
        if total == 0:
            return report

        print(f"[probe] candidates={total}, levels={sorted(level_hints) or []}, parallel={par_probe}")

        kept: list[CatalogEntry] = []
        pending: set[asyncio.Task] = set()
        it = iter(entries)

        # Helper: schedule next probe task (skips tables that don't claim needed levels)
        async def schedule_next() -> bool:
            try:
                e = next(it)
            except StopIteration:
                return False
            check_levels = (level_hints & e.level_codes) if level_hints else set()
            if not check_levels:
                return True  # nothing to probe for this one; move on

            async def _one(e_: CatalogEntry) -> tuple[CatalogEntry, bool]:
                # Hard timeout per table to avoid indefinite stalls
                per_table_timeout = max(10.0, 2.0 * float(get_settings().request_timeout))
                try:
                    counts = await asyncio.wait_for(
                        _probe_counts_for_levels(client, e_.id, check_levels),
                        timeout=per_table_timeout,
                    )
                    ok = eval_coverage(cov_ast, counts)
                    return e_, ok
                except Exception:
                    return e_, False

            t = asyncio.create_task(_one(e))
            pending.add(t)
            return True

        # Prime the initial probe window
        for _ in range(par_probe):
            if not await schedule_next():
                break

        probed = 0
        # Drain as tasks complete; keep backfilling to maintain parallelism; early stop when enough kept
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                probed += 1
                e, ok = await t
                if ok:
                    kept.append(e)

                # progress tick every 10 completes, or when we hit the target
                if (probed % 10 == 0) or (need and len(kept) >= need):
                    pct = (probed * 100) // max(1, total)
                    print(f"\r[probe] probed={probed}/{total} kept={len(kept)} ({pct}%)", end="", flush=True)

                if need and len(kept) >= need:
                    pending.clear()
                    break

                while len(pending) < par_probe:
                    more = await schedule_next()
                    if not more:
                        break

        print()  # newline after progress

        # 3) Cap to limit (if any) and record discovery
        if need is not None:
            kept = kept[:need]
        report.discovered_ids = [e.id for e in kept]

        # 4) Skip already ingested
        with sqlite_session() as conn:
            existing = {int(r[0]) for r in conn.execute("SELECT id FROM agregados")}
        to_do = [e.id for e in kept if e.id not in existing]
        report.scheduled_ids = list(to_do)
        report.skipped_existing = [e.id for e in kept if e.id in existing]

        if not to_do:
            return report

        # 5) Ingest concurrently (bounded)
        print(f"[ingest] {len(to_do)} tables (parallel={concurrency})")
        sem = asyncio.Semaphore(max(1, concurrency))

        async def worker(tid: int) -> None:
            async with sem:
                try:
                    await ingest_table(tid)
                    report.ingested_ids.append(tid)
                    print(f"  ingested {tid}")
                except Exception as exc:
                    report.failed.append((tid, str(exc)[:200]))
                    print(f"  failed {tid}: {exc}")

        await asyncio.gather(*(worker(t) for t in to_do))

    return report

