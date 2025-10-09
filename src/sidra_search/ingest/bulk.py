from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..net.api_client import SidraApiClient
from ..db.session import sqlite_session, ensure_full_schema
from .ingest_table import ingest_table
from ..search.coverage import parse_coverage_expr, extract_levels, eval_coverage
from ..config import get_settings
from ..search.normalize import normalize_basic

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

def _extract_subject_fields(survey: Any, ag: Any) -> tuple[str | None, int | None]:
    """
    Try hard to extract a human-readable subject name and id
    from either the table (agregado) or the survey object.
    Handles: plain str, nested dict {'nome','id'}, and a few common aliases.
    """
    def _name_and_id(obj: Any) -> tuple[str | None, int | None]:
        if not isinstance(obj, dict):
            return None, None
        v = obj.get("assunto")
        # variante 1: nested object
        if isinstance(v, dict):
            name = v.get("nome") or v.get("name")
            sid  = v.get("id") or obj.get("idAssunto")
            return (name if isinstance(name, str) else None,
                    int(sid) if isinstance(sid, (int, str)) and str(sid).isdigit() else None)
        # variante 2: plain string
        if isinstance(v, str):
            sid = obj.get("idAssunto")
            return v, int(sid) if isinstance(sid, (int, str)) and str(sid).isdigit() else None
        # variante 3: loose aliases some payloads use
        for k in ("assuntoNome", "assunto_nome", "assuntoDescricao", "assunto_descricao"):
            w = obj.get(k)
            if isinstance(w, str):
                sid = obj.get("idAssunto")
                return w, int(sid) if isinstance(sid, (int, str)) and str(sid).isdigit() else None
        return None, None

    # Prefer table-level, fallback to survey-level
    a_name, a_id = _name_and_id(ag)
    if a_name:
        return a_name, a_id
    s_name, s_id = _name_and_id(survey)
    return s_name, s_id

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
            assunto_name, assunto_id = _extract_subject_fields(survey, ag)

            entry = CatalogEntry(
                id = int(ag.get("id")),
                nome = ag.get("nome") or ag.get("tabela"),
                pesquisa = survey.get("pesquisa") or survey.get("nome"),
                pesquisa_id = survey.get("idPesquisa") or survey.get("id"),
                assunto = assunto_name,
                assunto_id = assunto_id,
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
    subj_q = normalize_basic(subject_contains) if subject_contains else None
    surv_q = normalize_basic(survey_contains) if survey_contains else None

    out: list[CatalogEntry] = []
    for e in entries:
        codes = e.level_codes
        if any_levels and not (codes & any_levels): continue
        if all_levels and not all_levels.issubset(codes): continue
        if excluded and (codes & excluded): continue
        subj_val = normalize_basic(e.assunto or "")
        surv_val = normalize_basic(e.pesquisa or "")
        if subj_q and subj_q not in subj_val: continue
        if surv_q and surv_q not in surv_val: continue
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
) -> tuple[dict[str, int], dict[str, list]]:
    """
    Fetch locality lists per level for a table and return:
      - counts: {LEVEL: count}
      - payloads: {LEVEL: list_of_localities}  (for reuse during ingest)
    Levels are normalized to UPPERCASE strings. Non-list responses are treated as empty lists.
    """
    lvls = sorted({str(l).upper() for l in levels if l})
    if not lvls:
        return {}, {}

    async def one(lvl: str) -> tuple[str, list]:
        try:
            payload = await client.fetch_localities(table_id, lvl)
            if isinstance(payload, list):
                return (lvl, payload)
            try:
                return (lvl, list(payload))
            except Exception:
                return (lvl, [])
        except Exception:
            return (lvl, [])

    results = await asyncio.gather(*(one(l) for l in lvls))
    payloads: dict[str, list] = {k: v for (k, v) in results}
    counts: dict[str, int] = {k: (len(v) if isinstance(v, list) else 0) for k, v in payloads.items()}
    return counts, payloads

async def _subject_gate_with_metadata(
    *,
    entries: list["CatalogEntry"],
    subject_contains: str,
    client: SidraApiClient,
    need: int | None,
    parallel: int,
) -> list["CatalogEntry"]:
    """
    Keep only entries whose SUBJECT matches `subject_contains`.
    Prefer catalog subject; otherwise check local DB; otherwise fetch /{id}/metadados.
    Bounded concurrency, early-stop, and progress ticks.
    """
    q = normalize_basic(subject_contains)
    if not q:
        return entries

    total = len(entries)
    kept: list[CatalogEntry] = []
    done = 0

    print(f"[subject] filtering {total} entries by assunto~{subject_contains!r} (parallel={parallel})")

    async def check_one(e: CatalogEntry) -> tuple[CatalogEntry, bool]:
        # 0) DB cache (already ingested)
        with sqlite_session() as conn:
            row = conn.execute("SELECT assunto FROM agregados WHERE id=?", (e.id,)).fetchone()
        if row and row[0]:
            subj_db = normalize_basic(str(row[0] or ""))
            if subj_db and (q in subj_db):
                return e, True
            # DB had a subject and it didn't match → skip network
            return e, False

        # 1) catalog-provided subject
        subj_cat = normalize_basic(e.assunto or "")
        if subj_cat and (q in subj_cat):
            return e, True

        # 2) fallback: metadata
        per_table_timeout = max(10.0, 2.0 * float(get_settings().request_timeout))
        try:
            md = await asyncio.wait_for(client.fetch_metadata(e.id), timeout=per_table_timeout)
        except Exception:
            return e, False

        subj = md.get("assunto")
        if isinstance(subj, dict):
            subj = subj.get("nome")
        subj = normalize_basic(str(subj or ""))
        return e, (q in subj)

    sem = asyncio.Semaphore(max(1, int(parallel)))
    it = iter(entries)
    pending: set[asyncio.Task] = set()

    async def schedule_next() -> bool:
        try:
            e = next(it)
        except StopIteration:
            return False
        async def _task():
            async with sem:
                return await check_one(e)
        t = asyncio.create_task(_task())
        pending.add(t)
        return True

    # prime
    for _ in range(parallel):
        if not await schedule_next():
            break

    # drain with progress + early-stop
    while pending:
        done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done_set:
            e, ok = await t
            done += 1
            if ok:
                kept.append(e)
            if (done % 10 == 0) or (need and len(kept) >= need):
                pct = (done * 100) // max(1, total)
                print(f"\r[subject] checked={done}/{total} kept={len(kept)} ({pct}%)", end="", flush=True)
            if need and len(kept) >= need:
                pending.clear()
                break
        while len(pending) < parallel:
            more = await schedule_next()
            if not more:
                break
    print()  # newline

    return kept

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

    - Catalog fetch pruned by mentioned levels (cheap).
    - Optional text narrowing on survey (cheap).
    - Optional subject narrowing that falls back to metadata (bounded + early stop).
    - Coverage probe (bounded + early stop).
    """
    ensure_full_schema()
    report = BulkReport()
    
    # Cache per-table locality payloads for levels we probed (e.g., N3/N6)
    prefetched_localities: dict[int, dict[str, list]] = {}

    # Parse coverage and extract hinted levels
    try:
        cov_ast = parse_coverage_expr(coverage)
    except Exception as exc:
        raise RuntimeError(f"Invalid --coverage expression: {coverage!r}: {exc}") from exc
    level_hints = extract_levels(cov_ast)

    need = max(0, int(limit)) if (limit is not None and limit >= 0) else None
    par_probe = max(1, int(probe_concurrent or concurrency))

    async with SidraApiClient() as client:
        # 1) Catalog fetch with hinted levels (server-side pruning)
        entries = await fetch_catalog_entries(client=client, levels=sorted(level_hints) or None)

        # 2) Survey narrowing (accent-insensitive, already implemented in filter_catalog_entries)
        entries = filter_catalog_entries(
            entries,
            require_any_levels=None,
            require_all_levels=None,
            exclude_levels=None,
            subject_contains=None,           # handled below via metadata fallback
            survey_contains=survey_contains,
        )
        if not entries:
            return report

        # 3) Subject narrowing with metadata fallback (bounded, early-stop)
        if subject_contains:
            # probe a bit more than 'need' so the coverage step still has room to filter
            subj_need = need if need is not None else None
            entries = await _subject_gate_with_metadata(
                entries=entries,
                subject_contains=subject_contains,
                client=client,
                need=subj_need,
                parallel=par_probe,
            )
            if not entries:
                return report

        total = len(entries)
        print(f"[probe] candidates={total}, levels={sorted(level_hints) or []}, parallel={par_probe}")

        # 4) Coverage probe (bounded, early-stop)
        kept: list[CatalogEntry] = []
        pending: set[asyncio.Task] = set()
        it = iter(entries)

        async def schedule_next() -> bool:
            try:
                e = next(it)
            except StopIteration:
                return False
            check_levels = (level_hints & e.level_codes) if level_hints else set()
            if not check_levels:
                return True

            async def _one(e_: CatalogEntry) -> tuple[CatalogEntry, bool, dict[str, list]]:
                per_table_timeout = max(10.0, 2.0 * float(get_settings().request_timeout))
                try:
                    counts, payloads = await asyncio.wait_for(
                        _probe_counts_for_levels(client, e_.id, check_levels),
                        timeout=per_table_timeout,
                    )
                    ok = eval_coverage(cov_ast, counts)
                    return e_, ok, payloads if ok else {}
                except Exception:
                    return e_, False, {}

            t = asyncio.create_task(_one(e))
            pending.add(t)
            return True

        for _ in range(par_probe):
            if not await schedule_next():
                break

        probed = 0
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            probed += 1
            e, ok, payloads = await t
            if ok:
                kept.append(e)
                if payloads:
                    prefetched_localities[e.id] = payloads
                if (probed % 10 == 0) or (need and len(kept) >= need):
                    pct = (probed * 100) // max(1, total)
                    inflight = len(pending)
                    print(
                        f"\r[probe] done={probed}/{total} "
                        f"(accepted={len(kept)} • {pct}%) | inflight={inflight}",
                        end="",
                        flush=True,
                    )
                if need and len(kept) >= need:
                    pending.clear()
                    break
            while len(pending) < par_probe:
                more = await schedule_next()
                if not more:
                    break

        print()  # newline after progress

        if need is not None:
            kept = kept[:need]
        report.discovered_ids = [e.id for e in kept]

        # 5) Skip existing + ingest
        with sqlite_session() as conn:
            existing = {int(r[0]) for r in conn.execute("SELECT id FROM agregados")}
        to_do = [e.id for e in kept if e.id not in existing]
        report.scheduled_ids = list(to_do)
        report.skipped_existing = [e.id for e in kept if e.id in existing]

        if not to_do:
            return report

        print(f"[ingest] {len(to_do)} tables (parallel={concurrency})")
        sem = asyncio.Semaphore(max(1, concurrency))

        async def worker(tid: int) -> None:
            async with sem:
                try:
                    await ingest_table(
                        tid,
                        prefetched_localities=prefetched_localities.get(tid),
                    )
                    report.ingested_ids.append(tid)
                    print(f"  ingested {tid}")
                except Exception as exc:
                    report.failed.append((tid, str(exc)[:200]))
                    print(f"  failed {tid}: {exc}")

        await asyncio.gather(*(worker(t) for t in to_do))

    return report
