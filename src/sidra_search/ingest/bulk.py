from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..net.api_client import SidraApiClient
from ..db.session import sqlite_session, ensure_full_schema
from .ingest_table import ingest_table


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


async def ingest_by_coverage(
    *, require_any_levels: Iterable[str] | None = None,
    require_all_levels: Iterable[str] | None = None,
    exclude_levels: Iterable[str] | None = None,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
    limit: int | None = None,
    concurrency: int = 8,
) -> BulkReport:
    ensure_full_schema()
    report = BulkReport()
    async with SidraApiClient() as client:
        entries = await fetch_catalog_entries(
            client=client, levels=[*(require_any_levels or ()), *(require_all_levels or ())]
        )
        entries = filter_catalog_entries(
            entries,
            require_any_levels=require_any_levels,
            require_all_levels=require_all_levels,
            exclude_levels=exclude_levels,
            subject_contains=subject_contains,
            survey_contains=survey_contains,
        )
        if limit is not None and limit >= 0:
            entries = entries[:limit]
        report.discovered_ids = [e.id for e in entries]

        # skip already ingested (optional)
        with sqlite_session() as conn:
            existing = {int(r[0]) for r in conn.execute("SELECT id FROM agregados")}
        to_do = [e.id for e in entries if e.id not in existing]
        report.scheduled_ids = list(to_do)

        sem = asyncio.Semaphore(max(1, concurrency))
        async def worker(tid: int) -> None:
            async with sem:
                try:
                    await ingest_table(tid)
                    report.ingested_ids.append(tid)
                except Exception as exc:
                    report.failed.append((tid, str(exc)[:200]))

        await asyncio.gather(*(worker(t) for t in to_do))
    return report
