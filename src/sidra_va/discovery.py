"""Catalog discovery helpers for SIDRA agregados."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .api_client import SidraApiClient


def _normalize_levels(payload: Any) -> dict[str, list[str]]:
    """Return a mapping of level type -> normalized codes from raw API payload."""

    result: dict[str, list[str]] = {}
    if isinstance(payload, dict):
        for key, values in payload.items():
            values_iter: Iterable[Any]
            if isinstance(values, str):
                values_iter = [values]
            elif isinstance(values, Iterable):
                values_iter = values
            else:
                continue
            normalized = []
            for value in values_iter:
                if isinstance(value, str):
                    normalized.append(value.upper())
                elif isinstance(value, Mapping):
                    code = value.get("codigo") or value.get("nivel") or value.get("id")
                    if isinstance(code, str):
                        normalized.append(code.upper())
            if normalized:
                result[str(key)] = normalized
    elif isinstance(payload, list):
        normalized: list[str] = []
        for value in payload:
            if isinstance(value, str):
                normalized.append(value.upper())
            elif isinstance(value, Mapping):
                code = value.get("codigo") or value.get("nivel") or value.get("id")
                if isinstance(code, str):
                    normalized.append(code.upper())
        if normalized:
            result["codes"] = normalized
    return result


@dataclass(frozen=True)
class CatalogEntry:
    """Lightweight representation of an agregados catalog record."""

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
        """Return all territorial level codes exposed by this agregados."""

        codes: set[str] = set()
        for values in self.nivel_territorial.values():
            for value in values:
                codes.add(value.upper())
        codes.update(self.level_hints)
        return codes

    @classmethod
    def from_api(
        cls,
        payload: dict[str, Any],
        *,
        pesquisa: dict[str, Any] | None = None,
        level_hints: Sequence[str] | None = None,
    ) -> "CatalogEntry":
        """Build a catalog entry from the API payload."""

        level_payload = payload.get("nivelTerritorial")
        nome = payload.get("nome") or payload.get("tabela")
        pesquisa_nome = None
        pesquisa_id = None
        assunto_nome = None
        assunto_id = None
        periodicidade = None

        if pesquisa is None:
            pesquisa = {}

        if isinstance(pesquisa, dict):
            pesquisa_nome = pesquisa.get("pesquisa") or pesquisa.get("nome")
            pesquisa_id = pesquisa.get("idPesquisa") or pesquisa.get("id")
            assunto = pesquisa.get("assunto") or {}
            if isinstance(assunto, dict):
                assunto_nome = assunto.get("nome")
                assunto_id = assunto.get("id")
            else:
                assunto_nome = pesquisa.get("assunto")
                assunto_id = pesquisa.get("idAssunto")
            periodicidade = pesquisa.get("periodicidade")

        return cls(
            id=int(payload.get("id")),
            nome=str(nome) if nome is not None else None,
            pesquisa=str(pesquisa_nome) if pesquisa_nome is not None else None,
            pesquisa_id=int(pesquisa_id) if isinstance(pesquisa_id, int) else None,
            assunto=str(assunto_nome) if assunto_nome is not None else None,
            assunto_id=int(assunto_id) if isinstance(assunto_id, int) else None,
            periodicidade=periodicidade,
            nivel_territorial=_normalize_levels(level_payload),
            level_hints=frozenset(code.upper() for code in level_hints or [] if code),
        )


async def fetch_catalog_entries(
    *,
    client: SidraApiClient | None = None,
    subject_id: int | None = None,
    periodicity: str | None = None,
    levels: Sequence[str] | None = None,
) -> list[CatalogEntry]:
    """Fetch the agregados catalog and normalize it into CatalogEntry rows."""

    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True

    normalized_levels = [code.upper() for code in levels or [] if code]

    try:
        catalog = await client.fetch_catalog(
            subject_id=subject_id,
            periodicity=periodicity,
            levels=normalized_levels or None,
        )
    finally:
        if own_client:
            await client.close()

    entries: list[CatalogEntry] = []
    if not isinstance(catalog, Iterable):
        return entries

    for survey in catalog:
        agregados = None
        if isinstance(survey, dict):
            agregados = survey.get("agregados")
        if not isinstance(agregados, Iterable):
            continue
        for agregado in agregados:
            if not isinstance(agregado, dict):
                continue
            try:
                entry = CatalogEntry.from_api(
                    agregado,
                    pesquisa=survey,
                    level_hints=normalized_levels,
                )
            except Exception:  # noqa: BLE001
                continue
            entries.append(entry)
    return entries


def filter_catalog_entries(
    entries: Sequence[CatalogEntry],
    *,
    require_any_levels: Iterable[str] | None = None,
    require_all_levels: Iterable[str] | None = None,
    exclude_levels: Iterable[str] | None = None,
    subject_contains: str | None = None,
    survey_contains: str | None = None,
) -> list[CatalogEntry]:
    """Filter CatalogEntry items by territorial coverage and optional metadata."""

    any_levels = {code.upper() for code in require_any_levels or ()}
    all_levels = {code.upper() for code in require_all_levels or ()}
    excluded = {code.upper() for code in exclude_levels or ()}
    subject_query = subject_contains.lower() if subject_contains else None
    survey_query = survey_contains.lower() if survey_contains else None

    filtered: list[CatalogEntry] = []
    for entry in entries:
        codes = entry.level_codes
        if any_levels and not (codes & any_levels):
            continue
        if all_levels and not all_levels.issubset(codes):
            continue
        if excluded and (codes & excluded):
            continue
        if subject_query and (entry.assunto or "").lower().find(subject_query) == -1:
            continue
        if survey_query and (entry.pesquisa or "").lower().find(survey_query) == -1:
            continue
        filtered.append(entry)
    return filtered


__all__ = [
    "CatalogEntry",
    "fetch_catalog_entries",
    "filter_catalog_entries",
]
