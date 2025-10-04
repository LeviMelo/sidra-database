"""Helpers to enumerate stored agregados and coverage metadata."""
from __future__ import annotations

from dataclasses import dataclass

from .db import sqlite_session


@dataclass(frozen=True)
class AgregadoRecord:
    """Lightweight snapshot of an agregados entry."""

    id: int
    nome: str
    assunto: str | None
    pesquisa: str | None
    municipality_locality_count: int
    covers_national_municipalities: bool
    fetched_at: str


def list_agregados(
    *,
    requires_national_munis: bool = False,
    min_municipality_count: int | None = None,
    limit: int | None = None,
    order_by: str = "municipalities",
) -> list[AgregadoRecord]:
    """Return agregados rows matching basic coverage filters."""

    valid_order = {
        "municipalities": "municipality_locality_count DESC, id ASC",
        "id": "id ASC",
        "name": "nome COLLATE NOCASE ASC",
        "fetched": "fetched_at DESC",
    }
    if order_by not in valid_order:
        raise ValueError(f"Unsupported order_by value: {order_by}")

    conditions: list[str] = []
    params: list[object] = []
    if requires_national_munis:
        conditions.append("covers_national_municipalities = 1")
    if min_municipality_count is not None:
        conditions.append("municipality_locality_count >= ?")
        params.append(int(min_municipality_count))

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None and limit > 0 else ""

    sql = (
        "SELECT id, nome, assunto, pesquisa, municipality_locality_count, "
        "covers_national_municipalities, fetched_at FROM agregados "
        f"{where_clause} ORDER BY {valid_order[order_by]} {limit_clause}"
    ).strip()

    rows: list[AgregadoRecord] = []
    with sqlite_session() as conn:
        for row in conn.execute(sql, params):
            rows.append(
                AgregadoRecord(
                    id=row["id"],
                    nome=row["nome"],
                    assunto=row["assunto"],
                    pesquisa=row["pesquisa"],
                    municipality_locality_count=int(row["municipality_locality_count"] or 0),
                    covers_national_municipalities=bool(row["covers_national_municipalities"]),
                    fetched_at=row["fetched_at"],
                )
            )
    return rows


__all__ = ["AgregadoRecord", "list_agregados"]
