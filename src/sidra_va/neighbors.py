from __future__ import annotations

from typing import Mapping

from .db import create_connection
from .schema_migrations import apply_va_schema
from .search_va import VaResult
from .synonyms import normalize_basic


def _levels(row) -> set[str]:
    levels = set()
    if row["has_n1"]:
        levels.add("N1")
    if row["has_n2"]:
        levels.add("N2")
    if row["has_n3"]:
        levels.add("N3")
    if row["has_n6"]:
        levels.add("N6")
    return levels


def _period_range(row) -> tuple[int, int]:
    start = int(row["period_start"] or row["period_end"] or 0)
    end = int(row["period_end"] or row["period_start"] or 0)
    if start and end and start > end:
        start, end = end, start
    return start, end


def _load_dims(conn, va_ids: list[str]) -> dict[str, list[Mapping[str, str]]]:
    if not va_ids:
        return {}
    placeholder = ",".join("?" for _ in va_ids)
    cursor = conn.execute(
        f"SELECT va_id, classification_name, category_name FROM value_atom_dims WHERE va_id IN ({placeholder})",
        tuple(va_ids),
    )
    dims: dict[str, list[Mapping[str, str]]] = {}
    for va_id, class_name, category_name in cursor.fetchall():
        dims.setdefault(va_id, []).append(
            {"classification_name": class_name, "category_name": category_name}
        )
    return dims


def _compat_score(seed, candidate, seed_dims, candidate_dims, require_same_unit: bool) -> float:
    var_compat = 1.0 if seed["variable_id"] == candidate["variable_id"] else 0.9 if seed["fingerprint"] == candidate["fingerprint"] else 0.0
    unit_seed = seed["unit"] or ""
    unit_candidate = candidate["unit"] or ""
    if require_same_unit and normalize_basic(unit_seed) != normalize_basic(unit_candidate):
        return 0.0
    unit_compat = 1.0 if normalize_basic(unit_seed) == normalize_basic(unit_candidate) else 0.5 if unit_seed and unit_candidate else 0.0

    seed_cats = {normalize_basic(dim["category_name"]) for dim in seed_dims if dim["category_name"]}
    cand_cats = {
        normalize_basic(dim["category_name"]) for dim in candidate_dims if dim["category_name"]
    }
    if seed_cats or cand_cats:
        union = seed_cats | cand_cats
        dim_compat = (len(seed_cats & cand_cats) / len(union)) if union else 1.0
    else:
        dim_compat = 1.0

    seed_levels = _levels(seed)
    candidate_levels = _levels(candidate)
    if seed_levels:
        geo_compat = len(seed_levels & candidate_levels) / len(seed_levels)
    else:
        geo_compat = 0.5

    seed_start, seed_end = _period_range(seed)
    cand_start, cand_end = _period_range(candidate)
    latest_start = max(seed_start, cand_start)
    earliest_end = min(seed_end, cand_end)
    period_compat = 1.0 if latest_start <= earliest_end else 0.0

    return (
        0.45 * var_compat
        + 0.20 * unit_compat
        + 0.20 * dim_compat
        + 0.10 * geo_compat
        + 0.05 * period_compat
    )


def find_neighbors_for_va(
    seed_va_id: str,
    *,
    top_k: int = 50,
    require_same_unit: bool = True,
) -> list[tuple[VaResult, float]]:
    conn = create_connection()
    try:
        apply_va_schema(conn)
        cursor = conn.execute(
            """
            SELECT va.va_id, va.agregado_id, va.variable_id, va.unit, va.text, va.dims_json,
                   va.has_n1, va.has_n2, va.has_n3, va.has_n6,
                   va.period_start, va.period_end,
                   ag.nome AS table_title, ag.pesquisa AS survey, ag.assunto AS subject,
                   vf.fingerprint AS fingerprint,
                   var.nome AS variable_name, var.unidade AS variable_unit
            FROM value_atoms AS va
            JOIN agregados AS ag ON ag.id = va.agregado_id
            JOIN variables AS var
                ON var.id = va.variable_id AND var.agregado_id = va.agregado_id
            LEFT JOIN variable_fingerprints AS vf ON vf.variable_id = va.variable_id
            WHERE va.va_id = ?
            """,
            (seed_va_id,),
        )
        seed_row = cursor.fetchone()
        if not seed_row:
            return []
        seed_dims_map = _load_dims(conn, [seed_va_id])
        seed_dims = seed_dims_map.get(seed_va_id, [])

        cursor = conn.execute(
            """
            SELECT va.va_id, va.agregado_id, va.variable_id, va.unit, va.text, va.dims_json,
                   va.has_n1, va.has_n2, va.has_n3, va.has_n6,
                   va.period_start, va.period_end,
                   ag.nome AS table_title, ag.pesquisa AS survey, ag.assunto AS subject,
                   vf.fingerprint AS fingerprint,
                   var.nome AS variable_name, var.unidade AS variable_unit
            FROM value_atoms AS va
            JOIN agregados AS ag ON ag.id = va.agregado_id
            JOIN variables AS var ON var.id = va.variable_id
            LEFT JOIN variable_fingerprints AS vf ON vf.variable_id = va.variable_id
            WHERE va.va_id != ? AND (va.variable_id = ? OR vf.fingerprint = ?)
            """,
            (seed_va_id, seed_row["variable_id"], seed_row["fingerprint"]),
        )
        candidates = cursor.fetchall()
        dims_map = _load_dims(conn, [row["va_id"] for row in candidates])
    finally:
        conn.close()

    neighbors: list[tuple[VaResult, float]] = []
    for cand in candidates:
        cand_dims = dims_map.get(cand["va_id"], [])
        score = _compat_score(seed_row, cand, seed_dims, cand_dims, require_same_unit)
        if score <= 0:
            continue
        title = cand["variable_name"]
        if cand_dims:
            title += " | " + ", ".join(
                f"{dim['classification_name']}={dim['category_name']}" for dim in cand_dims
            )
        if cand["unit"]:
            title += f" ({cand['unit']})"
        metadata = {
            "survey": cand["survey"] or "",
            "subject": cand["subject"] or "",
            "table_title": cand["table_title"] or "",
            "period_start": cand["period_start"] or "",
            "period_end": cand["period_end"] or "",
        }
        why = f"compat={score:.2f}; variable={'same' if cand['variable_id']==seed_row['variable_id'] else 'fingerprint'}"
        neighbors.append(
            (
                VaResult(
                    va_id=cand["va_id"],
                    agregado_id=cand["agregado_id"],
                    variable_id=cand["variable_id"],
                    title=title,
                    text=cand["text"],
                    score=score,
                    rrf_score=0.0,
                    struct_score=0.0,
                    metadata=metadata,
                    why=why,
                ),
                score,
            )
        )

    neighbors.sort(key=lambda item: item[1], reverse=True)
    return neighbors[:top_k]


__all__ = ["find_neighbors_for_va"]
