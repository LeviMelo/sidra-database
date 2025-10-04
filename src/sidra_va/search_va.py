from __future__ import annotations

import asyncio
import json
import math
from array import array
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .db import create_connection
from .embedding_client import EmbeddingClient
from .schema_migrations import apply_va_schema
from .scoring import DEFAULT_WEIGHTS, StructureMatch, rrf
from .synonyms import SynonymMap, load_synonyms_into_memory, normalize_basic


@dataclass(frozen=True)
class VaSearchFilters:
    require_levels: tuple[str, ...] = ()
    period_start: int | None = None
    period_end: int | None = None
    must_variable_ids: tuple[int, ...] = ()
    must_variable_names: tuple[str, ...] = ()
    must_classification_names: tuple[str, ...] = ()
    must_category_names: tuple[str, ...] = ()
    min_municipalities: int | None = None
    requires_national_munis: bool = False


@dataclass(frozen=True)
class VaResult:
    va_id: str
    agregado_id: int
    variable_id: int
    title: str
    text: str
    score: float
    rrf_score: float
    struct_score: float
    metadata: Mapping[str, str]
    why: str


def _tokenize_query(text: str, synonyms: SynonymMap | None) -> list[str]:
    words = text.split()
    return [normalize_basic(word) for word in words if normalize_basic(word)]


def _build_fts_query(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return " ".join(tokens)


def _blob_to_vector(blob: bytes, dimension: int) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    values = list(arr)
    if dimension and len(values) > dimension:
        values = values[:dimension]
    return values


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def search_value_atoms(
    query: str,
    *,
    filters: VaSearchFilters | None = None,
    limit: int = 20,
    weights: Mapping[str, float] | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> list[VaResult]:
    filters = filters or VaSearchFilters()
    weight_cfg = {**DEFAULT_WEIGHTS, **(weights or {})}

    conn = create_connection()
    try:
        apply_va_schema(conn)
        synonyms = load_synonyms_into_memory(conn)

        tokens = _tokenize_query(query, synonyms)
        fts_query = _build_fts_query(tokens)
        lexical_candidates: list[str] = []
        if fts_query:
            cursor = conn.execute(
                "SELECT va_id FROM value_atoms_fts WHERE value_atoms_fts MATCH ? LIMIT ?",
                (fts_query, limit * 5),
            )
            lexical_candidates = [row[0] for row in cursor.fetchall()]
        lexical_ranks = {va_id: idx + 1 for idx, va_id in enumerate(lexical_candidates)}

        semantic_ranks: Dict[str, int] = {}
        query_vector: list[float] | None = None
        if embedding_client is not None:
            query_vector = await asyncio.to_thread(
                embedding_client.embed_text,
                query,
                model=embedding_client.model,
            )
            cursor = conn.execute(
                """
                SELECT entity_id, agregado_id, dimension, vector
                FROM embeddings
                WHERE entity_type = 'va' AND model = ?
                """,
                (embedding_client.model,),
            )
            similarities: list[tuple[str, float]] = []
            for row in cursor.fetchall():
                vector = _blob_to_vector(row["vector"], row["dimension"])
                sim = _cosine_similarity(query_vector, vector)
                if sim > 0:
                    similarities.append((row["entity_id"], sim))
            similarities.sort(key=lambda item: item[1], reverse=True)
            semantic_ranks = {va_id: idx + 1 for idx, (va_id, _) in enumerate(similarities[: limit * 5])}

        all_candidate_ids = set(lexical_ranks) | set(semantic_ranks)
        if not all_candidate_ids:
            return []

        placeholder = ",".join("?" for _ in all_candidate_ids)
        cursor = conn.execute(
            f"""
            SELECT va.va_id, va.agregado_id, va.variable_id, va.unit, va.text, va.dims_json,
                   va.has_n1, va.has_n2, va.has_n3, va.has_n6,
                   va.period_start, va.period_end,
                   ag.municipality_locality_count, ag.covers_national_municipalities,
                   ag.nome AS table_title, ag.pesquisa AS survey, ag.assunto AS subject,
                   var.nome AS variable_name, var.unidade AS variable_unit
            FROM value_atoms AS va
            JOIN agregados AS ag ON ag.id = va.agregado_id
            JOIN variables AS var ON var.id = va.variable_id
            WHERE va.va_id IN ({placeholder})
            """,
            tuple(all_candidate_ids),
        )
        rows = cursor.fetchall()

        dims_map: Dict[str, list[Mapping[str, str]]] = {}
        cursor = conn.execute(
            f"SELECT va_id, classification_name, category_name FROM value_atom_dims WHERE va_id IN ({placeholder})",
            tuple(all_candidate_ids),
        )
        for va_id, class_name, category_name in cursor.fetchall():
            dims_map.setdefault(va_id, []).append(
                {"classification_name": class_name, "category_name": category_name}
            )
    finally:
        conn.close()

    query_tokens = set(tokens)
    results: list[VaResult] = []
    for row in rows:
        dims = dims_map.get(row["va_id"], [])
        if not _passes_filters(row, dims, filters):
            continue
        struct = _compute_structure_score(row, dims, filters, query_tokens)
        struct_score = struct.score()
        rrf_scores = 0.0
        rank_inputs = {}
        if row["va_id"] in lexical_ranks:
            rank_inputs[row["va_id"]] = lexical_ranks[row["va_id"]]
        if row["va_id"] in semantic_ranks:
            rank_inputs[row["va_id"]] = min(
                semantic_ranks[row["va_id"]], rank_inputs.get(row["va_id"], semantic_ranks[row["va_id"]])
            )
        if rank_inputs:
            rrf_scores = sum(rrf(rank_inputs).values())
        combined = weight_cfg.get("struct", 0.7) * struct_score + weight_cfg.get("rrf", 0.3) * rrf_scores
        title = row["variable_name"]
        if dims:
            dim_parts = [f"{d['classification_name']}={d['category_name']}" for d in dims]
            title += " | " + ", ".join(dim_parts)
        if row["unit"]:
            title += f" ({row['unit']})"
        metadata = {
            "period_start": row["period_start"] or "",
            "period_end": row["period_end"] or "",
            "levels": ",".join(_levels_from_row(row)),
            "survey": row["survey"] or "",
            "subject": row["subject"] or "",
            "table_title": row["table_title"] or "",
        }
        why = _explain_match(row, dims, filters, query_tokens)
        results.append(
            VaResult(
                va_id=row["va_id"],
                agregado_id=row["agregado_id"],
                variable_id=row["variable_id"],
                title=title,
                text=row["text"],
                score=combined,
                rrf_score=rrf_scores,
                struct_score=struct_score,
                metadata=metadata,
                why=why,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


def _levels_from_row(row) -> list[str]:
    levels = []
    if row["has_n1"]:
        levels.append("N1")
    if row["has_n2"]:
        levels.append("N2")
    if row["has_n3"]:
        levels.append("N3")
    if row["has_n6"]:
        levels.append("N6")
    return levels


def _passes_filters(row, dims, filters: VaSearchFilters) -> bool:
    levels = set(_levels_from_row(row))
    for level in filters.require_levels:
        if level not in levels:
            return False

    start = row["period_start"]
    end = row["period_end"]
    if filters.period_start is not None and start:
        if int(end or start) < filters.period_start:
            return False
    if filters.period_end is not None and end:
        if int(start or end) > filters.period_end:
            return False

    if filters.must_variable_ids and row["variable_id"] not in filters.must_variable_ids:
        return False

    if filters.must_variable_names:
        var_norm = normalize_basic(row["variable_name"])
        if not any(var_norm == normalize_basic(name) for name in filters.must_variable_names):
            return False

    dim_categories = {normalize_basic(d["category_name"]) for d in dims}
    if filters.must_category_names:
        wanted = {normalize_basic(name) for name in filters.must_category_names}
        if not wanted.issubset(dim_categories):
            return False

    dim_classes = {normalize_basic(d["classification_name"]) for d in dims}
    if filters.must_classification_names:
        wanted = {normalize_basic(name) for name in filters.must_classification_names}
        if not wanted.issubset(dim_classes):
            return False

    if filters.min_municipalities is not None:
        if (row["municipality_locality_count"] or 0) < filters.min_municipalities:
            return False

    if filters.requires_national_munis and row["covers_national_municipalities"] != 1:
        return False

    return True


def _compute_structure_score(row, dims, filters: VaSearchFilters, query_tokens: set[str]) -> StructureMatch:
    var_norm = normalize_basic(row["variable_name"])
    variable_score = 0.0
    if filters.must_variable_ids and row["variable_id"] in filters.must_variable_ids:
        variable_score = 1.0
    elif filters.must_variable_names:
        wanted = {normalize_basic(name) for name in filters.must_variable_names}
        if var_norm in wanted:
            variable_score = 1.0
    else:
        if var_norm and var_norm in query_tokens:
            variable_score = 1.0
        elif var_norm and any(tok in var_norm for tok in query_tokens):
            variable_score = 0.6

    unit = row["unit"] or row["variable_unit"] or ""
    unit_norm = normalize_basic(unit)
    unit_score = 0.0
    if unit_norm and unit_norm in query_tokens:
        unit_score = 1.0
    elif unit:
        unit_score = 0.5

    dim_norms = {normalize_basic(d["category_name"]): d for d in dims}
    dim_score = 0.0
    if filters.must_category_names:
        wanted = {normalize_basic(name) for name in filters.must_category_names}
        if wanted:
            matched = len([name for name in wanted if name in dim_norms])
            dim_score = matched / len(wanted) if wanted else 0.0
    elif query_tokens:
        matches = len([tok for tok in query_tokens if tok in dim_norms])
        if matches:
            dim_score = min(1.0, matches / max(1, len(dim_norms) or 1))

    period_score = 0.5
    if filters.period_start or filters.period_end:
        start = int(row["period_start"] or row["period_end"] or 0)
        end = int(row["period_end"] or row["period_start"] or 0)
        window_start = filters.period_start or start
        window_end = filters.period_end or end
        latest_start = max(start, window_start)
        earliest_end = min(end, window_end)
        if latest_start <= earliest_end:
            period_score = 1.0
        else:
            period_score = 0.0

    geo_score = 0.5
    if filters.require_levels:
        levels = set(_levels_from_row(row))
        matched = len([lvl for lvl in filters.require_levels if lvl in levels])
        geo_score = matched / len(filters.require_levels)
    elif query_tokens:
        levels = set(_levels_from_row(row))
        tokens_levels = {tok for tok in query_tokens if tok.upper().startswith("N")}
        if tokens_levels:
            matched = len(tokens_levels & levels)
            geo_score = matched / len(tokens_levels)

    return StructureMatch(variable_score, unit_score, dim_score, period_score, geo_score)


def _explain_match(row, dims, filters: VaSearchFilters, query_tokens: set[str]) -> str:
    parts = [f"var={row['variable_name']}"]
    if row["unit"]:
        parts.append(f"unit={row['unit']}")
    if dims:
        dim_parts = [f"{d['classification_name']}: {d['category_name']}" for d in dims]
        parts.append("class=" + "; ".join(dim_parts))
    levels = _levels_from_row(row)
    if levels:
        parts.append("levels=" + ",".join(levels))
    if row["period_start"] or row["period_end"]:
        parts.append(f"period={row['period_start'] or ''}-{row['period_end'] or ''}")
    if filters.require_levels:
        parts.append("require_levels=" + ",".join(filters.require_levels))
    if filters.must_category_names:
        parts.append("filter_category=" + ",".join(filters.must_category_names))
    return "; ".join(parts)


__all__ = [
    "VaSearchFilters",
    "VaResult",
    "search_value_atoms",
]
