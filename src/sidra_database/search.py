"""Semantic search helpers over stored embeddings."""
from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import math
import re
import unicodedata
from typing import Mapping, Sequence

from .db import sqlite_session
from .embedding import EmbeddingClient


@dataclass(frozen=True)
class SemanticMatch:
    """Result item returned by semantic_search."""

    entity_type: str
    entity_id: str
    agregado_id: int | None
    score: float
    model: str


@dataclass(frozen=True)
class SemanticResult:
    """Semantic match enriched with human-readable metadata."""

    entity_type: str
    entity_id: str
    agregado_id: int | None
    score: float
    model: str
    title: str
    description: str | None
    metadata: Mapping[str, str]
    lexical_score: float = 0.0
    combined_score: float = 0.0
    lexical_table_score: float = 0.0
    lexical_children_score: float = 0.0
    child_matches: Sequence["ChildMatch"] = ()


@dataclass(frozen=True)
class SearchFilters:
    """Structured filters to refine search results."""

    min_municipalities: int | None = None
    requires_national_munis: bool = False
    subject_contains: str | None = None
    survey_contains: str | None = None
    period_start: int | None = None
    period_end: int | None = None
    has_variables: Sequence[int] | None = None
    has_classifications: Sequence[int] | None = None
    has_categories: Sequence[tuple[int, int]] | None = None


@dataclass(frozen=True)
class ChildMatch:
    """Lexical evidence from child entities used to justify a table hit."""

    entity_type: str
    entity_id: str
    title: str
    lexical_score: float
    metadata: Mapping[str, str]


def _decode_vector(blob: bytes, dimension: int) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    if len(values) < dimension:
        return []
    if len(values) > dimension:
        values = values[:dimension]
    return [float(v) for v in values]


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(query: Sequence[float], candidate: Sequence[float], query_norm: float) -> float:
    candidate_norm = _vector_norm(candidate)
    if query_norm == 0 or candidate_norm == 0:
        return 0.0
    dot = sum(q * c for q, c in zip(query, candidate))
    return dot / (query_norm * candidate_norm)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if ord(ch) < 128)


def _tokenize(text: str) -> list[str]:
    folded = _strip_accents(text.lower())
    tokens = [token for token in re.findall(r"\w+", folded) if len(token) >= 3]
    return tokens


def _normalize_substring(value: str | None) -> str:
    return (value or "").strip().lower()


def semantic_search(
    query: str,
    *,
    entity_types: Sequence[str] | None = None,
    limit: int = 10,
    embedding_client: EmbeddingClient | None = None,
    model: str | None = None,
) -> list[SemanticMatch]:
    """Return the best-matching stored embedding rows for the given query text."""

    if limit <= 0:
        return []

    client = embedding_client or EmbeddingClient(model=model)
    model_name = model or client.model
    query_vector = [float(value) for value in client.embed_text(query, model=model_name)]
    if not query_vector:
        return []

    query_norm = _vector_norm(query_vector)
    if query_norm == 0:
        return []

    sql = (
        "SELECT entity_type, entity_id, agregado_id, model, dimension, vector "
        "FROM embeddings WHERE model = ?"
    )
    params: list[object] = [model_name]
    if entity_types:
        placeholders = ", ".join("?" for _ in entity_types)
        sql += f" AND entity_type IN ({placeholders})"
        params.extend(entity_types)

    matches: list[SemanticMatch] = []
    with sqlite_session() as conn:
        for row in conn.execute(sql, params):
            candidate = _decode_vector(row["vector"], row["dimension"])
            if not candidate or len(candidate) != len(query_vector):
                continue
            score = _cosine_similarity(query_vector, candidate, query_norm)
            matches.append(
                SemanticMatch(
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    agregado_id=row["agregado_id"],
                    score=score,
                    model=row["model"],
                )
            )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:limit]


def _filters_to_sql(
    filters: SearchFilters | None,
    *,
    table_alias: str | None = None,
) -> tuple[list[str], list[object]]:
    if filters is None:
        return [], []
    conditions: list[str] = []
    params: list[object] = []
    prefix = f"{table_alias}." if table_alias else ""
    id_ref = f"{table_alias}.id" if table_alias else "id"
    if filters.min_municipalities is not None:
        conditions.append(f"{prefix}municipality_locality_count >= ?")
        params.append(int(filters.min_municipalities))
    if filters.requires_national_munis:
        conditions.append(f"{prefix}covers_national_municipalities = 1")
    if filters.subject_contains:
        conditions.append(f"LOWER({prefix}assunto) LIKE ?")
        params.append(f"%{filters.subject_contains.lower()}%")
    if filters.survey_contains:
        conditions.append(f"LOWER({prefix}pesquisa) LIKE ?")
        params.append(f"%{filters.survey_contains.lower()}%")
    if filters.period_start is not None:
        conditions.append(
            f"({prefix}periodo_fim IS NULL OR CAST({prefix}periodo_fim AS INTEGER) >= ?)"
        )
        params.append(int(filters.period_start))
    if filters.period_end is not None:
        conditions.append(
            f"({prefix}periodo_inicio IS NULL OR CAST({prefix}periodo_inicio AS INTEGER) <= ?)"
        )
        params.append(int(filters.period_end))
    for variable_id in filters.has_variables or ():
        conditions.append(
            f"EXISTS (SELECT 1 FROM variables v WHERE v.agregado_id = {id_ref} AND v.id = ?)"
        )
        params.append(int(variable_id))
    for classification_id in filters.has_classifications or ():
        conditions.append(
            f"EXISTS (SELECT 1 FROM classifications c WHERE c.agregado_id = {id_ref} AND c.id = ?)"
        )
        params.append(int(classification_id))
    for category in filters.has_categories or ():
        classification_id, category_id = category
        conditions.append(
            "EXISTS (SELECT 1 FROM categories cat WHERE cat.agregado_id = {id_ref} "
            "AND cat.classification_id = ? AND cat.categoria_id = ?)".format(id_ref=id_ref)
        )
        params.extend([int(classification_id), int(category_id)])
    return conditions, params


def _lexical_candidates(
    conn,
    tokens: Sequence[str],
    filters: SearchFilters | None,
    limit: int,
) -> list[tuple[int, float]]:
    if not tokens or limit <= 0:
        return []

    conditions, params = _filters_to_sql(filters, table_alias="a")
    token_clauses: list[str] = []
    for token in tokens:
        pattern = f"%{token}%"
        token_clauses.append(
            "(LOWER(a.nome) LIKE ? OR LOWER(a.pesquisa) LIKE ? OR LOWER(a.assunto) LIKE ?)"
        )
        params.extend([pattern, pattern, pattern])

    where_parts = conditions[:]
    if token_clauses:
        where_parts.append(" AND ".join(token_clauses))

    sql = "SELECT a.id, a.nome, a.pesquisa, a.assunto FROM agregados a"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += " LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    results: list[tuple[int, float]] = []
    for row in rows:
        blob = " ".join(
            part for part in [row["nome"], row["pesquisa"], row["assunto"]] if part
        ).lower()
        if not blob:
            continue
        hits = sum(1 for token in tokens if token in blob)
        if hits == 0:
            continue
        lexical_score = hits / len(tokens)
        results.append((int(row["id"]), float(lexical_score)))
    return results


def _compute_lexical_score(
    tokens: Sequence[str],
    text_parts: Sequence[str],
) -> float:
    if not tokens:
        return 0.0
    blob = " ".join(part for part in text_parts if part).lower()
    if not blob:
        return 0.0
    hits = sum(1 for token in tokens if token in blob)
    return hits / len(tokens)


def _combine_scores(semantic_score: float, lexical_score: float) -> float:
    if semantic_score <= 0 and lexical_score <= 0:
        return 0.0
    if semantic_score <= 0:
        return 0.45 * lexical_score
    if lexical_score <= 0:
        return semantic_score
    return 0.65 * semantic_score + 0.35 * lexical_score


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "sem": 0.62,
    "lex_table": 0.18,
    "lex_children": 0.20,
}


def _resolve_weights(overrides: Mapping[str, float] | None) -> Mapping[str, float]:
    if not overrides:
        return DEFAULT_WEIGHTS
    merged = dict(DEFAULT_WEIGHTS)
    for key, value in overrides.items():
        if key in merged and value >= 0:
            merged[key] = float(value)
    return merged


def _lexical_children(
    conn,
    tokens: Sequence[str],
    filters: SearchFilters | None,
    limit: int,
    child_types: Sequence[str],
) -> list[tuple[int, ChildMatch]]:
    if not tokens or not child_types or limit <= 0:
        return []

    overfetch = max(limit, 50)
    results: list[tuple[int, ChildMatch]] = []
    normalized_types = [child.lower() for child in child_types]

    if "variable" in normalized_types:
        results.extend(
            _lexical_child_query(
                conn,
                tokens,
                filters,
                overfetch,
                sql="""
                    SELECT v.agregado_id AS table_id,
                           v.id AS child_id,
                           v.nome,
                           v.unidade,
                           NULL AS classification_id,
                           NULL AS category_id
                    FROM variables v
                    JOIN agregados a ON a.id = v.agregado_id
                """,
                fields=["v.nome", "v.unidade"],
                entity_type="variable",
            )
        )

    if "classification" in normalized_types:
        results.extend(
            _lexical_child_query(
                conn,
                tokens,
                filters,
                overfetch,
                sql="""
                    SELECT c.agregado_id AS table_id,
                           c.id AS child_id,
                           c.nome,
                           NULL AS unidade,
                           NULL AS classification_id,
                           NULL AS category_id
                    FROM classifications c
                    JOIN agregados a ON a.id = c.agregado_id
                """,
                fields=["c.nome"],
                entity_type="classification",
            )
        )

    if "category" in normalized_types:
        results.extend(
            _lexical_child_query(
                conn,
                tokens,
                filters,
                overfetch,
                sql="""
                    SELECT cat.agregado_id AS table_id,
                           cat.categoria_id AS child_id,
                           cat.nome,
                           cat.unidade,
                           cat.classification_id,
                           NULL AS category_id
                    FROM categories cat
                    JOIN agregados a ON a.id = cat.agregado_id
                """,
                fields=["cat.nome", "cat.unidade"],
                entity_type="category",
            )
        )

    return results


def _lexical_child_query(
    conn,
    tokens: Sequence[str],
    filters: SearchFilters | None,
    limit: int,
    sql: str,
    fields: Sequence[str],
    entity_type: str,
) -> list[tuple[int, ChildMatch]]:
    conditions, params = _filters_to_sql(filters, table_alias="a")
    token_clauses: list[str] = []
    for token in tokens:
        pattern = f"%{token}%"
        field_clauses = [f"LOWER({field}) LIKE ?" for field in fields]
        token_clauses.append("(" + " OR ".join(field_clauses) + ")")
        params.extend([pattern] * len(fields))

    where_segments = conditions[:]
    if token_clauses:
        where_segments.append(" AND ".join(token_clauses))

    query = sql
    if where_segments:
        query += " WHERE " + " AND ".join(where_segments)
    query += " LIMIT ?"
    params.append(int(limit))

    rows = conn.execute(query, params).fetchall()
    hits: list[tuple[int, ChildMatch]] = []
    for row in rows:
        name = row["nome"] or ""
        keys = set(row.keys())
        unit = row["unidade"] if "unidade" in keys else None
        score = _compute_lexical_score(tokens, [name, unit or ""])
        if score <= 0:
            continue
        metadata: dict[str, str] = {}
        if unit:
            metadata["unit"] = str(unit)
        classification_id = row["classification_id"] if "classification_id" in keys else None
        if classification_id is not None:
            metadata["classification_id"] = str(classification_id)
        child_id = str(row["child_id"])
        match = ChildMatch(
            entity_type=entity_type,
            entity_id=child_id,
            title=str(name).strip() or f"{entity_type.title()} {child_id}",
            lexical_score=float(score),
            metadata=metadata,
        )
        hits.append((int(row["table_id"]), match))
    return hits


def _aggregate_children_by_table(
    child_hits: Sequence[tuple[int, ChildMatch]],
    tokens: Sequence[str],
    max_matches: int,
) -> dict[int, dict[str, object]]:
    if not child_hits:
        return {}

    aggregated: dict[int, dict[str, object]] = {}
    for table_id, match in child_hits:
        bucket = aggregated.setdefault(table_id, {"matches": []})
        bucket["matches"].append(match)

    token_count = max(1, len(tokens))
    for table_id, payload in aggregated.items():
        matches: list[ChildMatch] = sorted(
            payload["matches"],
            key=lambda item: item.lexical_score,
            reverse=True,
        )
        top_matches = matches[:max_matches] if max_matches > 0 else matches
        total = sum(item.lexical_score for item in top_matches)
        denom = max(1, len(top_matches))
        payload["score"] = total / denom
        payload["matches"] = top_matches
    return aggregated


def _build_agregado_summary(row) -> tuple[str, str | None, dict[str, str]]:
    title = str(row["nome"] or "").strip() or f"Table {row['id']}"
    parts: list[str] = []
    metadata: dict[str, str] = {"table_id": str(row["id"])}
    if row["pesquisa"]:
        metadata["survey"] = row["pesquisa"]
        parts.append(str(row["pesquisa"]))
    if row["assunto"]:
        metadata["subject"] = row["assunto"]
        parts.append(str(row["assunto"]))
    freq = row["freq"]
    if freq:
        metadata["frequency"] = str(freq)
        parts.append(f"Frequency: {freq}")
    start = row["periodo_inicio"]
    end = row["periodo_fim"]
    if start or end:
        metadata["period_start"] = str(start) if start is not None else ""
        metadata["period_end"] = str(end) if end is not None else ""
        if start and end and start != end:
            parts.append(f"Period {start}–{end}")
        elif start or end:
            parts.append(f"Period {start or end}")
    if row["url"]:
        metadata["url"] = row["url"]
    muni_count = row["municipality_locality_count"] if "municipality_locality_count" in row.keys() else None
    if muni_count is not None and int(muni_count) > 0:
        metadata["municipality_locality_count"] = str(int(muni_count))
        parts.append(f"Municipalities: {int(muni_count):,}")
    covers_national = (
        row["covers_national_municipalities"] if "covers_national_municipalities" in row.keys() else None
    )
    if covers_national is not None:
        metadata["covers_national_municipalities"] = str(int(covers_national))
        if int(covers_national):
            parts.append("National municipal coverage")
    description = " | ".join(filter(None, (part.strip() for part in parts))) or None
    return title, description, metadata


def _build_variable_summary(row) -> tuple[str, str | None, dict[str, str]]:
    variable_name = str(row["nome"] or "").strip() or f"Variable {row['id']}"
    title = f"{variable_name}"
    parts: list[str] = []
    metadata: dict[str, str] = {
        "variable_id": str(row["id"]),
        "table_id": str(row["agregado_id"]),
    }
    if row["agregado_nome"]:
        parts.append(f"Table {row['agregado_id']}: {row['agregado_nome']}")
        metadata["table_name"] = row["agregado_nome"]
    if row["unidade"]:
        parts.append(f"Unit: {row['unidade']}")
        metadata["unit"] = row["unidade"]
    if row["pesquisa"]:
        metadata["survey"] = row["pesquisa"]
    if "municipality_locality_count" in row.keys() and int(row["municipality_locality_count"] or 0) > 0:
        metadata["municipality_locality_count"] = str(int(row["municipality_locality_count"]))
        parts.append(f"Municipalities: {int(row['municipality_locality_count']):,}")
    if "covers_national_municipalities" in row.keys() and int(row["covers_national_municipalities"] or 0):
        metadata["covers_national_municipalities"] = str(int(row["covers_national_municipalities"]))
        parts.append("National municipal coverage")
    description = " | ".join(parts) or None
    return title, description, metadata


def _build_classification_summary(row) -> tuple[str, str | None, dict[str, str]]:
    name = str(row["nome"] or "").strip() or f"Classification {row['id']}"
    title = name
    metadata: dict[str, str] = {
        "classification_id": str(row["id"]),
        "table_id": str(row["agregado_id"]),
    }
    parts: list[str] = []
    if row["agregado_nome"]:
        parts.append(f"Table {row['agregado_id']}: {row['agregado_nome']}")
        metadata["table_name"] = row["agregado_nome"]
    if "municipality_locality_count" in row.keys() and int(row["municipality_locality_count"] or 0) > 0:
        metadata["municipality_locality_count"] = str(int(row["municipality_locality_count"]))
    if "covers_national_municipalities" in row.keys() and int(row["covers_national_municipalities"] or 0):
        metadata["covers_national_municipalities"] = str(int(row["covers_national_municipalities"]))
    if row["sumarizacao_status"] is not None:
        metadata["summary_enabled"] = str(bool(row["sumarizacao_status"]))
        parts.append(f"Summarization: {'on' if row['sumarizacao_status'] else 'off'}")
    description = " | ".join(parts) or None
    return title, description, metadata


def _build_category_summary(row) -> tuple[str, str | None, dict[str, str]]:
    name = str(row["nome"] or "").strip() or f"Category {row['categoria_id']}"
    title = name
    metadata: dict[str, str] = {
        "category_id": str(row["categoria_id"]),
        "classification_id": str(row["classification_id"]),
        "table_id": str(row["agregado_id"]),
    }
    parts: list[str] = []
    if row["classification_nome"]:
        parts.append(f"Classification {row['classification_id']}: {row['classification_nome']}")
        metadata["classification_name"] = row["classification_nome"]
    if row["agregado_nome"]:
        parts.append(f"Table {row['agregado_id']}: {row['agregado_nome']}")
        metadata["table_name"] = row["agregado_nome"]
    if "municipality_locality_count" in row.keys() and int(row["municipality_locality_count"] or 0) > 0:
        metadata["municipality_locality_count"] = str(int(row["municipality_locality_count"]))
        parts.append(f"Municipalities: {int(row['municipality_locality_count']):,}")
    if "covers_national_municipalities" in row.keys() and int(row["covers_national_municipalities"] or 0):
        metadata["covers_national_municipalities"] = str(int(row["covers_national_municipalities"]))
        parts.append("National municipal coverage")
    if row["unidade"]:
        parts.append(f"Unit: {row['unidade']}")
        metadata["unit"] = row["unidade"]
    if row["nivel"] is not None:
        metadata["level"] = str(row["nivel"])
        parts.append(f"Level: {row['nivel']}")
    description = " | ".join(parts) or None
    return title, description, metadata


def _enrich_match(conn, match: SemanticMatch) -> tuple[SemanticResult, Mapping[str, object] | None]:
    if match.entity_type == "agregado" and match.agregado_id is not None:
        row = conn.execute(
            """
            SELECT id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim,
                   municipality_locality_count, covers_national_municipalities
            FROM agregados
            WHERE id = ?
            """,
            (match.agregado_id,),
        ).fetchone()
        if row:
            title, description, metadata = _build_agregado_summary(row)
            return (
                SemanticResult(
                    **match.__dict__,
                    title=title,
                    description=description,
                    metadata=metadata,
                ),
                row,
            )

    if match.entity_type == "variable" and match.agregado_id is not None:
        parts = match.entity_id.split(":")
        variable_id = int(parts[-1]) if parts and parts[-1].isdigit() else None
        if variable_id is not None:
            row = conn.execute(
                """
                SELECT v.id,
                       v.nome,
                       v.unidade,
                       v.agregado_id,
                       a.nome AS agregado_nome,
                       a.pesquisa,
                       a.covers_national_municipalities,
                       a.municipality_locality_count
                FROM variables v
                JOIN agregados a ON a.id = v.agregado_id
                WHERE v.agregado_id = ? AND v.id = ?
                """,
                (match.agregado_id, variable_id),
            ).fetchone()
            if row:
                title, description, metadata = _build_variable_summary(row)
                return (
                    SemanticResult(
                        **match.__dict__,
                        title=title,
                        description=description,
                        metadata=metadata,
                    ),
                    None,
                )

    if match.entity_type == "classification" and match.agregado_id is not None:
        parts = match.entity_id.split(":")
        classification_id = int(parts[-1]) if parts and parts[-1].isdigit() else None
        if classification_id is not None:
            row = conn.execute(
                """
                SELECT c.id,
                       c.nome,
                       c.agregado_id,
                       c.sumarizacao_status,
                       a.nome AS agregado_nome,
                       a.covers_national_municipalities,
                       a.municipality_locality_count
                FROM classifications c
                JOIN agregados a ON a.id = c.agregado_id
                WHERE c.agregado_id = ? AND c.id = ?
                """,
                (match.agregado_id, classification_id),
            ).fetchone()
            if row:
                title, description, metadata = _build_classification_summary(row)
                return (
                    SemanticResult(
                        **match.__dict__,
                        title=title,
                        description=description,
                        metadata=metadata,
                    ),
                    None,
                )

    if match.entity_type == "category" and match.agregado_id is not None:
        parts = match.entity_id.split(":")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            classification_id = int(parts[-2])
            category_id = int(parts[-1])
            row = conn.execute(
                """
                SELECT cat.agregado_id,
                       cat.classification_id,
                       cat.categoria_id,
                       cat.nome,
                       cat.unidade,
                       cat.nivel,
                       cls.nome AS classification_nome,
                       ag.nome AS agregado_nome,
                       ag.covers_national_municipalities,
                       ag.municipality_locality_count
                FROM categories cat
                JOIN classifications cls
                  ON cls.agregado_id = cat.agregado_id AND cls.id = cat.classification_id
                JOIN agregados ag ON ag.id = cat.agregado_id
                WHERE cat.agregado_id = ? AND cat.classification_id = ? AND cat.categoria_id = ?
                """,
                (match.agregado_id, classification_id, category_id),
            ).fetchone()
            if row:
                title, description, metadata = _build_category_summary(row)
                return (
                    SemanticResult(
                        **match.__dict__,
                        title=title,
                        description=description,
                        metadata=metadata,
                    ),
                    None,
                )

    # Fallback when we could not resolve metadata
    return (
        SemanticResult(
            **match.__dict__,
            title=f"{match.entity_type.title()} {match.entity_id}",
            description=None,
            metadata={},
        ),
        None,
    )


def semantic_search_with_metadata(
    query: str,
    *,
    entity_types: Sequence[str] | None = None,
    limit: int = 10,
    embedding_client: EmbeddingClient | None = None,
    model: str | None = None,
) -> list[SemanticResult]:
    """Run semantic search and attach friendly metadata for presentation."""

    matches = semantic_search(
        query,
        entity_types=entity_types,
        limit=limit,
        embedding_client=embedding_client,
        model=model,
    )
    if not matches:
        return []

    enriched: list[SemanticResult] = []
    with sqlite_session() as conn:
        for match in matches:
            result, _ = _enrich_match(conn, match)
            enriched.append(replace(result, combined_score=result.score))
    return enriched


def _augment_agregado_metadata(conn, agregado_id: int, metadata: dict[str, str]) -> None:
    variable_rows = conn.execute(
        "SELECT nome FROM variables WHERE agregado_id = ? ORDER BY id LIMIT 5",
        (agregado_id,),
    ).fetchall()
    if variable_rows:
        metadata["variables_sample"] = "; ".join(row["nome"] for row in variable_rows if row["nome"])
        count = conn.execute(
            "SELECT COUNT(*) FROM variables WHERE agregado_id = ?",
            (agregado_id,),
        ).fetchone()[0]
        metadata["variables_count"] = str(int(count))

    classification_rows = conn.execute(
        "SELECT nome FROM classifications WHERE agregado_id = ? ORDER BY id LIMIT 5",
        (agregado_id,),
    ).fetchall()
    if classification_rows:
        metadata["classifications_sample"] = "; ".join(
            row["nome"] for row in classification_rows if row["nome"]
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM classifications WHERE agregado_id = ?",
            (agregado_id,),
        ).fetchone()[0]
        metadata["classifications_count"] = str(int(count))

    level_rows = conn.execute(
        """
        SELECT level_id, level_name, level_type, locality_count
        FROM agregados_levels
        WHERE agregado_id = ?
        ORDER BY locality_count DESC, level_id
        """,
        (agregado_id,),
    ).fetchall()
    if level_rows:
        samples: list[str] = []
        for row in level_rows[:5]:
            level_id = row["level_id"]
            level_name = row["level_name"] or ""
            level_type = row["level_type"] or ""
            locality_count = row["locality_count"]
            snippet_parts = [level_id]
            if level_name:
                snippet_parts.append(level_name)
            if level_type:
                snippet_parts.append(level_type)
            if locality_count is not None:
                snippet_parts.append(f"{int(locality_count):,}")
            samples.append(" / ".join(str(part) for part in snippet_parts if part))
        metadata["levels_sample"] = "; ".join(samples)
        metadata["levels_count"] = str(len(level_rows))


def _fetch_agregado_row(conn, agregado_id: int | None):
    if agregado_id is None:
        return None
    return conn.execute(
        """
        SELECT id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim,
               municipality_locality_count, covers_national_municipalities
        FROM agregados
        WHERE id = ?
        """,
        (agregado_id,),
    ).fetchone()


def _passes_filters(row, filters: SearchFilters | None) -> bool:
    if filters is None or row is None:
        return True

    def _to_int(value) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    if filters.min_municipalities is not None:
        count = _to_int(row["municipality_locality_count"])
        if count is None or count < filters.min_municipalities:
            return False
    if filters.requires_national_munis and int(row["covers_national_municipalities"] or 0) != 1:
        return False
    if filters.subject_contains and filters.subject_contains.lower() not in _normalize_substring(row["assunto"]):
        return False
    if filters.survey_contains and filters.survey_contains.lower() not in _normalize_substring(row["pesquisa"]):
        return False

    start = _to_int(row["periodo_inicio"])
    end = _to_int(row["periodo_fim"])
    if filters.period_start is not None and end is not None and end < filters.period_start:
        return False
    if filters.period_end is not None and start is not None and start > filters.period_end:
        return False
    return True


def hybrid_search(
    query: str,
    *,
    limit: int = 10,
    filters: SearchFilters | None = None,
    embedding_client: EmbeddingClient | None = None,
    model: str | None = None,
    child_types: Sequence[str] | None = None,
    max_child_matches: int = 6,
    weights: Mapping[str, float] | None = None,
) -> list[SemanticResult]:
    if limit <= 0:
        return []

    child_types = list(child_types) if child_types is not None else [
        "variable",
        "classification",
        "category",
    ]
    tokens = _tokenize(query)
    weight_map = _resolve_weights(weights)
    overfetch = max(limit * 5, 50)

    matches = semantic_search(
        query,
        entity_types=["agregado"],
        limit=overfetch,
        embedding_client=embedding_client,
        model=model,
    )

    table_candidates: dict[int, dict[str, object]] = {}

    def _ensure_entry(table_id: int) -> dict[str, object]:
        entry = table_candidates.get(table_id)
        if entry is None:
            entry = {
                "match": SemanticMatch(
                    entity_type="agregado",
                    entity_id=str(table_id),
                    agregado_id=table_id,
                    score=0.0,
                    model=model or "lexical",
                ),
                "semantic": 0.0,
                "lex_table": 0.0,
            }
            table_candidates[table_id] = entry
        return entry

    for match in matches:
        if match.agregado_id is None:
            continue
        entry = _ensure_entry(match.agregado_id)
        if match.score > entry["semantic"]:
            entry["match"] = match
        entry["semantic"] = max(entry["semantic"], match.score)

    with sqlite_session() as conn:
        lexical_candidates = _lexical_candidates(
            conn,
            tokens,
            filters,
            max(overfetch, 50),
        )
        for agregado_id, lexical_score in lexical_candidates:
            entry = _ensure_entry(agregado_id)
            entry["lex_table"] = max(float(entry.get("lex_table", 0.0)), float(lexical_score))

        child_hits = _lexical_children(
            conn,
            tokens,
            filters,
            max(overfetch, 200),
            child_types,
        )
        child_map = _aggregate_children_by_table(
            child_hits,
            tokens,
            max_child_matches,
        )

        for agregado_id in child_map.keys():
            _ensure_entry(agregado_id)

        enriched: list[SemanticResult] = []
        for table_id, payload in table_candidates.items():
            match: SemanticMatch = payload["match"]  # type: ignore[assignment]
            semantic_score = float(payload.get("semantic", 0.0))
            result, aggregator_row = _enrich_match(conn, match)

            if aggregator_row is None:
                aggregator_row = _fetch_agregado_row(conn, table_id)

            if not _passes_filters(aggregator_row, filters):
                continue

            if match.entity_type == "agregado" and aggregator_row is not None:
                metadata = dict(result.metadata)
                _augment_agregado_metadata(conn, table_id, metadata)
                result = replace(result, metadata=metadata)

            lexical_table_score = float(payload.get("lex_table", 0.0))
            lexical_from_text = _compute_lexical_score(
                tokens,
                [result.title or "", result.description or ""],
            )
            lexical_table_score = max(lexical_table_score, lexical_from_text)

            child_payload = child_map.get(table_id, {})
            lexical_children_score = float(child_payload.get("score", 0.0))
            child_matches = tuple(child_payload.get("matches", ()))

            if semantic_score <= 0:
                semantic_score = result.score
            result = replace(result, score=semantic_score)

            boost = 0.0
            if aggregator_row is not None:
                covers_national = int(aggregator_row["covers_national_municipalities"] or 0)
                if covers_national:
                    boost += 0.03
                if filters and filters.period_start is not None:
                    try:
                        period_end = int(str(aggregator_row["periodo_fim"])) if aggregator_row["periodo_fim"] else None
                    except (TypeError, ValueError):
                        period_end = None
                    if period_end is not None and period_end >= filters.period_start:
                        boost += 0.02

            combined = (
                weight_map.get("sem", 0.0) * semantic_score
                + weight_map.get("lex_table", 0.0) * lexical_table_score
                + weight_map.get("lex_children", 0.0) * lexical_children_score
                + boost
            )

            lexical_total = lexical_table_score + lexical_children_score
            result = replace(
                result,
                lexical_table_score=lexical_table_score,
                lexical_children_score=lexical_children_score,
                lexical_score=lexical_total,
                combined_score=combined,
                child_matches=child_matches,
            )
            enriched.append(result)

    enriched.sort(key=lambda item: item.combined_score, reverse=True)
    return enriched[:limit]


__all__ = [
    "SemanticMatch",
    "SemanticResult",
    "SearchFilters",
    "ChildMatch",
    "semantic_search",
    "semantic_search_with_metadata",
    "hybrid_search",
]
