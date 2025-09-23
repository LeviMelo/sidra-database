"""Semantic search helpers over stored embeddings."""
from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
import math
import re
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


@dataclass(frozen=True)
class SearchFilters:
    """Structured filters to refine search results."""

    min_municipalities: int | None = None
    requires_national_munis: bool = False
    subject_contains: str | None = None
    survey_contains: str | None = None
    period_start: int | None = None
    period_end: int | None = None


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


def _tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"\w+", text)]
    return [token for token in tokens if len(token) >= 3]


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


def _filters_to_sql(filters: SearchFilters | None) -> tuple[list[str], list[object]]:
    if filters is None:
        return [], []
    conditions: list[str] = []
    params: list[object] = []
    if filters.min_municipalities is not None:
        conditions.append("municipality_locality_count >= ?")
        params.append(int(filters.min_municipalities))
    if filters.requires_national_munis:
        conditions.append("covers_national_municipalities = 1")
    if filters.subject_contains:
        conditions.append("LOWER(assunto) LIKE ?")
        params.append(f"%{filters.subject_contains.lower()}%")
    if filters.survey_contains:
        conditions.append("LOWER(pesquisa) LIKE ?")
        params.append(f"%{filters.survey_contains.lower()}%")
    if filters.period_start is not None:
        conditions.append(
            "(periodo_fim IS NULL OR CAST(periodo_fim AS INTEGER) >= ?)"
        )
        params.append(int(filters.period_start))
    if filters.period_end is not None:
        conditions.append(
            "(periodo_inicio IS NULL OR CAST(periodo_inicio AS INTEGER) <= ?)"
        )
        params.append(int(filters.period_end))
    return conditions, params


def _lexical_candidates(
    conn,
    tokens: Sequence[str],
    filters: SearchFilters | None,
    limit: int,
) -> list[tuple[int, float]]:
    if not tokens or limit <= 0:
        return []

    conditions, params = _filters_to_sql(filters)
    token_clauses: list[str] = []
    for token in tokens:
        pattern = f"%{token}%"
        token_clauses.append(
            "(LOWER(nome) LIKE ? OR LOWER(pesquisa) LIKE ? OR LOWER(assunto) LIKE ?)"
        )
        params.extend([pattern, pattern, pattern])

    where_parts = conditions[:]
    if token_clauses:
        where_parts.append(" AND ".join(token_clauses))

    sql = "SELECT id, nome, pesquisa, assunto FROM agregados"
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
    entity_types: Sequence[str] | None = None,
    limit: int = 10,
    filters: SearchFilters | None = None,
    embedding_client: EmbeddingClient | None = None,
    model: str | None = None,
) -> list[SemanticResult]:
    if limit <= 0:
        return []

    tokens = _tokenize(query)
    overfetch = max(limit * 5, limit)

    matches = semantic_search(
        query,
        entity_types=entity_types,
        limit=overfetch,
        embedding_client=embedding_client,
        model=model,
    )

    result_map: dict[tuple[str, str], dict[str, object]] = {}
    for match in matches:
        key = (match.entity_type, match.entity_id)
        result_map[key] = {"match": match, "lexical_hint": 0.0}

    with sqlite_session() as conn:
        lexical_candidates = _lexical_candidates(conn, tokens, filters, max(overfetch, 50))
        for agregado_id, lexical_score in lexical_candidates:
            key = ("agregado", str(agregado_id))
            if key not in result_map:
                result_map[key] = {
                    "match": SemanticMatch(
                        entity_type="agregado",
                        entity_id=str(agregado_id),
                        agregado_id=agregado_id,
                        score=0.0,
                        model=model or "lexical",
                    ),
                    "lexical_hint": lexical_score,
                }
            else:
                result_map[key]["lexical_hint"] = max(
                    float(result_map[key]["lexical_hint"]), float(lexical_score)
                )

        enriched: list[SemanticResult] = []
        for key, payload in result_map.items():
            match: SemanticMatch = payload["match"]  # type: ignore[assignment]
            result, aggregator_row = _enrich_match(conn, match)

            row_for_filters = aggregator_row or _fetch_agregado_row(conn, match.agregado_id)

            if match.entity_type == "agregado" and match.agregado_id is not None and row_for_filters is not None:
                metadata = dict(result.metadata)
                _augment_agregado_metadata(conn, match.agregado_id, metadata)
                result = replace(result, metadata=metadata)

            if not _passes_filters(row_for_filters, filters):
                continue

            text_parts = [result.title or "", result.description or ""]
            if isinstance(result.metadata, Mapping):
                text_parts.extend(str(value) for value in result.metadata.values())
            lexical_score = _compute_lexical_score(tokens, text_parts)
            lexical_score = max(float(lexical_score), float(payload["lexical_hint"]))
            combined_score = _combine_scores(result.score, lexical_score)
            result = replace(result, lexical_score=lexical_score, combined_score=combined_score)
            enriched.append(result)

    enriched.sort(key=lambda item: item.combined_score, reverse=True)
    return enriched[:limit]


__all__ = [
    "SemanticMatch",
    "SemanticResult",
    "SearchFilters",
    "semantic_search",
    "semantic_search_with_metadata",
    "hybrid_search",
]
