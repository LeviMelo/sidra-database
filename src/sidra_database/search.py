"""Semantic search helpers over stored embeddings."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from typing import Sequence

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


__all__ = ["SemanticMatch", "semantic_search"]
