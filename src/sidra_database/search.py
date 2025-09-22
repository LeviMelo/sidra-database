"""Semantic search helpers over stored embeddings."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
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


def _enrich_match(conn, match: SemanticMatch) -> SemanticResult:
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
            return SemanticResult(**match.__dict__, title=title, description=description, metadata=metadata)

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
                return SemanticResult(**match.__dict__, title=title, description=description, metadata=metadata)

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
                return SemanticResult(**match.__dict__, title=title, description=description, metadata=metadata)

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
                return SemanticResult(**match.__dict__, title=title, description=description, metadata=metadata)

    # Fallback when we could not resolve metadata
    return SemanticResult(
        **match.__dict__,
        title=f"{match.entity_type.title()} {match.entity_id}",
        description=None,
        metadata={},
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
            enriched.append(_enrich_match(conn, match))
    return enriched


__all__ = [
    "SemanticMatch",
    "SemanticResult",
    "semantic_search",
    "semantic_search_with_metadata",
]
