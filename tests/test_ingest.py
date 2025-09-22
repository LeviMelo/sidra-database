from __future__ import annotations

from array import array
from pathlib import Path
from typing import Sequence
import asyncio

import pytest

from sidra_database import (
    ensure_schema,
    get_database_path,
    ingest_agregado,
    semantic_search,
    semantic_search_with_metadata,
    get_settings,
)
from sidra_database.db import create_connection


class FakeSidraClient:
    def __init__(self) -> None:
        self._localities = {
            ("Administrativo", "N1"): [
                {"id": "1", "nome": "Brasil", "nivel": {"nome": "Brasil"}},
            ],
            ("Administrativo", "N3"): [
                {"id": "11", "nome": "Rondonia", "nivel": {"nome": "Unidade da Federacao"}},
                {"id": "12", "nome": "Acre", "nivel": {"nome": "Unidade da Federacao"}},
            ],
        }

    async def fetch_metadata(self, agregado_id: int):
        assert agregado_id == 1234
        return {
            "id": agregado_id,
            "nome": "Tabela teste",
            "pesquisa": "Pesquisa demo",
            "assunto": "Demografia",
            "URL": "https://sidra.ibge.gov.br/tabela/1234",
            "periodicidade": {"frequencia": "anual", "inicio": 2000, "fim": 2020},
            "nivelTerritorial": {
                "Administrativo": ["N1", "N3"],
                "Especial": [],
                "IBGE": [],
            },
            "variaveis": [
                {"id": 99, "nome": "Populacao", "unidade": "Pessoas", "sumarizacao": ["nivelTerritorial"]}
            ],
            "classificacoes": [
                {
                    "id": 1,
                    "nome": "Sexo",
                    "sumarizacao": {"status": True, "excecao": []},
                    "categorias": [
                        {"id": 0, "nome": "Total", "unidade": None, "nivel": 0},
                        {"id": 4, "nome": "Homens", "unidade": None, "nivel": 1},
                    ],
                }
            ],
        }

    async def fetch_periods(self, agregado_id: int):
        assert agregado_id == 1234
        return [
            {"id": "2000", "literals": ["2000"], "modificacao": "01/01/2021"},
            {"id": "2020", "literals": ["2020"], "modificacao": "01/01/2021"},
        ]

    async def fetch_localities(self, agregado_id: int, level: str):
        assert agregado_id == 1234
        for (level_type, level_code), payload in self._localities.items():
            if level_code == level:
                return payload
        return []


class FakeEmbeddingClient:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = [tuple(float(value) for value in vector) for vector in vectors]
        self._model = "fake-model"
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
        if model is not None:
            assert model == self._model
        index = len(self.calls)
        self.calls.append(text)
        if index < len(self._vectors):
            return list(self._vectors[index])
        return list(self._vectors[-1])


@pytest.fixture(autouse=True)
def override_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "sidra-test.db"
    monkeypatch.setenv("SIDRA_DATABASE_PATH", str(db_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    if db_path.exists():
        db_path.unlink()


def test_ingest_agregado_writes_metadata():
    ensure_schema()
    client = FakeSidraClient()
    embedding_vectors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
    ]
    embedding_client = FakeEmbeddingClient(embedding_vectors)

    asyncio.run(ingest_agregado(1234, client=client, embedding_client=embedding_client))

    conn = create_connection()
    try:
        row = conn.execute("SELECT nome, pesquisa, assunto FROM agregados WHERE id = 1234").fetchone()
        assert row["nome"] == "Tabela teste"
        assert row["pesquisa"] == "Pesquisa demo"

        variables = conn.execute("SELECT COUNT(*) FROM variables WHERE agregado_id = 1234").fetchone()[0]
        assert variables == 1

        categories = conn.execute("SELECT COUNT(*) FROM categories WHERE agregado_id = 1234").fetchone()[0]
        assert categories == 2

        periods = conn.execute("SELECT periodo_id FROM periods WHERE agregado_id = 1234 ORDER BY periodo_id").fetchall()
        assert [p["periodo_id"] for p in periods] == ["2000", "2020"]

        localities = conn.execute(
            "SELECT COUNT(*) FROM localities WHERE agregado_id = 1234 AND level_id = ?",
            ("N3",),
        ).fetchone()[0]
        assert localities == 2

        embedding_rows = conn.execute(
            """
            SELECT entity_type, entity_id, agregado_id, dimension, vector, model
            FROM embeddings
            WHERE agregado_id = 1234
            ORDER BY entity_type, entity_id
            """
        ).fetchall()
        assert len(embedding_rows) == 5
        assert all(row["model"] == embedding_client.model for row in embedding_rows)
        assert all(row["agregado_id"] == 1234 for row in embedding_rows)

        expected = {
            ("agregado", "1234"): embedding_vectors[0],
            ("variable", "1234:99"): embedding_vectors[1],
            ("classification", "1234:1"): embedding_vectors[2],
            ("category", "1234:1:0"): embedding_vectors[3],
            ("category", "1234:1:4"): embedding_vectors[4],
        }
        for row in embedding_rows:
            key = (row["entity_type"], row["entity_id"])
            assert key in expected
            unpacked = array("f")
            unpacked.frombytes(row["vector"])
            assert len(unpacked) == row["dimension"]
            assert list(unpacked) == pytest.approx(expected[key])

        assert len(embedding_client.calls) == 5
        assert embedding_client.calls[0].startswith("Table 1234")
        assert any("Classification 1" in call for call in embedding_client.calls)
    finally:
        conn.close()


def test_get_database_path_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "alt.db"
    monkeypatch.setenv("SIDRA_DATABASE_PATH", str(target))
    ensure_schema()
    path = get_database_path()
    assert path == target.resolve()


def test_semantic_search_orders_results():
    ensure_schema()
    conn = create_connection()
    try:
        conn.execute("DELETE FROM embeddings")

        def insert(entity_type: str, entity_id: str, vector: Sequence[float], text_hash: str) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (
                    entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_type,
                    entity_id,
                    1234,
                    text_hash,
                    "fake-model",
                    len(vector),
                    array("f", vector).tobytes(),
                    "2024-01-01T00:00:00Z",
                ),
            )

        insert("variable", "var_a", (1.0, 0.0), "h1")
        insert("variable", "var_b", (0.5, 0.5), "h2")
        insert("category", "cat_c", (0.0, 1.0), "h3")
        insert("variable", "var_d", (-1.0, 0.0), "h4")
        conn.commit()
    finally:
        conn.close()

    class QueryEmbeddingClient:
        def __init__(self) -> None:
            self._model = "fake-model"

        @property
        def model(self) -> str:
            return self._model

        def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
            assert model == self._model
            return [1.0, 0.0]

    client = QueryEmbeddingClient()
    results = semantic_search("population", embedding_client=client, limit=3)
    assert [item.entity_id for item in results] == ["var_a", "var_b", "cat_c"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.70710677, rel=1e-5)
    assert results[2].score == pytest.approx(0.0)

    filtered = semantic_search(
        "population",
        embedding_client=client,
        limit=2,
        entity_types=["variable"],
    )
    assert [item.entity_id for item in filtered] == ["var_a", "var_b"]


def test_semantic_search_with_metadata_enriches_results():
    ensure_schema()
    client = FakeSidraClient()
    embedding_vectors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
    ]
    embedding_client = FakeEmbeddingClient(embedding_vectors)
    asyncio.run(ingest_agregado(1234, client=client, embedding_client=embedding_client))

    class QueryEmbeddingClient:
        def __init__(self, vector: Sequence[float]) -> None:
            self._model = "fake-model"
            self._vector = [float(value) for value in vector]

        @property
        def model(self) -> str:
            return self._model

        def embed_text(self, text: str, *, model: str | None = None) -> list[float]:
            assert model == self._model
            return list(self._vector)

    aggregate_results = semantic_search_with_metadata(
        "table overview",
        limit=3,
        embedding_client=QueryEmbeddingClient((1.0, 0.0, 0.0)),
    )
    assert aggregate_results
    top = aggregate_results[0]
    assert top.entity_type == "agregado"
    assert top.metadata["table_id"] == "1234"
    assert top.title == "Tabela teste"
    assert "Pesquisa demo" in (top.description or "")

    variable_results = semantic_search_with_metadata(
        "variable",
        limit=3,
        embedding_client=QueryEmbeddingClient((0.0, 1.0, 0.0)),
    )
    variable = next(item for item in variable_results if item.entity_type == "variable")
    assert variable.metadata["unit"] == "Pessoas"
    assert "Table 1234" in (variable.description or "")

    classification_results = semantic_search_with_metadata(
        "classification",
        limit=3,
        embedding_client=QueryEmbeddingClient((0.0, 0.0, 1.0)),
    )
    classification = next(item for item in classification_results if item.entity_type == "classification")
    assert classification.metadata["classification_id"] == "1"
    assert "Summarization" in (classification.description or "")
