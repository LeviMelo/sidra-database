import asyncio
from array import array

from sidra_database.db import create_connection, ensure_schema

from sidra_va.schema_migrations import apply_va_schema
from sidra_va.search_va import VaSearchFilters, search_value_atoms
from sidra_va.value_index import build_va_index_for_agregado


class StubEmbeddingClient:
    def __init__(self, vector):
        self._vector = vector
        self._model = "stub-model"

    @property
    def model(self):
        return self._model

    def embed_text(self, text: str, *, model: str | None = None):
        return list(self._vector)


def _reset(conn):
    for table in [
        "embeddings",
        "synonyms",
        "value_atom_dims",
        "value_atoms_fts",
        "value_atoms",
        "variable_fingerprints",
        "categories",
        "classifications",
        "variables",
        "agregados_levels",
        "agregados",
    ]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def _insert_base_fixture(conn):
    conn.execute(
        """
        INSERT INTO agregados (id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at,
                               municipality_locality_count, covers_national_municipalities)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            9999,
            "População alfabetizada",
            "Censo",
            "Educação",
            None,
            None,
            "2010",
            "2012",
            b"{}",
            "2024-01-01T00:00:00Z",
            5000,
            1,
        ),
    )
    conn.execute(
        "INSERT INTO agregados_levels (agregado_id, level_id, level_name, level_type) VALUES (?, ?, ?, ?)",
        (9999, "N3", "Estado", "N"),
    )
    conn.execute(
        "INSERT INTO agregados_levels (agregado_id, level_id, level_name, level_type) VALUES (?, ?, ?, ?)",
        (9999, "N6", "Município", "N"),
    )
    conn.execute(
        """
        INSERT INTO variables (id, agregado_id, nome, unidade, sumarizacao, text_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (100, 9999, "Alfabetização", "%", "{}", "hash"),
    )
    conn.execute(
        """
        INSERT INTO classifications (id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao)
        VALUES (?, ?, ?, ?, ?)
        """,
        (10, 9999, "Cor ou raça", 0, None),
    )
    for cat_id, nome in [(1, "Branca"), (2, "Preta"), (3, "Indígena")]:
        conn.execute(
            """
            INSERT INTO categories (agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (9999, 10, cat_id, nome, None, None, "hash"),
        )
    conn.commit()


def test_search_value_atoms_promotes_exact_matches():
    ensure_schema()
    conn = create_connection()
    apply_va_schema(conn)
    _reset(conn)
    _insert_base_fixture(conn)
    conn.execute(
        "INSERT INTO synonyms(kind, key, alt) VALUES(?, ?, ?)",
        ("category", "indigena", "indígena"),
    )
    conn.commit()

    asyncio.run(build_va_index_for_agregado(9999))

    cursor = conn.execute("SELECT va_id, text FROM value_atoms ORDER BY va_id")
    vectors = {}
    for idx, (va_id, text) in enumerate(cursor.fetchall(), start=1):
        vector = [1.0 if "Indígena" in text else 0.5, 0.1 * idx]
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "va",
                va_id,
                9999,
                "hash",
                "stub-model",
                2,
                array("f", vector).tobytes(),
                "2024-01-01T00:00:00Z",
            ),
        )
    conn.commit()

    client = StubEmbeddingClient([1.0, 0.0])
    filters = VaSearchFilters(require_levels=("N6",), period_start=2010, period_end=2012)
    results = asyncio.run(
        search_value_atoms(
            "alfabetização indígena município 2010-2012",
            filters=filters,
            embedding_client=client,
            limit=5,
        )
    )
    assert results
    top = results[0]
    assert top.va_id.endswith("c10:3")
    assert "Indígena" in top.title
    assert "levels=N3,N6" in top.why
    assert "period=2010-2012" in top.why

    conn.close()
