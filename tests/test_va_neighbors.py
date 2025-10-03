import asyncio

from sidra_database.db import create_connection, ensure_schema

from sidra_va.neighbors import find_neighbors_for_va
from sidra_va.schema_migrations import apply_va_schema
from sidra_va.value_index import build_va_index_for_agregado


def _reset(conn):
    for table in [
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


def _insert_agregado(conn, agregado_id: int, variable_id: int):
    conn.execute(
        """
        INSERT INTO agregados (id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agregado_id,
            "População alfabetizada",
            "Censo",
            "Educação",
            None,
            None,
            "2010",
            "2012",
            b"{}",
            "2024-01-01T00:00:00Z",
        ),
    )
    conn.execute(
        "INSERT INTO agregados_levels (agregado_id, level_id, level_name, level_type) VALUES (?, ?, ?, ?)",
        (agregado_id, "N6", "Município", "N"),
    )
    conn.execute(
        """
        INSERT INTO variables (id, agregado_id, nome, unidade, sumarizacao, text_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (variable_id, agregado_id, "Alfabetização", "%", "{}", "hash"),
    )
    conn.execute(
        """
        INSERT INTO classifications (id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao)
        VALUES (?, ?, ?, ?, ?)
        """,
        (10, agregado_id, "Cor ou raça", 0, None),
    )
    conn.execute(
        """
        INSERT INTO categories (agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (agregado_id, 10, 3, "Indígena", None, None, "hash"),
    )
    conn.commit()


def test_find_neighbors_matches_by_fingerprint():
    ensure_schema()
    conn = create_connection()
    apply_va_schema(conn)
    _reset(conn)

    _insert_agregado(conn, 1000, 200)
    _insert_agregado(conn, 2000, 300)

    asyncio.run(build_va_index_for_agregado(1000))
    asyncio.run(build_va_index_for_agregado(2000))

    seed_va = conn.execute("SELECT va_id FROM value_atoms WHERE agregado_id = ? AND va_id LIKE ?", (1000, "%c10:3")).fetchone()[0]
    neighbors = find_neighbors_for_va(seed_va)
    assert neighbors
    top, score = neighbors[0]
    assert top.va_id.endswith("c10:3")
    assert score >= 0.9

    conn.close()
