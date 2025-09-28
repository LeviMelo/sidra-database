import asyncio
import json

from sidra_database.db import create_connection, ensure_schema

from sidra_va.schema_migrations import apply_va_schema
from sidra_va.value_index import build_va_index_for_agregado


def _reset_tables(conn):
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
        "synonyms",
    ]:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def test_build_va_index_creates_rows():
    ensure_schema()
    conn = create_connection()
    apply_va_schema(conn)
    _reset_tables(conn)

    raw_json = json.dumps(
        {
            "periodicidade": {"inicio": "2010", "fim": "2012"},
            "nivelTerritorial": {"N": ["N3", "N6"]},
            "pesquisa": "Censo",
            "assunto": "Educação",
            "nome": "População alfabetizada",
        }
    )
    conn.execute(
        """
        INSERT INTO agregados (id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            raw_json.encode("utf-8"),
            "2024-01-01T00:00:00Z",
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

    created = asyncio.run(build_va_index_for_agregado(9999))
    assert created == 4

    rows = conn.execute(
        "SELECT va_id, dims_json, has_n3, has_n6, period_start, period_end FROM value_atoms ORDER BY va_id"
    ).fetchall()
    assert len(rows) == 4
    first = rows[0]
    assert first["va_id"] == "9999::v100"
    dims = json.loads(first["dims_json"])
    assert dims == []
    assert first["has_n3"] == 1 and first["has_n6"] == 1
    assert first["period_start"] == "2010"
    assert first["period_end"] == "2012"

    third = rows[2]
    dims = json.loads(third["dims_json"])
    assert dims[0]["classification_name"] == "Cor ou raça"
    assert conn.execute("SELECT COUNT(*) FROM value_atoms_fts").fetchone()[0] == 4

    fingerprint = conn.execute("SELECT fingerprint FROM variable_fingerprints WHERE variable_id = ?", (100,)).fetchone()
    assert fingerprint and len(fingerprint[0]) == 64

    conn.close()
