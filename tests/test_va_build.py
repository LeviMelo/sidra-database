import asyncio
import json

from sidra_database.db import create_connection, ensure_schema

from sidra_va.schema_migrations import apply_va_schema
from sidra_va.value_index import build_va_index_for_agregado, build_va_index_for_all


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


def _seed_sample_agregado(conn, agregado_id: int = 9999) -> None:
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
            agregado_id,
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
        (agregado_id, "N3", "Estado", "N"),
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
        (agregado_id + 1, agregado_id, "Alfabetização", "%", "{}", "hash"),
    )
    conn.execute(
        """
        INSERT INTO classifications (id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao)
        VALUES (?, ?, ?, ?, ?)
        """,
        (agregado_id + 11, agregado_id, "Cor ou raça", 0, None),
    )
    for idx, nome in enumerate(["Branca", "Preta", "Indígena"], start=1):
        conn.execute(
            """
            INSERT INTO categories (agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (agregado_id, agregado_id + 11, idx, nome, None, None, "hash"),
        )
    conn.commit()


def test_build_va_index_creates_rows():
    ensure_schema()
    conn = create_connection()
    apply_va_schema(conn)
    _reset_tables(conn)
    _seed_sample_agregado(conn, agregado_id=9999)

    created = asyncio.run(build_va_index_for_agregado(9999))
    assert created == 4

    rows = conn.execute(
        "SELECT va_id, dims_json, has_n3, has_n6, period_start, period_end FROM value_atoms ORDER BY va_id"
    ).fetchall()
    assert len(rows) == 4
    first = rows[0]
    assert first["va_id"] == "9999::v10000"
    dims = json.loads(first["dims_json"])
    assert dims == []
    assert first["has_n3"] == 1 and first["has_n6"] == 1
    assert first["period_start"] == "2010"
    assert first["period_end"] == "2012"

    third = rows[2]
    dims = json.loads(third["dims_json"])
    assert dims[0]["classification_name"] == "Cor ou raça"
    assert conn.execute("SELECT COUNT(*) FROM value_atoms_fts").fetchone()[0] == 4

    fingerprint = conn.execute(
        "SELECT fingerprint FROM variable_fingerprints WHERE variable_id = ?",
        (10000,),
    ).fetchone()
    assert fingerprint and len(fingerprint[0]) == 64

    conn.close()


def test_build_va_index_for_all_serial_writes():
    ensure_schema()
    conn = create_connection()
    apply_va_schema(conn)
    _reset_tables(conn)
    agregado_id = 12000
    _seed_sample_agregado(conn, agregado_id=agregado_id)
    conn.close()

    results = asyncio.run(build_va_index_for_all(concurrency=1))
    assert results == {str(agregado_id): 4}

    conn = create_connection()
    try:
        apply_va_schema(conn)
        count = conn.execute("SELECT COUNT(*) FROM value_atoms WHERE agregado_id = ?", (agregado_id,)).fetchone()[0]
        assert count == 4
    finally:
        conn.close()
