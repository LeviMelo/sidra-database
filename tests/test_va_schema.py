import sqlite3

from sidra_database.db import create_connection, ensure_schema

from sidra_va.schema_migrations import apply_va_schema, get_schema_version


def test_apply_va_schema_idempotent(tmp_path):
    conn = create_connection()
    try:
        ensure_schema(conn)
        apply_va_schema(conn)
        version_first = get_schema_version(conn)
        apply_va_schema(conn)
        version_second = get_schema_version(conn)
        assert version_first == 1
        assert version_second == 1

        tables = [
            "value_atoms",
            "value_atom_dims",
            "value_atoms_fts",
            "synonyms",
            "variable_fingerprints",
        ]
        for table in tables:
            conn.execute(f"SELECT * FROM {table} LIMIT 0")
    finally:
        conn.close()


def test_schema_version_persists_across_connections():
    conn = create_connection()
    try:
        ensure_schema(conn)
        apply_va_schema(conn)
        assert get_schema_version(conn) == 1
    finally:
        conn.close()

    conn = create_connection()
    try:
        ensure_schema(conn)
        assert get_schema_version(conn) == 1
    finally:
        conn.close()
