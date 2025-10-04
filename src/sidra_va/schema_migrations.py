from __future__ import annotations

import sqlite3
from datetime import datetime

VA_SCHEMA_VERSION = 2


def _ensure_meta_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS meta_kv (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )


def get_schema_version(connection: sqlite3.Connection) -> int:
    _ensure_meta_table(connection)
    cur = connection.execute(
        "SELECT value FROM meta_kv WHERE key = ?", ("sidra_va_schema_version",)
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def bump_schema_version(connection: sqlite3.Connection, to_version: int) -> None:
    _ensure_meta_table(connection)
    connection.execute(
        "INSERT INTO meta_kv(key, value) VALUES(?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ("sidra_va_schema_version", str(to_version)),
    )


def apply_va_schema(connection: sqlite3.Connection) -> None:
    """Apply additive schema objects for the VA subsystem."""

    current_version = get_schema_version(connection)
    if current_version >= VA_SCHEMA_VERSION:
        return

    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              entity_type TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              agregado_id INTEGER,
              text_hash TEXT NOT NULL,
              model TEXT NOT NULL,
              dimension INTEGER NOT NULL,
              vector BLOB NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (entity_type, entity_id, model)
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_agregado
            ON embeddings(agregado_id, model)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS value_atoms (
              va_id TEXT PRIMARY KEY,
              agregado_id INTEGER NOT NULL,
              variable_id INTEGER NOT NULL,
              unit TEXT,
              text TEXT NOT NULL,
              dims_json TEXT NOT NULL,
              has_n1 INTEGER DEFAULT 0,
              has_n2 INTEGER DEFAULT 0,
              has_n3 INTEGER DEFAULT 0,
              has_n6 INTEGER DEFAULT 0,
              period_start TEXT,
              period_end TEXT,
              survey TEXT,
              subject TEXT,
              table_title TEXT,
              created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS value_atom_dims (
              va_id TEXT NOT NULL,
              classification_id INTEGER NOT NULL,
              classification_name TEXT NOT NULL,
              category_id INTEGER NOT NULL,
              category_name TEXT NOT NULL,
              PRIMARY KEY (va_id, classification_id, category_id),
              FOREIGN KEY (va_id) REFERENCES value_atoms(va_id)
            )
            """
        )

        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS value_atoms_fts
            USING fts5(va_id UNINDEXED, text, table_title, survey, subject, tokenize='unicode61')
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS synonyms (
              kind TEXT NOT NULL,
              key TEXT NOT NULL,
              alt TEXT NOT NULL,
              PRIMARY KEY (kind, key, alt)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS variable_fingerprints (
              variable_id INTEGER PRIMARY KEY,
              fingerprint TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_value_atoms_agregado ON value_atoms(agregado_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_value_atoms_variable ON value_atoms(variable_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_value_atoms_levels ON value_atoms(has_n3, has_n6)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_value_atoms_period ON value_atoms(period_start, period_end)
            """
        )

        bump_schema_version(connection, VA_SCHEMA_VERSION)


__all__ = [
    "apply_va_schema",
    "get_schema_version",
    "bump_schema_version",
]
