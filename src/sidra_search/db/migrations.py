from __future__ import annotations

import sqlite3

from .search_schema import apply_search_schema as _apply_schema

SEARCH_SCHEMA_VERSION = 4
KEY = "sidra_search_schema_version"

def _ensure_meta(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS meta_kv(
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
    """)

def get_search_schema_version(connection: sqlite3.Connection) -> int:
    _ensure_meta(connection)
    cur = connection.execute("SELECT value FROM meta_kv WHERE key = ?", (KEY,))
    row = cur.fetchone()
    return int(row[0]) if row else 0

def bump_search_schema_version(connection: sqlite3.Connection, to_version: int) -> None:
    _ensure_meta(connection)
    connection.execute(
        "INSERT INTO meta_kv(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (KEY, str(to_version)),
    )

def apply_search_schema(connection: sqlite3.Connection) -> None:
    current = get_search_schema_version(connection)
    if current >= SEARCH_SCHEMA_VERSION:
        return
    with connection:
        _apply_schema(connection)
        bump_search_schema_version(connection, SEARCH_SCHEMA_VERSION)
