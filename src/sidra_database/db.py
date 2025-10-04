"""SQLite utilities for the SIDRA metadata store."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings
from .schema import apply_schema


def get_database_path() -> Path:
    settings = get_settings()
    return Path(settings.database_path).expanduser().resolve()


def create_connection() -> sqlite3.Connection:
    settings = get_settings()
    path = get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(float(settings.database_timeout), 60.0)
    connection = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    busy_timeout_ms = max(int(timeout * 1000), 60000)
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    connection.row_factory = sqlite3.Row
    return connection


def ensure_schema(connection: sqlite3.Connection | None = None) -> None:
    close_after = False
    if connection is None:
        connection = create_connection()
        close_after = True
    try:
        apply_schema(connection)
    finally:
        if close_after:
            connection.close()


@contextmanager
def sqlite_session() -> Iterator[sqlite3.Connection]:
    connection = create_connection()
    try:
        yield connection
    finally:
        connection.close()


__all__ = ["create_connection", "ensure_schema", "sqlite_session", "get_database_path"]
