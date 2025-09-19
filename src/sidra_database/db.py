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
    path = get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
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
