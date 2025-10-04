"""SQLite helpers for sidra_va operations."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import get_settings


def _has_base_tables(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return False
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('agregados', 'variables')"
        )
        names = {row[0] for row in cursor.fetchall()}
        return "agregados" in names and "variables" in names
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _resolve_database_path(env_value: str | None) -> Path:
    candidate = Path(env_value).expanduser().resolve() if env_value else None
    default_path = Path("sidra.db").expanduser().resolve()
    if candidate is not None:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not candidate.exists():
            if _has_base_tables(default_path):
                return default_path
            return candidate
        if _has_base_tables(candidate):
            return candidate
        if _has_base_tables(default_path):
            return default_path
        return candidate
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


_LAST_ENV_VALUE: str | None = None
_DATABASE_PATH: Path | None = None


def get_database_path() -> Path:
    """Resolve the SQLite database path used for VA operations."""

    global _DATABASE_PATH, _LAST_ENV_VALUE
    env_value = os.getenv("SIDRA_DATABASE_PATH")
    if _DATABASE_PATH is None or env_value != _LAST_ENV_VALUE:
        _DATABASE_PATH = _resolve_database_path(env_value)
        _LAST_ENV_VALUE = env_value
    return _DATABASE_PATH


def create_connection() -> sqlite3.Connection:
    """Create a SQLite connection configured for concurrent reads/writes."""

    settings = get_settings()
    timeout = max(float(settings.database_timeout), 30.0)
    connection = sqlite3.connect(
        get_database_path(),
        timeout=timeout,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    busy_timeout_ms = max(int(timeout * 1000), 60000)
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    return connection


@contextmanager
def sqlite_session() -> Iterator[sqlite3.Connection]:
    connection = create_connection()
    try:
        yield connection
    finally:
        connection.close()


__all__ = ["create_connection", "get_database_path", "sqlite_session"]

