from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from .base_schema import apply_base_schema
from .migrations import apply_search_schema


def _has_base_tables(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('agregados','variables')"
        )
        names = {r[0] for r in cur.fetchall()}
        return "agregados" in names and "variables" in names
    except sqlite3.Error:
        return False
    finally:
        try: conn.close()
        except: pass


def _resolve_db_path() -> Path:
    env = os.getenv("SIDRA_DATABASE_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    # default
    p = Path("sidra.db").resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_DB_PATH: Path | None = None

def get_database_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = _resolve_db_path()
    return _DB_PATH


def create_connection() -> sqlite3.Connection:
    s = get_settings()
    conn = sqlite3.connect(
        get_database_path(),
        timeout=max(float(s.database_timeout), 30.0),
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {int(max(s.database_timeout, 60.0) * 1000)}")
    return conn


@contextmanager
def sqlite_session() -> Iterator[sqlite3.Connection]:
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()


def ensure_full_schema() -> None:
    conn = create_connection()
    try:
        apply_base_schema(conn)
        apply_search_schema(conn)
        conn.commit()
    finally:
        conn.close()
