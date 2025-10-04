from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from typing import Iterable

from .synonyms import normalize_basic


def utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_tokens(tokens: Iterable[str]) -> str:
    normalized = [normalize_basic(tok) for tok in tokens if tok]
    return " ".join(sorted(tok for tok in normalized if tok))


def run_with_retries(fn, *, retries: int = 8, base_sleep: float = 0.05, exc=sqlite3.OperationalError):
    for attempt in range(retries):
        try:
            return fn()
        except exc as error:  # noqa: PERF203 - limited to OperationalError cases
            message = str(error).lower()
            if "database is locked" not in message and "busy" not in message:
                raise
            if attempt == retries - 1:
                raise
            time.sleep(base_sleep * (2**attempt))


__all__ = [
    "utcnow_iso",
    "json_dumps",
    "sha256_text",
    "canonicalize_tokens",
    "run_with_retries",
]
