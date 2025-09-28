from __future__ import annotations

import hashlib
import json
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


__all__ = [
    "utcnow_iso",
    "json_dumps",
    "sha256_text",
    "canonicalize_tokens",
]
