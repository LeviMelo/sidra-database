from __future__ import annotations

import hashlib
from typing import Mapping

from .synonyms import SynonymMap, normalize_basic, normalize_token


def variable_fingerprint(
    name: str,
    unit: str | None,
    synonyms: SynonymMap | None = None,
) -> str:
    name_norm = normalize_basic(name)
    if synonyms is not None:
        name_norm = normalize_token(name, synonyms=synonyms, kind="variable")
    unit_norm = normalize_basic(unit or "")
    if synonyms is not None and unit:
        unit_norm = normalize_token(unit, synonyms=synonyms, kind="unit")
    combined = f"{name_norm}::{unit_norm}".strip(":")
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


__all__ = ["variable_fingerprint"]
