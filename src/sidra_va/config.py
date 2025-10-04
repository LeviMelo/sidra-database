"""Configuration helpers for the sidra_va package."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class VaSettings:
    """Runtime configuration for sidra_va components."""

    embedding_api_url: str = "http://127.0.0.1:1234/v1/embeddings"
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b@f16"
    request_timeout: float = 30.0
    user_agent: str = "sidra-va/0.1"
    database_timeout: float = 60.0


def _lookup_env(name: str) -> str | None:
    candidates = {name, name.upper(), name.lower()}
    for candidate in candidates:
        if candidate in os.environ:
            return os.environ[candidate]
    return None


def _load_settings() -> VaSettings:
    defaults = VaSettings()
    prefix = "SIDRA_VA_"
    overrides: dict[str, object] = {}
    for field in ("embedding_api_url", "embedding_model", "user_agent"):
        value = _lookup_env(f"{prefix}{field.upper()}")
        if value:
            overrides[field] = value
    timeout_raw = _lookup_env(f"{prefix}REQUEST_TIMEOUT")
    if timeout_raw:
        try:
            overrides["request_timeout"] = float(timeout_raw)
        except ValueError:
            pass
    db_timeout_raw = _lookup_env(f"{prefix}DATABASE_TIMEOUT")
    if db_timeout_raw:
        try:
            overrides["database_timeout"] = float(db_timeout_raw)
        except ValueError:
            pass
    if overrides:
        return VaSettings(**{**defaults.__dict__, **overrides})
    return defaults


@lru_cache(maxsize=1)
def get_settings() -> VaSettings:
    """Return cached VA configuration settings."""

    return _load_settings()


__all__ = ["VaSettings", "get_settings"]

