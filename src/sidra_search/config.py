from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    # HTTP
    sidra_base_url: str = "https://servicodados.ibge.gov.br/api/v3/agregados"
    request_timeout: float = 30.0
    request_retries: int = 3
    user_agent: str = "sidra-search/0.1"

    # DB
    database_timeout: float = 60.0
    municipality_national_threshold: int = 5000

    # Embeddings (optional, for title semantics)
    embedding_api_url: str = "http://127.0.0.1:1234/v1/embeddings"
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b@f16"

    # Features
    enable_titles_fts: bool = True
    enable_title_embeddings: bool = True


def _env(name: str) -> str | None:
    for key in (name, name.upper(), name.lower()):
        if key in os.environ:
            return os.environ[key]
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    d = Settings().__dict__.copy()

    # strings
    for f in ("sidra_base_url", "user_agent", "embedding_api_url", "embedding_model"):
        v = _env(f"SIDRA_SEARCH_{f.upper()}")
        if v:
            d[f] = v

    # numerics/bools
    def _float(env, key):
        v = _env(env)
        if v:
            try: d[key] = float(v)
            except: pass

    def _int(env, key):
        v = _env(env)
        if v:
            try: d[key] = int(v)
            except: pass

    def _bool(env, key):
        v = _env(env)
        if v is not None:
            d[key] = v not in ("0", "false", "False", "")

    _float("SIDRA_SEARCH_REQUEST_TIMEOUT", "request_timeout")
    _float("SIDRA_SEARCH_DATABASE_TIMEOUT", "database_timeout")
    _int("SIDRA_SEARCH_REQUEST_RETRIES", "request_retries")
    _int("SIDRA_SEARCH_MUNICIPALITY_NATIONAL_THRESHOLD", "municipality_national_threshold")
    _bool("SIDRA_SEARCH_ENABLE_TITLES_FTS", "enable_titles_fts")
    _bool("SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS", "enable_title_embeddings")

    return Settings(**d)
