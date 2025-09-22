"""Application configuration via environment variables."""
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
import os
import json
from pathlib import Path


class Settings(BaseSettings):
    """Runtime configuration."""

    sidra_base_url: str = Field(
        default="https://servicodados.ibge.gov.br/api/v3/agregados",
        description="Base URL for the IBGE SIDRA aggregated-data API.",
    )
    database_path: str = Field(
        default="sidra.db",
        description="SQLite database filename (relative to project root unless absolute).",
    )
    embedding_api_url: str = Field(
        default="http://127.0.0.1:1234/v1/embeddings",
        description="LM Studio endpoint for embedding generation.",
    )
    embedding_model: str = Field(
        default="text-embedding-qwen3-embedding-0.6b@f16",
        description="Default embedding model identifier used by LM Studio.",
    )
    request_timeout: float = Field(
        default=30.0,
        description="HTTP timeout in seconds for API calls.",
    )
    request_retries: int = Field(
        default=3,
        description="Number of retry attempts for transient API errors.",
    )
    user_agent: str = Field(
        default="sidra-database/0.1",
        description="User agent string for outbound HTTP requests.",
    )
    municipality_national_threshold: int = Field(
        default=4000,
        description="Minimum municipality count (N6) to flag national coverage.",
    )

    model_config = SettingsConfigDict(
        env_prefix="SIDRA_",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    override = _load_config_file()
    if override:
        return Settings(**override)
    return Settings()


def _load_config_file() -> dict[str, object] | None:
    candidates = [
        Path(os.getenv("SIDRA_CONFIG_PATH", "sidra.config.json")),
        Path("sidra.config.json"),
    ]
    for path in candidates:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to load config file at {path}") from exc
    return None


__all__ = ["Settings", "get_settings"]
