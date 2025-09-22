"""Application configuration via environment variables."""
from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


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

    class Config:
        env_prefix = "SIDRA_"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
