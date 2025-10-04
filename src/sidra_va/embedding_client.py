"""Embedding client used exclusively within sidra_va."""
from __future__ import annotations

from typing import Iterable, Sequence

import httpx
import orjson

from .config import get_settings


class EmbeddingClient:
    """Synchronous HTTP client for embedding generation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._base_url = base_url or settings.embedding_api_url
        self._model = model or settings.embedding_model
        self._timeout = timeout or settings.request_timeout
        self._headers = {"Content-Type": "application/json", "User-Agent": settings.user_agent}

    @property
    def model(self) -> str:
        return self._model

    def embed_text(self, text: str, *, model: str | None = None) -> Sequence[float]:
        payload = {
            "model": model or self._model,
            "input": text,
        }
        response = httpx.post(
            self._base_url,
            content=orjson.dumps(payload),
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def embed_batch(self, texts: Iterable[str], *, model: str | None = None) -> list[Sequence[float]]:
        payload = {
            "model": model or self._model,
            "input": list(texts),
        }
        response = httpx.post(
            self._base_url,
            content=orjson.dumps(payload),
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


__all__ = ["EmbeddingClient"]

