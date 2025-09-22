"""HTTP client for the SIDRA aggregated-data API."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Self

import httpx
import orjson
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import get_settings


class SidraApiError(RuntimeError):
    """Raised when the SIDRA API returns a non-successful response."""


class SidraApiClient:
    """Asynchronous client wrapping SIDRA aggregated-data API endpoints."""

    def __init__(self, *, base_url: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.sidra_base_url,
            timeout=timeout or settings.request_timeout,
            headers={"User-Agent": settings.user_agent},
        )
        self._settings = settings

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(get_settings().request_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, SidraApiError)),
        reraise=True,
    )
    async def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        response = await self._client.get(path, params=params)
        if response.status_code >= 400:
            raise SidraApiError(
                f"SIDRA API request failed ({response.status_code}): {response.text[:200]}"
            )
        return orjson.loads(response.content)

    async def fetch_metadata(self, agregado_id: int) -> Any:
        return await self._get_json(f"/{agregado_id}/metadados")

    async def fetch_periods(self, agregado_id: int) -> Any:
        return await self._get_json(f"/{agregado_id}/periodos")

    async def fetch_localities(self, agregado_id: int, level: str) -> Any:
        return await self._get_json(f"/{agregado_id}/localidades/{level}")

    async def fetch_catalog(
        self,
        *,
        subject_id: int | None = None,
        periodicity: str | None = None,
        levels: list[str] | None = None,
    ) -> Any:
        """Return the agregados catalog grouped by survey/subject."""

        params: dict[str, Any] = {}
        if subject_id is not None:
            params["assunto"] = subject_id
        if periodicity:
            params["periodicidade"] = periodicity
        if levels:
            normalized = [code.upper() for code in levels if code]
            if normalized:
                params["nivel"] = "|".join(normalized)
        return await self._get_json("", params=params or None)

    async def fetch_values(
        self,
        agregado_id: int,
        period: str,
        variable: int,
        *,
        localidades: str | None = None,
        classificacao: str | None = None,
        view: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if localidades:
            params["localidades"] = localidades
        if classificacao:
            params["classificacao"] = classificacao
        if view:
            params["view"] = view
        return await self._get_json(f"/{agregado_id}/periodos/{period}/variaveis/{variable}", params=params)

    async def fetch_latest(self, agregado_id: int, variable: int, **params: Any) -> Any:
        return await self._get_json(f"/{agregado_id}/variaveis/{variable}", params=params)


def fetch_metadata_sync(agregado_id: int) -> Any:
    """Convenience synchronous helper for quick scripts."""

    async def _runner() -> Any:
        async with SidraApiClient() as client:
            return await client.fetch_metadata(agregado_id)

    return asyncio.run(_runner())


__all__ = ["SidraApiClient", "SidraApiError", "fetch_metadata_sync"]
