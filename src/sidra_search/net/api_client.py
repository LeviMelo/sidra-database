from __future__ import annotations

import asyncio
from typing import Any, Mapping, Self

import httpx
import orjson
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import get_settings


class SidraApiError(RuntimeError):
    pass


class SidraApiClient:
    def __init__(self, *, base_url: str | None = None, timeout: float | None = None) -> None:
        s = get_settings()
        # Explicit timeouts so slow servers can't stall forever on read/close
        t = float(timeout or s.request_timeout)
        http_timeout = httpx.Timeout(connect=t, read=t, write=t, pool=t)
        # Keep connections bounded (Windows sockets behave better this way)
        limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

        self._client = httpx.AsyncClient(
            base_url=base_url or s.sidra_base_url,
            timeout=http_timeout,
            headers={"User-Agent": s.user_agent},
            limits=limits,
        )

    async def __aenter__(self) -> Self: return self
    async def __aexit__(self, *_): await self.close()
    async def close(self) -> None: await self._client.aclose()

    @retry(
        stop=stop_after_attempt(get_settings().request_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, SidraApiError)),
        reraise=True,
    )
    async def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        r = await self._client.get(path, params=params)
        if r.status_code in (429,) or r.status_code >= 500:
            raise SidraApiError(f"SIDRA API failed {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"SIDRA API failed {r.status_code}: {r.text[:200]}")
        return orjson.loads(r.content)

    async def fetch_metadata(self, agregado_id: int) -> Any:
        return await self._get_json(f"/{agregado_id}/metadados")

    async def fetch_periods(self, agregado_id: int) -> Any:
        return await self._get_json(f"/{agregado_id}/periodos")

    async def fetch_localities(self, agregado_id: int, level: str) -> Any:
        return await self._get_json(f"/{agregado_id}/localidades/{level}")

    async def fetch_catalog(self, *, subject_id: int | None = None, periodicity: str | None = None,
                            levels: list[str] | None = None) -> Any:
        params: dict[str, Any] = {}
        if subject_id is not None: params["assunto"] = subject_id
        if periodicity: params["periodicidade"] = periodicity
        if levels:
            lv = [c.upper() for c in levels if c]
            if lv: params["nivel"] = "|".join(lv)
        return await self._get_json("", params=params or None)


def fetch_metadata_sync(agregado_id: int) -> Any:
    async def _r() -> Any:
        async with SidraApiClient() as c:
            return await c.fetch_metadata(agregado_id)
    return asyncio.run(_r())


__all__ = ["SidraApiClient", "SidraApiError", "fetch_metadata_sync"]
