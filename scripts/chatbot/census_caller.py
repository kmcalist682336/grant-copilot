"""Async Census API client with cache-first fetch + parallel batching.

Single-vCPU friendly: all parallelism is asyncio (no threads), so many
concurrent fetches share one Python process without contention.

Public API:
    CensusCaller(api_key, cache, *, connect_timeout=5, read_timeout=10)
        async fetch(plan: APIPlanCall) -> dict
        async fetch_all(plans: list[APIPlanCall]) -> list[FetchResult]

Cache key is the full URL (including the API key in production —
multiple keys would otherwise pollute each others' entries).

Errors are reported per-call rather than raised, so a partial-failure
batch returns a mix of success + failure results.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from scripts.chatbot.api_cache import APICache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan + result data classes
# ---------------------------------------------------------------------------

@dataclass
class APIPlanCall:
    """One Census API call to make."""
    url: str                        # full URL minus API key
    table_id: str
    variables: list[str]
    geo_level: str
    geo_filter_ids: list[str]       # IDs to keep after fetch
    year: int
    dataset: str                    # 'acs/acs5', 'dec/pl', etc.
    ttl_seconds: int                # cache TTL for this response

    @property
    def cache_key(self) -> str:
        """Stable identifier for this call (used for logging)."""
        return f"{self.dataset}/{self.year}/{self.table_id}/{self.geo_level}"


@dataclass
class FetchResult:
    """Outcome of one fetch — success carries data, failure carries reason."""
    plan: APIPlanCall
    rows: list[dict]                # parsed (header→value) dicts
    cache_hit: bool
    elapsed_s: float
    error: Optional[str] = None     # None on success

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Caller
# ---------------------------------------------------------------------------

class CensusCaller:
    def __init__(
        self,
        api_key: Optional[str],
        cache: APICache,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        total_timeout: float = 20.0,
        max_concurrent: int = 20,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """
        Args:
            connect_timeout: httpx connect phase timeout (TCP handshake).
            read_timeout: httpx per-read idle timeout.
            total_timeout: HARD wall-clock ceiling per request. httpx's
                per-phase timeouts don't bound total latency (a slow-
                but-steady server can hold a connection open
                indefinitely), so we wrap each call in
                ``asyncio.wait_for(..., total_timeout)`` to guarantee
                progress. Default 20 s — enough for the slowest Census
                endpoints under load, short enough to abandon a stuck
                variable and move on.
            transport: Optional httpx transport — used by tests to inject
                a MockTransport so live calls aren't made.
        """
        self.api_key = api_key
        self.cache = cache
        self.timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=read_timeout,
        )
        self.total_timeout = total_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._transport = transport
        # Reuse a single AsyncClient — connection pooling matters for
        # the many small calls a typical chat turn issues.
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        kwargs: dict = {"timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        self._client = httpx.AsyncClient(**kwargs)
        return self

    async def __aexit__(self, *exc):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _url_with_key(self, url: str) -> str:
        if not self.api_key:
            return url
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}key={self.api_key}"

    def _cache_key_url(self, url: str) -> str:
        """The URL we use for caching — strip the API key so multiple keys
        share the cache. (Census responses don't depend on which key.)"""
        return url

    # ------------------------------------------------------------------
    # Single fetch
    # ------------------------------------------------------------------

    async def fetch(self, plan: APIPlanCall) -> FetchResult:
        """Cache-first; on miss, calls Census and caches the result."""
        import time
        t0 = time.time()
        cache_url = self._cache_key_url(plan.url)
        cached = self.cache.get(cache_url)
        if cached is not None:
            return FetchResult(
                plan=plan,
                rows=self._parse_rows(cached),
                cache_hit=True,
                elapsed_s=time.time() - t0,
            )

        # Live call (semaphore-bounded, wrapped in a hard wall-clock
        # timeout so a single stuck variable can't hold up the whole
        # query response).
        async with self._semaphore:
            try:
                if self._client is None:
                    raise RuntimeError(
                        "CensusCaller used outside async context manager"
                    )
                live_url = self._url_with_key(plan.url)
                resp = await asyncio.wait_for(
                    self._client.get(live_url),
                    timeout=self.total_timeout,
                )
            except asyncio.TimeoutError:
                return FetchResult(
                    plan=plan, rows=[], cache_hit=False,
                    elapsed_s=time.time() - t0,
                    error=f"abandoned_after_{self.total_timeout:.0f}s",
                )
            except httpx.ConnectTimeout as e:
                return FetchResult(
                    plan=plan, rows=[], cache_hit=False,
                    elapsed_s=time.time() - t0,
                    error=f"connect_timeout: {e}",
                )
            except httpx.ReadTimeout as e:
                return FetchResult(
                    plan=plan, rows=[], cache_hit=False,
                    elapsed_s=time.time() - t0,
                    error=f"read_timeout: {e}",
                )
            except httpx.HTTPError as e:
                return FetchResult(
                    plan=plan, rows=[], cache_hit=False,
                    elapsed_s=time.time() - t0,
                    error=f"http_error: {e}",
                )

        if resp.status_code == 204:
            return FetchResult(
                plan=plan, rows=[], cache_hit=False,
                elapsed_s=time.time() - t0,
                error="no_content_204",
            )
        if resp.status_code == 404:
            return FetchResult(
                plan=plan, rows=[], cache_hit=False,
                elapsed_s=time.time() - t0,
                error="not_found_404",
            )
        if resp.status_code != 200:
            return FetchResult(
                plan=plan, rows=[], cache_hit=False,
                elapsed_s=time.time() - t0,
                error=f"http_{resp.status_code}",
            )

        try:
            data = resp.json()
        except ValueError as e:
            return FetchResult(
                plan=plan, rows=[], cache_hit=False,
                elapsed_s=time.time() - t0,
                error=f"bad_json: {e}",
            )

        # Cache the raw [headers, *rows] structure
        try:
            self.cache.set(cache_url, data, ttl_seconds=plan.ttl_seconds)
        except Exception as e:
            logger.warning("Cache set failed for %s: %s", plan.cache_key, e)

        return FetchResult(
            plan=plan,
            rows=self._parse_rows(data),
            cache_hit=False,
            elapsed_s=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Batch fetch
    # ------------------------------------------------------------------

    async def fetch_all(self, plans: list[APIPlanCall]) -> list[FetchResult]:
        """Run every fetch concurrently. Returns same-order results."""
        if not plans:
            return []
        tasks = [self.fetch(p) for p in plans]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rows(census_response: list) -> list[dict]:
        """Census returns [header_row, *data_rows]. Convert to list[dict]."""
        if not census_response or len(census_response) < 2:
            return []
        headers = census_response[0]
        return [dict(zip(headers, row)) for row in census_response[1:]]
