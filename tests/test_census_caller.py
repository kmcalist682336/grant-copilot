"""Tests for the async Census caller. All offline — uses MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall, CensusCaller


def _plan(url="https://api.census.gov/data/2023/acs/acs5?get=B17001_001E&for=state:13"):
    return APIPlanCall(
        url=url, table_id="B17001", variables=["B17001_001E"],
        geo_level="state", geo_filter_ids=["13"], year=2023,
        dataset="acs/acs5", ttl_seconds=3600,
    )


@pytest.fixture
def cache(tmp_path):
    return APICache(tmp_path / "cache.db")


def _make_caller_with_transport(cache, handler, total_timeout: float = 20.0):
    """Inject a MockTransport so we never hit the real Census API."""
    return CensusCaller(
        api_key=None, cache=cache,
        transport=httpx.MockTransport(handler),
        total_timeout=total_timeout,
    )


@pytest.mark.asyncio
async def test_fetch_success_and_caches(cache):
    body = [["B17001_001E", "NAME", "state"],
            ["10000000", "Georgia", "13"]]

    def handler(request):
        return httpx.Response(200, json=body)

    caller = _make_caller_with_transport(cache, handler)
    plan = _plan()
    async with caller as c:
        r = await c.fetch(plan)

    assert r.succeeded
    assert r.cache_hit is False
    assert r.rows == [{"B17001_001E": "10000000", "NAME": "Georgia",
                        "state": "13"}]

    # Second call should hit the cache (no live request)
    async with caller as c:
        r2 = await c.fetch(plan)
    assert r2.succeeded
    assert r2.cache_hit is True
    assert r2.rows == r.rows


@pytest.mark.asyncio
async def test_fetch_204_returns_error(cache):
    def handler(request):
        return httpx.Response(204)

    caller = _make_caller_with_transport(cache, handler)
    async with caller as c:
        r = await c.fetch(_plan())
    assert not r.succeeded
    assert "204" in r.error


@pytest.mark.asyncio
async def test_fetch_404_returns_error(cache):
    def handler(request):
        return httpx.Response(404)

    caller = _make_caller_with_transport(cache, handler)
    async with caller as c:
        r = await c.fetch(_plan())
    assert not r.succeeded
    assert "404" in r.error


@pytest.mark.asyncio
async def test_fetch_all_parallel(cache):
    body_a = [["NAME"], ["A"]]
    body_b = [["NAME"], ["B"]]

    def handler(request):
        if "for=state:13" in str(request.url):
            return httpx.Response(200, json=body_a)
        return httpx.Response(200, json=body_b)

    caller = _make_caller_with_transport(cache, handler)
    plans = [
        _plan("https://api.census.gov/data/2023/acs/acs5?get=NAME&for=state:13"),
        _plan("https://api.census.gov/data/2023/acs/acs5?get=NAME&for=state:17"),
    ]
    async with caller as c:
        results = await c.fetch_all(plans)
    assert len(results) == 2
    assert all(r.succeeded for r in results)
    assert results[0].rows[0]["NAME"] == "A"
    assert results[1].rows[0]["NAME"] == "B"


@pytest.mark.asyncio
async def test_bad_json_reports_error(cache):
    def handler(request):
        return httpx.Response(200, content=b"not actually json")

    caller = _make_caller_with_transport(cache, handler)
    async with caller as c:
        r = await c.fetch(_plan())
    assert not r.succeeded
    assert "bad_json" in r.error


@pytest.mark.asyncio
async def test_empty_plan_list(cache):
    caller = _make_caller_with_transport(cache, lambda r: httpx.Response(200, json=[]))
    async with caller as c:
        results = await c.fetch_all([])
    assert results == []


@pytest.mark.asyncio
async def test_fetch_abandons_after_total_timeout(cache):
    """Slow endpoint: total_timeout fires before the MockTransport
    handler returns, so the fetch reports `abandoned_after_Xs`
    instead of hanging the pipeline."""
    import asyncio

    async def slow_handler(request):
        await asyncio.sleep(5)        # will exceed the 0.2 s cap below
        return httpx.Response(200, json=[["NAME"], ["x"]])

    caller = _make_caller_with_transport(
        cache, slow_handler, total_timeout=0.2,
    )
    async with caller as c:
        r = await c.fetch(_plan())
    assert not r.succeeded
    assert "abandoned_after" in r.error
    # Elapsed should be ~the timeout, not the 5 s the handler
    # would have taken — confirming the wait_for guard fired.
    assert r.elapsed_s < 1.0


@pytest.mark.asyncio
async def test_one_slow_variable_doesnt_block_others(cache):
    """fetch_all: even if one plan hits the total_timeout, the
    others still return their data. This is the grant-query
    pattern — one broken variable shouldn't kill the whole turn."""
    import asyncio

    url_fast = "https://api.census.gov/data/2023/acs/acs5?get=FAST&for=state:13"
    url_slow = "https://api.census.gov/data/2023/acs/acs5?get=SLOW&for=state:13"

    async def handler(request):
        if "SLOW" in str(request.url):
            await asyncio.sleep(5)
            return httpx.Response(200, json=[["NAME"], ["slow"]])
        return httpx.Response(200, json=[["NAME", "FAST"], ["fast", "ok"]])

    caller = _make_caller_with_transport(
        cache, handler, total_timeout=0.2,
    )
    plans = [
        APIPlanCall(
            url=url_fast, table_id="F", variables=["FAST"],
            geo_level="state", geo_filter_ids=["13"], year=2023,
            dataset="acs/acs5", ttl_seconds=0,
        ),
        APIPlanCall(
            url=url_slow, table_id="S", variables=["SLOW"],
            geo_level="state", geo_filter_ids=["13"], year=2023,
            dataset="acs/acs5", ttl_seconds=0,
        ),
    ]
    async with caller as c:
        results = await c.fetch_all(plans)
    assert len(results) == 2
    # Fast plan succeeded with real data
    assert results[0].succeeded
    assert results[0].rows[0]["FAST"] == "ok"
    # Slow plan abandoned cleanly
    assert not results[1].succeeded
    assert "abandoned_after" in results[1].error
