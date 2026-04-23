"""Tests for the SQLite API cache."""
from __future__ import annotations

import pytest

from scripts.chatbot.api_cache import APICache


@pytest.fixture
def cache(tmp_path):
    return APICache(tmp_path / "test_cache.db")


def test_get_returns_none_for_missing_url(cache):
    assert cache.get("https://example.com/nope") is None


def test_set_and_get_roundtrip(cache):
    payload = [["NAME", "B17001_001E"], ["Georgia", "10000000"]]
    cache.set("https://api.census.gov/foo", payload, ttl_seconds=3600)
    assert cache.get("https://api.census.gov/foo") == payload


def test_ttl_expires_entry(cache):
    payload = [["NAME"], ["Georgia"]]
    cache.set("https://api.census.gov/foo", payload, ttl_seconds=100, now=1000)
    # Within TTL
    assert cache.get("https://api.census.gov/foo", now=1050) == payload
    # Past TTL
    assert cache.get("https://api.census.gov/foo", now=1101) is None


def test_delete_removes(cache):
    cache.set("https://api.census.gov/foo", [], ttl_seconds=3600)
    cache.delete("https://api.census.gov/foo")
    assert cache.get("https://api.census.gov/foo") is None


def test_purge_expired(cache):
    cache.set("https://api.census.gov/a", [["x"]], ttl_seconds=100, now=1000)
    cache.set("https://api.census.gov/b", [["y"]], ttl_seconds=100, now=2000)
    n = cache.purge_expired(now=1500)
    assert n == 1
    # Read back at simulated time 1500 — `a` is gone (purged), `b` still present
    assert cache.get("https://api.census.gov/a", now=1500) is None
    assert cache.get("https://api.census.gov/b", now=1500) is not None


def test_stats(cache):
    cache.set("u1", [["a", "b"]], ttl_seconds=3600)
    cache.set("u2", [["a", "b"], ["c", "d"]], ttl_seconds=3600)
    s = cache.stats()
    assert s["entries"] == 2
    assert s["total_bytes"] > 0


def test_set_overwrites_existing(cache):
    cache.set("u", "first", ttl_seconds=3600)
    cache.set("u", "second", ttl_seconds=3600)
    assert cache.get("u") == "second"
