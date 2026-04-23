"""Tests for the Tier 0.5 decomposition cache."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.chatbot.decomposition_cache import (
    Decomposition, DecompositionCache, MAX_SUB_CONCEPTS,
    normalize_concept,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    return DecompositionCache(tmp_path / "test_decomp.db")


def _gentrification(**overrides) -> Decomposition:
    defaults = dict(
        concept="gentrification",
        sub_concepts=[
            "median home value",
            "median gross rent",
            "racial composition",
            "educational attainment",
            "household income",
        ],
        confidence=0.85,
        rationale="Standard 5-axis proxy set for displacement risk.",
        model_version="gemini-2.5-flash@2026-04-01",
    )
    defaults.update(overrides)
    return Decomposition(**defaults)


# ---------------------------------------------------------------------------
# normalize_concept
# ---------------------------------------------------------------------------

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_concept("  Gentrification  ") == "gentrification"
    assert normalize_concept("Fresh\tFood   Access") == "fresh food access"


def test_normalize_strips_outer_punctuation_keeps_internal():
    assert normalize_concept("'gentrification'.") == "gentrification"
    assert normalize_concept("re-development") == "re-development"


# ---------------------------------------------------------------------------
# Pydantic Decomposition validation
# ---------------------------------------------------------------------------

def test_decomposition_caps_at_max_sub_concepts():
    with pytest.raises(ValidationError):
        Decomposition(
            concept="too many",
            sub_concepts=[f"proxy_{i}" for i in range(MAX_SUB_CONCEPTS + 1)],
            confidence=0.5,
        )


def test_decomposition_dedupes_case_insensitive():
    d = Decomposition(
        concept="dupes",
        sub_concepts=["Income", "income", "INCOME ", "rent"],
        confidence=0.5,
    )
    assert d.sub_concepts == ["Income", "rent"]


def test_decomposition_drops_empty_strings():
    d = Decomposition(
        concept="empties",
        sub_concepts=["income", "", "  ", "rent"],
        confidence=0.5,
    )
    assert d.sub_concepts == ["income", "rent"]


def test_decomposition_requires_at_least_one_sub_concept():
    with pytest.raises(ValidationError):
        Decomposition(
            concept="empty after cleanup",
            sub_concepts=["", "  "],
            confidence=0.5,
        )


def test_decomposition_confidence_bounded():
    with pytest.raises(ValidationError):
        Decomposition(concept="x", sub_concepts=["y"], confidence=1.5)
    with pytest.raises(ValidationError):
        Decomposition(concept="x", sub_concepts=["y"], confidence=-0.1)


# ---------------------------------------------------------------------------
# get / set roundtrip
# ---------------------------------------------------------------------------

def test_get_returns_none_for_missing(cache):
    assert cache.get("never-cached-concept") is None


def test_set_and_get_roundtrip(cache):
    d = _gentrification()
    cache.set(d)
    out = cache.get("gentrification")
    assert out is not None
    assert out.concept == "gentrification"
    assert out.sub_concepts == d.sub_concepts
    assert out.confidence == 0.85
    assert out.rationale == d.rationale
    assert out.model_version == d.model_version


def test_get_is_normalized_key_insensitive(cache):
    cache.set(_gentrification())
    # Whitespace + casing variants must hit the same row
    assert cache.get("Gentrification") is not None
    assert cache.get("  GENTRIFICATION  ") is not None
    assert cache.get("'gentrification'") is not None


def test_set_replaces_existing_entry(cache):
    cache.set(_gentrification(confidence=0.5))
    cache.set(_gentrification(
        confidence=0.95,
        sub_concepts=["income", "rent"],
    ))
    out = cache.get("gentrification")
    assert out.confidence == 0.95
    assert out.sub_concepts == ["income", "rent"]


def test_delete_removes(cache):
    cache.set(_gentrification())
    cache.delete("gentrification")
    assert cache.get("gentrification") is None


# ---------------------------------------------------------------------------
# TTL behavior
# ---------------------------------------------------------------------------

def test_ttl_expires_entry(cache):
    d = _gentrification()
    cache.set(d, ttl_seconds=100, now=1000)
    assert cache.get("gentrification", now=1050) is not None
    assert cache.get("gentrification", now=1101) is None


def test_expired_get_lazy_purges(cache):
    d = _gentrification()
    cache.set(d, ttl_seconds=100, now=1000)
    cache.get("gentrification", now=1101)   # triggers lazy purge
    # Second get within TTL would now miss because the row was deleted
    assert cache.get("gentrification", now=1050) is None


def test_purge_expired_returns_count(cache):
    cache.set(_gentrification(concept="a"), ttl_seconds=100, now=1000)
    cache.set(_gentrification(concept="b"), ttl_seconds=100, now=1000)
    cache.set(_gentrification(concept="c"), ttl_seconds=10000, now=1000)
    assert cache.purge_expired(now=1500) == 2
    # now= must thread through to get() too — otherwise the high-TTL
    # entry looks expired against wall-clock time().
    assert cache.get("a", now=1500) is None
    assert cache.get("c", now=1500) is not None


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def test_list_all_returns_insertion_order(cache):
    cache.set(_gentrification(concept="alpha"), now=100)
    cache.set(_gentrification(concept="beta"), now=200)
    cache.set(_gentrification(concept="gamma"), now=150)
    out = cache.list_all()
    assert [d.concept for d in out] == ["alpha", "gamma", "beta"]


def test_list_all_returns_full_decompositions(cache):
    cache.set(_gentrification())
    out = cache.list_all()
    assert len(out) == 1
    assert isinstance(out[0], Decomposition)
    assert out[0].sub_concepts == _gentrification().sub_concepts


def test_stats_reports_entries_and_avg_confidence(cache):
    cache.set(_gentrification(concept="a", confidence=0.6))
    cache.set(_gentrification(concept="b", confidence=0.8))
    s = cache.stats()
    assert s["entries"] == 2
    assert s["avg_confidence"] == 0.7
    assert s["path"].endswith("test_decomp.db")


def test_empty_cache_stats(cache):
    s = cache.stats()
    assert s["entries"] == 0
    assert s["avg_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Persistence across handles (sanity — same DB file, new instance)
# ---------------------------------------------------------------------------

def test_data_persists_across_cache_instances(tmp_path):
    path = tmp_path / "persist.db"
    DecompositionCache(path).set(_gentrification())

    fresh = DecompositionCache(path)
    out = fresh.get("gentrification")
    assert out is not None
    assert out.sub_concepts == _gentrification().sub_concepts
