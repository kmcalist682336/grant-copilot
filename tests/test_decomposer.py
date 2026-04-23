"""Tests for the LLM decomposer (Phase I)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.chatbot.concept_map import ConceptEntry, ConceptMap, ConceptVariables
from scripts.chatbot.decomposer import (
    DecompositionLLMError, decompose_concept,
)
from scripts.chatbot.decomposition_cache import DecompositionCache
from scripts.chatbot.llm_client import LLMCallError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(canonical: str) -> ConceptEntry:
    return ConceptEntry(
        canonical=canonical, aliases=[],
        table_id="B19013", dataset="acs/acs5",
        variables=ConceptVariables(value="B19013_001E"),
    )


@pytest.fixture
def small_cmap():
    return ConceptMap([
        _entry("median household income"),
        _entry("median gross rent"),
        _entry("racial composition"),
        _entry("bachelor's degree or higher"),
        _entry("poverty rate"),
    ])


class _ScriptedLLM:
    """LLMClient stub returning a queue of pre-baked responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_text": user_text,
            "schema": schema,
            "temperature": temperature,
        })
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Happy path — LLM returns a clean Decomposition
# ---------------------------------------------------------------------------

def test_decompose_returns_decomposition_and_writes_cache(
    small_cmap, tmp_path,
):
    cache = DecompositionCache(tmp_path / "decomp.db")
    llm = _ScriptedLLM([{
        "sub_concepts": [
            "median household income",
            "median gross rent",
            "bachelor's degree or higher",
        ],
        "confidence": 0.85,
        "rationale": "Standard 3-axis displacement-risk proxy.",
    }])
    out = decompose_concept(
        "gentrification", llm, small_cmap, cache=cache,
    )
    assert out.concept == "gentrification"
    assert out.sub_concepts == [
        "median household income",
        "median gross rent",
        "bachelor's degree or higher",
    ]
    assert out.confidence == 0.85
    assert out.rationale.startswith("Standard")
    assert out.model_version == "gemini-2.5-flash"
    # Cached
    cached = cache.get("gentrification")
    assert cached is not None
    assert cached.sub_concepts == out.sub_concepts


def test_cache_hit_skips_llm(small_cmap, tmp_path):
    """A pre-populated cache row short-circuits — LLM is never called."""
    from scripts.chatbot.decomposition_cache import Decomposition
    cache = DecompositionCache(tmp_path / "decomp.db")
    cache.set(Decomposition(
        concept="gentrification",
        sub_concepts=["median household income", "racial composition"],
        confidence=0.9,
    ))
    llm = _ScriptedLLM([])  # empty queue — would assert if called
    out = decompose_concept(
        "gentrification", llm, small_cmap, cache=cache,
    )
    assert llm.calls == []
    assert out.confidence == 0.9


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_system_prompt_includes_curated_canonicals(small_cmap, tmp_path):
    cache = DecompositionCache(tmp_path / "decomp.db")
    llm = _ScriptedLLM([{
        "sub_concepts": ["poverty rate"],
        "confidence": 0.5,
    }])
    decompose_concept(
        "fresh food access", llm, small_cmap, cache=cache,
    )
    prompt = llm.calls[0]["system_prompt"]
    # Every canonical from the small_cmap should appear verbatim.
    for entry in small_cmap:
        assert entry.canonical in prompt, (
            f"canonical {entry.canonical!r} missing from system prompt"
        )


def test_user_text_is_just_the_concept(small_cmap, tmp_path):
    cache = DecompositionCache(tmp_path / "decomp.db")
    llm = _ScriptedLLM([{
        "sub_concepts": ["poverty rate"], "confidence": 0.5,
    }])
    decompose_concept("food deserts", llm, small_cmap, cache=cache)
    assert llm.calls[0]["user_text"] == "food deserts"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_llm_call_failure_raises_decomposition_error(small_cmap):
    class _FailingLLM:
        def extract(self, **kw):
            raise LLMCallError("vertex 503")

    with pytest.raises(DecompositionLLMError, match="vertex 503"):
        decompose_concept("anything", _FailingLLM(), small_cmap)


def test_non_dict_response_raises(small_cmap):
    class _StringLLM:
        def extract(self, **kw):
            return "not a dict"

    with pytest.raises(DecompositionLLMError, match="non-dict"):
        decompose_concept("anything", _StringLLM(), small_cmap)


def test_invalid_output_raises(small_cmap):
    """Missing required field (sub_concepts) → validation error."""
    llm = _ScriptedLLM([{"confidence": 0.5}])
    with pytest.raises(DecompositionLLMError):
        decompose_concept("anything", llm, small_cmap)


def test_too_many_sub_concepts_caught_by_decomposition_validator(small_cmap):
    """LLM emits 6 sub-concepts; Decomposition validator caps at 5."""
    llm = _ScriptedLLM([{
        "sub_concepts": [f"thing {i}" for i in range(6)],
        "confidence": 0.5,
    }])
    with pytest.raises(DecompositionLLMError):
        decompose_concept("anything", llm, small_cmap)


def test_empty_sub_concepts_rejected(small_cmap):
    llm = _ScriptedLLM([{"sub_concepts": [], "confidence": 0.5}])
    with pytest.raises(DecompositionLLMError):
        decompose_concept("anything", llm, small_cmap)


def test_no_cache_still_works(small_cmap):
    """Decomposer should function without a cache (write-through optional)."""
    llm = _ScriptedLLM([{
        "sub_concepts": ["poverty rate"],
        "confidence": 0.5,
    }])
    out = decompose_concept("food deserts", llm, small_cmap, cache=None)
    assert out.sub_concepts == ["poverty rate"]
