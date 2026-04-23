"""Tests for GrantFrameMatcher + FrameRegistry."""
from __future__ import annotations

from typing import Any

import pytest

from scripts.chatbot.frames import (
    Frame, FrameRegistry, load_default_frames,
)
from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.nodes.grant_frame_matcher import (
    FrameMatch, GrantFrameMatcherError, match_frame,
)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


# ---------------------------------------------------------------------------
# FrameRegistry sanity
# ---------------------------------------------------------------------------

def test_default_registry_loads():
    reg = load_default_frames()
    names = reg.names()
    assert "specific_lookup" in names
    assert "other" in names
    assert "food_access" in names
    assert "demographic_snapshot" in names
    assert len(reg) >= 10


def test_registry_prompt_block_mentions_every_frame():
    reg = load_default_frames()
    block = reg.describe_for_prompt()
    for name in reg.names():
        assert name in block, f"frame {name!r} missing from prompt block"


def test_registry_rejects_missing_required_frames():
    with pytest.raises(ValueError, match="specific_lookup"):
        FrameRegistry(frames=[
            Frame(name="food_access", description="d"),
        ])
    with pytest.raises(ValueError, match="other"):
        FrameRegistry(frames=[
            Frame(name="specific_lookup", description="d"),
        ])


def test_registry_get_unknown_returns_other():
    reg = load_default_frames()
    other = reg.get("other")
    unknown = reg.get("does-not-exist")
    assert unknown.name == "other"
    assert unknown is other


# ---------------------------------------------------------------------------
# match_frame
# ---------------------------------------------------------------------------

def test_match_returns_specific_lookup_for_bare_number_question():
    """Deterministic trigger match on "what's" should short-circuit
    the LLM — the trigger list in frames.yaml has this phrase."""
    llm = _ScriptedLLM({
        "frame": "WRONG_IF_CALLED",     # canary
        "confidence": 0.0,
        "rationale": "",
    })
    out = match_frame(
        "What's the median income in Buckhead?",
        [ExtractedConcept(text="median household income")],
        [ExtractedGeoRef(text="Buckhead", ref_type="neighborhood")],
        llm,
    )
    assert out.frame == "specific_lookup"
    assert out.confidence >= 0.9
    assert len(llm.calls) == 0          # LLM must NOT be called
    assert "deterministic" in out.rationale


def test_match_returns_narrative_frame_for_grant_request():
    """'food pantry' is a food_access trigger; no LLM call needed."""
    llm = _ScriptedLLM({
        "frame": "WRONG_IF_CALLED",
        "confidence": 0.0,
        "rationale": "",
    })
    out = match_frame(
        "I'm writing a food pantry grant for southwest Atlanta.",
        [ExtractedConcept(text="food access")],
        [ExtractedGeoRef(text="southwest Atlanta", ref_type="region")],
        llm,
    )
    assert out.frame == "food_access"
    assert len(llm.calls) == 0


def test_match_food_banks_query_hits_deterministic_food_access():
    """Regression: 'food banks' should trigger food_access without
    an LLM call. Historically the matcher LLM hung on this query
    roughly half the time — the deterministic path was added as a
    fast + reliable short-circuit."""
    llm = _ScriptedLLM({
        "frame": "WRONG_IF_CALLED",
        "confidence": 0.0,
        "rationale": "",
    })
    out = match_frame(
        "I am working on a grant application that will provide funding "
        "for more food banks in Southwest Atlanta.",
        [],
        [],
        llm,
    )
    assert out.frame == "food_access"
    assert len(llm.calls) == 0


def test_match_ambiguous_query_falls_through_to_llm():
    """When trigger matching is ambiguous or absent, the LLM should
    still be consulted."""
    llm = _ScriptedLLM({
        "frame": "other",
        "confidence": 0.5,
        "rationale": "no clear narrative signal",
    })
    # Deliberately contrived query with no frame-specific triggers.
    out = match_frame(
        "quuuuux zorp widgets in the place",
        [],
        [],
        llm,
    )
    assert len(llm.calls) == 1
    assert out.frame == "other"


def test_match_coerces_unknown_frame_to_other():
    """LLM hallucinated a frame name that's not in the registry."""
    llm = _ScriptedLLM({
        "frame": "fabricated_frame",
        "confidence": 0.95,
        "rationale": "the llm invented a frame",
    })
    out = match_frame(
        "bleep bloop", [], [], llm,
    )
    assert out.frame == "other"
    # Confidence is clamped down.
    assert out.confidence <= 0.3
    assert "coerced from unknown frame" in out.rationale


def test_match_invalid_schema_raises():
    llm = _ScriptedLLM({
        # missing "frame" field
        "confidence": 0.5,
        "rationale": "oops",
    })
    with pytest.raises(GrantFrameMatcherError, match="schema validation"):
        match_frame("q", [], [], llm)


def test_match_confidence_out_of_range_raises():
    llm = _ScriptedLLM({
        "frame": "other",
        "confidence": 1.5,  # > 1.0
        "rationale": "x",
    })
    with pytest.raises(GrantFrameMatcherError, match="schema validation"):
        match_frame("q", [], [], llm)


def test_match_llm_failure_wrapped():
    from scripts.chatbot.llm_client import LLMCallError

    class _FailingLLM:
        def extract(self, **kwargs):
            raise LLMCallError("upstream hiccup")

    with pytest.raises(GrantFrameMatcherError, match="LLM call failed"):
        match_frame("q", [], [], _FailingLLM())


def test_match_with_custom_registry():
    custom = FrameRegistry(frames=[
        Frame(name="specific_lookup", description="default"),
        Frame(name="other", description="catch-all"),
        Frame(name="custom_frame", description="just for this test"),
    ])
    llm = _ScriptedLLM({
        "frame": "custom_frame",
        "confidence": 0.9,
        "rationale": "found",
    })
    out = match_frame("q", [], [], llm, registry=custom)
    assert out.frame == "custom_frame"


def test_match_llm_non_dict_raises():
    class _BadLLM:
        def extract(self, **kwargs):
            return "not a dict"
    with pytest.raises(GrantFrameMatcherError, match="non-dict"):
        match_frame("q", [], [], _BadLLM())
