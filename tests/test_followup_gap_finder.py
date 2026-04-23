"""Tests for FollowupGapFinder."""
from __future__ import annotations

from typing import Any

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.frames import Frame
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ResolvedGeography,
)
from scripts.chatbot.nodes.followup_gap_finder import (
    SuggestedFollowup, _compute_gap, find_followups,
)


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _geo() -> ResolvedGeography:
    return ResolvedGeography(
        geo_id="x", geo_level="neighborhood", geo_type="n",
        display_name="X", api_for_clause="a", confidence=1.0,
        data_level_available="tract",
        source_ref=ExtractedGeoRef(text="X", ref_type="administrative"),
    )


def _value(concept_text: str, value: float | None = 1.0,
           hint: str | None = None) -> AggregatedValue:
    return AggregatedValue(
        geo=_geo(),
        concept=ExtractedConcept(text=concept_text, canonical_hint=hint),
        year=2024, role="primary", value=value,
    )


def _frame(
    name: str = "food_access",
    additional: list[str] | None = None,
) -> Frame:
    return Frame(
        name=name,
        description="d",
        triggers=[],
        required_additional_concepts=additional or [
            "SNAP participation rate",
            "share without a vehicle",
            "child poverty rate",
        ],
        rhetorical_target="food_access case",
        standard_caveats=[],
    )


# ---------------------------------------------------------------------------
# _compute_gap
# ---------------------------------------------------------------------------

def test_gap_identifies_missing_concepts():
    frame = _frame()
    agg = AggregatedResult(values=[
        _value("poverty rate", 0.23),
        _value("median household income", 57_568),
    ])
    fetched, missing = _compute_gap(frame, agg)
    # Nothing in the frame's required list was fetched → all missing
    assert set(missing) == {
        "SNAP participation rate",
        "share without a vehicle",
        "child poverty rate",
    }
    assert "poverty rate" in fetched
    assert "median household income" in fetched


def test_gap_partial_fetched_reduces_missing():
    frame = _frame()
    agg = AggregatedResult(values=[
        _value("SNAP participation rate", 0.27),   # covered
        _value("poverty rate", 0.23),
    ])
    _, missing = _compute_gap(frame, agg)
    assert "SNAP participation rate" not in missing
    assert "share without a vehicle" in missing
    assert "child poverty rate" in missing


def test_gap_fetched_matches_via_canonical_hint():
    frame = _frame()
    agg = AggregatedResult(values=[
        _value("snap", 0.27, hint="SNAP participation rate"),
    ])
    _, missing = _compute_gap(frame, agg)
    # "SNAP participation rate" should be considered fetched
    # because the canonical_hint matches.
    assert "SNAP participation rate" not in missing


def test_gap_values_with_null_values_are_not_fetched():
    """A fetch that returned no rows (value=None, ratio=None) doesn't
    count as fetched."""
    frame = _frame()
    agg = AggregatedResult(values=[
        _value("SNAP participation rate", None),
    ])
    _, missing = _compute_gap(frame, agg)
    assert "SNAP participation rate" in missing


def test_gap_no_frame_returns_fetched_only():
    agg = AggregatedResult(values=[_value("poverty rate", 0.18)])
    fetched, missing = _compute_gap(None, agg)
    assert fetched == ["poverty rate"]
    assert missing == []


# ---------------------------------------------------------------------------
# find_followups — happy path
# ---------------------------------------------------------------------------

def test_find_followups_returns_llm_suggestions():
    llm = _ScriptedLLM({
        "followups": [
            {
                "question": "What share of households here receive SNAP?",
                "rationale": "core food-access concept",
                "frame_relevance": "food_access",
            },
            {
                "question": "How many households don't have a vehicle?",
                "rationale": "grocery access proxy",
                "frame_relevance": "food_access",
            },
        ],
    })
    agg = AggregatedResult(values=[_value("poverty rate", 0.23)])
    out = find_followups(
        user_query="Food pantry grant case",
        frame=_frame(),
        aggregated=agg,
        llm=llm,
    )
    assert len(out) == 2
    assert isinstance(out[0], SuggestedFollowup)
    assert "SNAP" in out[0].question
    assert len(llm.calls) == 1


def test_find_followups_caps_at_four():
    """If LLM returns 6 items (ignoring prompt), we keep only 4."""
    items = [
        {"question": f"q{i}", "rationale": "r", "frame_relevance": "f"}
        for i in range(6)
    ]
    llm = _ScriptedLLM({"followups": items})
    out = find_followups(
        user_query="x", frame=_frame(),
        aggregated=AggregatedResult(values=[_value("poverty rate")]),
        llm=llm,
    )
    assert len(out) == 4


def test_find_followups_on_empty_inputs_returns_empty():
    llm = _ScriptedLLM({"followups": []})
    out = find_followups(
        user_query="",
        frame=None,
        aggregated=AggregatedResult(values=[]),
        llm=llm,
    )
    assert out == []
    assert llm.calls == []   # short-circuits before LLM


def test_find_followups_survives_llm_failure():
    """FollowupGapFinder is non-critical — LLM errors return empty
    list rather than raising."""
    from scripts.chatbot.llm_client import LLMCallError

    class _Fail:
        def extract(self, **kwargs):
            raise LLMCallError("boom")

    out = find_followups(
        user_query="q", frame=_frame(),
        aggregated=AggregatedResult(values=[_value("poverty rate")]),
        llm=_Fail(),
    )
    assert out == []


def test_find_followups_survives_invalid_schema():
    llm = _ScriptedLLM({"followups": [{"question": "x"}]})   # missing rationale
    out = find_followups(
        user_query="q", frame=_frame(),
        aggregated=AggregatedResult(values=[_value("poverty rate")]),
        llm=llm,
    )
    assert out == []


def test_find_followups_survives_non_dict_response():
    class _Bad:
        def extract(self, **kwargs):
            return "not a dict"
    out = find_followups(
        user_query="q", frame=_frame(),
        aggregated=AggregatedResult(values=[_value("poverty rate")]),
        llm=_Bad(),
    )
    assert out == []


def test_find_followups_works_without_frame():
    """When no frame matched (pure-lookup queries), follow-ups still
    make sense — just less guided."""
    llm = _ScriptedLLM({
        "followups": [
            {
                "question": "How has this changed recently?",
                "rationale": "trend context",
            },
        ],
    })
    out = find_followups(
        user_query="What's the median income here?",
        frame=None,
        aggregated=AggregatedResult(values=[_value("median household income", 75000)]),
        llm=llm,
    )
    assert len(out) == 1
    assert "recently" in out[0].question.lower()
