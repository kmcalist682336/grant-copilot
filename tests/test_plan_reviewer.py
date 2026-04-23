"""Tests for the post-plan reviewer node."""
from __future__ import annotations

import pytest

from scripts.chatbot.llm_client import LLMCallError
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.nodes.plan_reviewer import (
    PlanReviewClarification, PlanReviewResult, review_plan,
)
from scripts.chatbot.nodes.clarifier import ClarificationOption


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _geo(name, level="county", conf=0.97, tracts=0):
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region",
    ) else level
    return ResolvedGeography(
        geo_id=f"{name.upper()}_ID", geo_level=level, geo_type=level,
        display_name=name, tract_geoids=[f"t{i}" for i in range(tracts)],
        county_geoid=None, api_for_clause=f"{level}:*",
        api_in_clause="", confidence=conf, assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type="administrative"),
    )


def _intent(concept_texts=("median household income",)):
    return ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb County",
                                   ref_type="administrative")],
        concepts=[ExtractedConcept(text=t, canonical_hint=t)
                  for t in concept_texts],
    )


class _StubPlan:
    """Minimal PlanResult-shape stub — matches what review_plan reads."""
    def __init__(self, concept_resolutions, calls):
        self.concept_resolutions = concept_resolutions
        self.calls = calls


class _StubResolution:
    def __init__(self, concept_text, tier="tier_semantic",
                 table_id="B19013", dataset="acs/acs5",
                 canonical_hint=None):
        self.concept = ExtractedConcept(
            text=concept_text, canonical_hint=canonical_hint,
        )
        self.tier = tier
        self.concept_entry = None
        self.table_candidate = None

        class _RR:
            class _T:
                def __init__(self, tid, dset):
                    self.target_table_id = tid
                    self.target_dataset = dset
            def __init__(self, tid, dset):
                self.top_tables = [self._T(tid, dset)]
        self.routed_result = _RR(table_id, dataset)


class _StubCall:
    def __init__(self, table_id, dataset, variables,
                 geo_level="county", year=2024, role="primary"):
        class _API:
            def __init__(self):
                self.table_id = table_id
                self.dataset = dataset
                self.variables = variables
                self.geo_level = geo_level
        self.api_call = _API()
        self.year = year
        self.role = role


# ---------------------------------------------------------------------------
# Happy path — high confidence, no clarifications
# ---------------------------------------------------------------------------

def test_review_returns_high_confidence_for_clear_plan():
    plan = _StubPlan(
        concept_resolutions=[_StubResolution("median household income")],
        calls=[_StubCall("B19013", "acs/acs5", ["B19013_001E"])],
    )
    llm = _ScriptedLLM([{
        "confidence": 0.94,
        "reasoning": "plan is standard B19013 lookup at county level",
        "clarifications": [],
    }])
    result = review_plan(
        query="median household income in DeKalb County",
        intent=_intent(), resolved_geos=[_geo("DeKalb County")],
        plan=plan, llm=llm,
    )
    assert result.confidence == 0.94
    assert result.clarifications == []


def test_review_clamps_more_than_3_clarifications():
    """Even if the LLM returns >3 clarifications, we cap at 3."""
    plan = _StubPlan(concept_resolutions=[], calls=[])
    llm = _ScriptedLLM([{
        "confidence": 0.2,
        "reasoning": "very unclear",
        "clarifications": [
            {"scope": "geo", "question": "q1", "options": []},
            {"scope": "concept", "question": "q2", "options": []},
            {"scope": "other", "question": "q3", "options": []},
            {"scope": "geo", "question": "q4 (should be cut)", "options": []},
            {"scope": "concept", "question": "q5 (should be cut)", "options": []},
        ],
    }])
    result = review_plan(
        query="q", intent=_intent(),
        resolved_geos=[_geo("DeKalb County")], plan=plan, llm=llm,
    )
    assert len(result.clarifications) == 3
    assert result.clarifications[-1].question == "q3"


# ---------------------------------------------------------------------------
# Low confidence → emits clarifications
# ---------------------------------------------------------------------------

def test_review_emits_geo_clarification_for_ambiguous_buckhead():
    plan = _StubPlan(
        concept_resolutions=[_StubResolution(
            "poverty rate", canonical_hint="poverty rate",
            table_id="S1701", dataset="acs/acs5/subject",
        )],
        calls=[_StubCall(
            "S1701", "acs/acs5/subject", ["S1701_C03_001E"],
            geo_level="tract",
        )],
    )
    buckhead = _geo("Buckhead", level="neighborhood", conf=0.84, tracts=33)
    llm = _ScriptedLLM([{
        "confidence": 0.72,
        "reasoning": "geography resolved at only 0.84 with 33 tracts",
        "clarifications": [{
            "scope": "geo",
            "question": "Which Buckhead did you mean?",
            "options": [
                {"label": "Whole Buckhead area (~90k)",
                 "new_geo_text": "Buckhead area",
                 "new_geo_ref_type": "neighborhood"},
                {"label": "North Buckhead (~19k)",
                 "new_geo_text": "North Buckhead",
                 "new_geo_ref_type": "neighborhood"},
            ],
        }],
    }])
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="Buckhead", ref_type="neighborhood")],
        concepts=[ExtractedConcept(text="poverty rate",
                                    canonical_hint="poverty rate")],
    )
    result = review_plan(
        query="poverty rate in Buckhead", intent=intent,
        resolved_geos=[buckhead], plan=plan, llm=llm,
    )
    assert result.confidence < 0.85
    assert len(result.clarifications) == 1
    clar = result.clarifications[0]
    assert clar.scope == "geo"
    assert len(clar.options) == 2
    assert clar.options[0].new_geo_text == "Buckhead area"


def test_review_emits_concept_clarification_for_universe_variable_bug():
    """The specific regression: plan picks B21001_001E (civ pop 18+)
    for a 'veterans' query. Reviewer should emit a concept
    clarification pointing the user at the right variable."""
    plan = _StubPlan(
        concept_resolutions=[_StubResolution("veterans")],
        calls=[_StubCall("B21001", "acs/acs5", ["B21001_001E"])],
    )
    llm = _ScriptedLLM([{
        "confidence": 0.45,
        "reasoning": "plan pulls the civilian-pop-18+ universe, not veterans",
        "clarifications": [{
            "scope": "concept",
            "question": "Total veterans or civilian pop 18+?",
            "options": [
                {"label": "Total veterans",
                 "new_concepts": ["civilian veterans 18 and over"]},
                {"label": "Civilian population 18+",
                 "new_concepts": ["civilian population 18+"]},
            ],
        }],
    }])
    result = review_plan(
        query="how many veterans in Cobb County",
        intent=_intent(concept_texts=("veterans",)),
        resolved_geos=[_geo("Cobb County")], plan=plan, llm=llm,
    )
    assert result.confidence < 0.85
    assert result.clarifications[0].scope == "concept"
    assert result.clarifications[0].options[0].new_concepts == [
        "civilian veterans 18 and over",
    ]


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_review_llm_error_returns_high_confidence():
    """On LLM failure, reviewer returns 1.0 so the pipeline proceeds."""
    class _FailingLLM:
        def extract(self, **kw):
            raise LLMCallError("Vertex 503")
    plan = _StubPlan(concept_resolutions=[], calls=[])
    result = review_plan(
        query="q", intent=_intent(),
        resolved_geos=[_geo("DeKalb County")], plan=plan,
        llm=_FailingLLM(),
    )
    assert result.confidence == 1.0
    assert "reviewer LLM failed" in result.reasoning


def test_review_non_dict_output_returns_high_confidence():
    class _StringLLM:
        def extract(self, **kw):
            return "not a dict"
    plan = _StubPlan(concept_resolutions=[], calls=[])
    result = review_plan(
        query="q", intent=_intent(),
        resolved_geos=[_geo("DeKalb County")], plan=plan,
        llm=_StringLLM(),
    )
    assert result.confidence == 1.0
    assert "non-dict" in result.reasoning


def test_review_hard_timeout_returns_high_confidence():
    """A hung LLM call must not stall the pipeline."""
    import time as _time

    class _HangingLLM:
        def extract(self, **kw):
            _time.sleep(5)
            return {"confidence": 0.5, "clarifications": []}
    plan = _StubPlan(concept_resolutions=[], calls=[])
    result = review_plan(
        query="q", intent=_intent(),
        resolved_geos=[_geo("DeKalb County")], plan=plan,
        llm=_HangingLLM(),
        hard_timeout_s=0.5,
    )
    assert result.confidence == 1.0
    assert "timed out" in result.reasoning.lower()


def test_review_schema_mismatch_returns_high_confidence():
    """Output missing required 'confidence' field → fall through cleanly."""
    llm = _ScriptedLLM([{"clarifications": []}])
    plan = _StubPlan(concept_resolutions=[], calls=[])
    result = review_plan(
        query="q", intent=_intent(),
        resolved_geos=[_geo("DeKalb County")], plan=plan, llm=llm,
    )
    assert result.confidence == 1.0
    assert "schema" in result.reasoning.lower()
