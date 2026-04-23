"""Tests for the synthesizer (Phase I)."""
from __future__ import annotations

import json

import pytest

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.concept_map import ConceptEntry, ConceptVariables
from scripts.chatbot.llm_client import LLMCallError
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, PlannedCall,
)
from scripts.chatbot.frames import Frame
from scripts.chatbot.nodes.anomaly_detector import AnomalyFlag
from scripts.chatbot.nodes.followup_gap_finder import SuggestedFollowup
from scripts.chatbot.nodes.magnitude_contextualizer import MagnitudeFraming
from scripts.chatbot.synthesizer import (
    Citation, CitedVariable, SynthesisError, SynthesizedAnswer,
    enrich_citation_variables, synthesize_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_text": user_text,
        })
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        return self._responses.pop(0)


def _geo(name: str, level: str = "county") -> ResolvedGeography:
    # data_level_available enum doesn't include 'neighborhood'/'corridor'/etc.
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region", "landmark_area",
    ) else level
    return ResolvedGeography(
        geo_id="X", geo_level=level, geo_type=level,
        display_name=name, tract_geoids=[], county_geoid=None,
        api_for_clause=f"{level}:*", api_in_clause="",
        confidence=0.95, assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type="administrative"),
    )


def _entry(canonical: str, table_id: str, dataset: str = "acs/acs5",
           units: str = "dollars") -> ConceptEntry:
    return ConceptEntry(
        canonical=canonical, aliases=[],
        table_id=table_id, dataset=dataset,
        variables=ConceptVariables(value=f"{table_id}_001E"),
        units=units, universe="Households",
    )


def _make_plan(geo: ResolvedGeography, concept: ExtractedConcept,
               entry: ConceptEntry, year: int) -> PlanResult:
    from scripts.chatbot.census_caller import APIPlanCall
    api = APIPlanCall(
        url="http://x", table_id=entry.table_id, variables=[f"{entry.table_id}_001E"],
        geo_level=geo.geo_level, geo_filter_ids=[],
        year=year, dataset=entry.dataset, ttl_seconds=3600,
    )
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0,
        year=year, role="primary", variables=entry.variables,
    )
    cr = ConceptResolution(
        concept=concept, tier="tier_1_concept_map", concept_entry=entry,
    )
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text=geo.display_name,
                                  ref_type="administrative")],
        concepts=[concept],
    )
    return PlanResult(
        intent=intent, resolved_geos=[geo],
        concept_resolutions=[cr], calls=[call],
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_synthesize_returns_answer_with_prose_findings_caveats():
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="median income")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    intent = plan.intent
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=2023,
            value=72500.0, rows_aggregated=1,
        ),
    ])
    llm = _ScriptedLLM([{
        "prose": "DeKalb County's median household income in 2023 "
                 "was $72,500 (B19013, ACS 5-year).",
        "key_findings": ["Median income: $72,500"],
        "caveats": [],
    }])
    out = synthesize_answer(
        "What's DeKalb's median income?", intent, plan, aggregated, llm,
    )
    assert isinstance(out, SynthesizedAnswer)
    assert "DeKalb" in out.prose
    assert "$72,500" in out.prose
    assert out.key_findings == ["Median income: $72,500"]
    # Citation derived from plan, not LLM
    assert len(out.citations) == 1
    assert out.citations[0].table_id == "B19013"
    assert out.citations[0].dataset == "acs/acs5"
    assert out.citations[0].year == 2023
    assert out.citations[0].table_title == "median household income"


def test_synthesize_payload_includes_aggregated_values():
    """The user-message JSON must carry every value so the LLM sees real
    numbers, not abstract structure."""
    geo = _geo("Test")
    concept = ExtractedConcept(text="x")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=2023,
            value=42.0, rows_aggregated=1,
        ),
    ])
    llm = _ScriptedLLM([{
        "prose": "ok", "key_findings": [], "caveats": [],
    }])
    synthesize_answer("x?", plan.intent, plan, aggregated, llm)
    user_text = llm.calls[0]["user_text"]
    payload = json.loads(user_text)
    assert payload["user_query"] == "x?"
    assert len(payload["aggregated_values"]) == 1
    assert payload["aggregated_values"][0]["value"] == 42.0


def test_synthesize_handles_ratio_value():
    geo = _geo("Test")
    concept = ExtractedConcept(text="poverty")
    entry = ConceptEntry(
        canonical="poverty rate", aliases=[],
        table_id="B17001", dataset="acs/acs5",
        variables=ConceptVariables(
            numerator="B17001_002E", denominator="B17001_001E",
        ),
        units="percent",
    )
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=2023,
            ratio=0.135, sample_size=12345, rows_aggregated=1,
        ),
    ])
    llm = _ScriptedLLM([{
        "prose": "Poverty rate was 13.5% in 2023.",
        "key_findings": ["13.5% poverty"],
        "caveats": [],
    }])
    out = synthesize_answer("poverty?", plan.intent, plan, aggregated, llm)
    payload = json.loads(llm.calls[0]["user_text"])
    assert payload["aggregated_values"][0]["ratio"] == 0.135
    assert payload["aggregated_values"][0]["sample_size"] == 12345


def test_synthesize_propagates_aggregation_caveat():
    geo = _geo("Test", "neighborhood")
    concept = ExtractedConcept(text="median income")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=2023,
            value=80000.0, rows_aggregated=5,
            aggregation_caveat="population-weighted mean of tract medians",
        ),
    ])
    llm = _ScriptedLLM([{
        "prose": "Median household income ~$80,000.",
        "key_findings": ["~$80k"],
        "caveats": ["Sub-county median estimated as tract mean."],
    }])
    out = synthesize_answer("hi?", plan.intent, plan, aggregated, llm)
    # The payload exposed the caveat to the LLM
    payload = json.loads(llm.calls[0]["user_text"])
    assert "population-weighted" in payload["aggregated_values"][0]["aggregation_caveat"]
    # And the LLM-emitted caveat made it through
    assert any("tract mean" in c for c in out.caveats)


def test_synthesize_empty_aggregated_still_runs():
    """No data → the LLM is told and should produce a friendly miss."""
    geo = _geo("Test")
    concept = ExtractedConcept(text="x")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[], fetch_failures=[
        "B19013/2023/county: http_500",
    ])
    llm = _ScriptedLLM([{
        "prose": "No data was returned by the Census API.",
        "key_findings": [],
        "caveats": ["1 fetch failure"],
    }])
    out = synthesize_answer("hi?", plan.intent, plan, aggregated, llm)
    assert "No data" in out.prose
    payload = json.loads(llm.calls[0]["user_text"])
    assert payload["fetch_failures"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_synthesize_llm_failure_raises():
    class _FailingLLM:
        def extract(self, **kw):
            raise LLMCallError("503")

    geo = _geo("Test")
    concept = ExtractedConcept(text="x")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[])
    with pytest.raises(SynthesisError, match="503"):
        synthesize_answer("?", plan.intent, plan, aggregated, _FailingLLM())


def test_synthesize_invalid_llm_output_raises():
    geo = _geo("Test")
    concept = ExtractedConcept(text="x")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[])
    llm = _ScriptedLLM([{"key_findings": ["no prose"]}])  # missing prose
    with pytest.raises(SynthesisError):
        synthesize_answer("?", plan.intent, plan, aggregated, llm)


def test_synthesize_non_dict_output_raises():
    geo = _geo("Test")
    concept = ExtractedConcept(text="x")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[])
    class _StringLLM:
        def extract(self, **kw):
            return "not json"
    with pytest.raises(SynthesisError, match="non-dict"):
        synthesize_answer("?", plan.intent, plan, aggregated, _StringLLM())


# ---------------------------------------------------------------------------
# Phase 4: frame + realism-agent pass-through into the user payload
# ---------------------------------------------------------------------------

def _simple_plan(year: int = 2023):
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="median income")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, year)
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=year,
            value=72500.0, rows_aggregated=1,
        ),
    ])
    llm = _ScriptedLLM([{
        "prose": "ok", "key_findings": [], "caveats": [],
    }])
    return plan, aggregated, llm


def _framing(**over) -> MagnitudeFraming:
    base = dict(
        geo_display_name="DeKalb County",
        concept_text="median household income",
        year=2023, value=72500.0,
        vs_msa=1.56, msa_value=46474.0,
        framing_label="high",
        framing_phrase=(
            "substantially above the metro level (1.56×); "
            "up 6.8% from 2018"
        ),
        prior_period_year=2018, prior_period_value=67874.0,
        trend_delta=4626.0, trend_pct=0.068,
        trend_label="rising_mild",
    )
    base.update(over)
    return MagnitudeFraming(**base)


def _anomaly(**over) -> AnomalyFlag:
    base = dict(
        geo_display_name="DeKalb County",
        concept_text="median household income",
        year=2023, value=72500.0,
        anomaly_kind="far_above_comparators",
        magnitude=1.56, triggering_comparator="metro",
        context="1.56× the metro average — far above the expected range",
    )
    base.update(over)
    return AnomalyFlag(**base)


def _followup(q: str, r: str = "fills a grant-narrative gap") -> SuggestedFollowup:
    return SuggestedFollowup(
        question=q, rationale=r,
        frame_relevance="strengthens the food-access case",
    )


def _frame() -> Frame:
    return Frame(
        name="food_access",
        description="grant case around food insecurity / access",
        triggers=["food desert", "food pantry"],
        required_additional_concepts=["SNAP receipt", "grocery access"],
        rhetorical_target="food-access grant paragraph",
        standard_caveats=["Food-access estimates are tract-level approximations."],
    )


def test_phase4_absent_means_no_extra_keys_in_payload():
    """Legacy callers that don't pass Phase 4 kwargs should see the
    original payload shape — no frame/magnitude/anomaly/followup keys."""
    plan, aggregated, llm = _simple_plan()
    synthesize_answer("q?", plan.intent, plan, aggregated, llm)
    payload = json.loads(llm.calls[0]["user_text"])
    assert "frame" not in payload
    assert "magnitude_framings" not in payload
    assert "anomaly_flags" not in payload
    assert "suggested_followups" not in payload


def test_phase4_frame_is_passed_through_in_payload():
    plan, aggregated, llm = _simple_plan()
    synthesize_answer(
        "food access in DeKalb?", plan.intent, plan, aggregated, llm,
        frame=_frame(),
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert payload["frame"]["name"] == "food_access"
    assert payload["frame"]["rhetorical_target"] == "food-access grant paragraph"
    assert "Food-access" in payload["frame"]["standard_caveats"][0]


def test_phase4_empty_lists_stay_out_of_payload():
    """Empty lists are treated as 'absent' — we don't want the LLM
    seeing empty realism-agent slots that could confuse it."""
    plan, aggregated, llm = _simple_plan()
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        magnitude_framings=[], anomaly_flags=[], followups=[],
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert "magnitude_framings" not in payload
    assert "anomaly_flags" not in payload
    assert "suggested_followups" not in payload


def test_phase4_magnitude_framings_serialize_typed_fields():
    plan, aggregated, llm = _simple_plan()
    f = _framing()
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        magnitude_framings=[f],
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert len(payload["magnitude_framings"]) == 1
    mf = payload["magnitude_framings"][0]
    # The synthesizer relies on these exact keys — freeze them.
    assert mf["framing_label"] == "high"
    assert mf["framing_phrase"].startswith("substantially above the metro")
    assert mf["vs_msa"] == 1.56
    assert mf["trend_label"] == "rising_mild"
    assert mf["prior_period_year"] == 2018
    assert mf["trend_pct"] == pytest.approx(0.068)


def test_phase4_anomaly_flags_serialize_typed_fields():
    plan, aggregated, llm = _simple_plan()
    a = _anomaly()
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        anomaly_flags=[a],
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert len(payload["anomaly_flags"]) == 1
    af = payload["anomaly_flags"][0]
    assert af["anomaly_kind"] == "far_above_comparators"
    assert af["magnitude"] == 1.56
    assert af["triggering_comparator"] == "metro"
    assert "far above" in af["context"]


def test_phase4_followups_serialize_as_suggested_followups_key():
    """Synthesizer prompt keys off `suggested_followups` (not
    `followups`) — freeze the wire name so a drive-by rename on the
    node side can't silently break the synthesizer contract."""
    plan, aggregated, llm = _simple_plan()
    fus = [
        _followup("How far is the nearest full-service grocery store?"),
        _followup("What share of households rely on SNAP?"),
    ]
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        followups=fus,
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert "suggested_followups" in payload
    assert len(payload["suggested_followups"]) == 2
    assert payload["suggested_followups"][0]["question"].startswith(
        "How far is the nearest"
    )
    assert "rationale" in payload["suggested_followups"][0]


def test_phase4_all_extras_together_reach_llm():
    """Smoke test that the orchestrator's 'everything on' path stays
    wire-compatible end-to-end."""
    plan, aggregated, llm = _simple_plan()
    out = synthesize_answer(
        "full case?", plan.intent, plan, aggregated, llm,
        frame=_frame(),
        magnitude_framings=[_framing()],
        anomaly_flags=[_anomaly()],
        followups=[_followup("Try the nearest grocery store distance?")],
    )
    assert isinstance(out, SynthesizedAnswer)
    payload = json.loads(llm.calls[0]["user_text"])
    assert payload["frame"]["name"] == "food_access"
    assert payload["magnitude_framings"][0]["framing_label"] == "high"
    assert payload["anomaly_flags"][0]["anomaly_kind"] == "far_above_comparators"
    assert payload["suggested_followups"][0]["question"].startswith("Try the nearest")
    # Citations still derived from the plan, not polluted by Phase 4.
    assert len(out.citations) == 1
    assert out.citations[0].table_id == "B19013"


def test_phase4_accepts_plain_dicts_not_just_pydantic():
    """The synthesizer falls back to raw dicts when items lack
    model_dump() — defensive path for callers that ferry JSON through
    an intermediate cache or test harness."""
    plan, aggregated, llm = _simple_plan()
    raw_framing = {
        "geo_display_name": "DeKalb County",
        "concept_text": "median household income",
        "year": 2023, "value": 72500.0,
        "framing_label": "high", "framing_phrase": "well above peers",
    }
    raw_anomaly = {
        "geo_display_name": "DeKalb County",
        "concept_text": "median household income",
        "year": 2023, "value": 72500.0,
        "anomaly_kind": "far_above_comparators",
        "magnitude": 1.56, "triggering_comparator": "metro",
        "context": "outlier",
    }
    raw_followup = {
        "question": "What's the SNAP uptake rate?",
        "rationale": "completes the food-access case",
    }
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        magnitude_framings=[raw_framing],
        anomaly_flags=[raw_anomaly],
        followups=[raw_followup],
    )
    payload = json.loads(llm.calls[0]["user_text"])
    assert payload["magnitude_framings"][0]["framing_label"] == "high"
    assert payload["anomaly_flags"][0]["anomaly_kind"] == "far_above_comparators"
    assert payload["suggested_followups"][0]["question"].startswith(
        "What's the SNAP"
    )


def test_phase4_system_prompt_teaches_phase4_handling():
    """The system prompt must mention the three realism-agent blocks.
    If someone rips out that prompt section, the LLM will stop using
    the structured fields even though they reach the payload — this
    test catches the prompt-level regression."""
    plan, aggregated, llm = _simple_plan()
    synthesize_answer(
        "q?", plan.intent, plan, aggregated, llm,
        frame=_frame(), magnitude_framings=[_framing()],
    )
    sys = llm.calls[0]["system_prompt"]
    assert "MAGNITUDE FRAMINGS" in sys
    assert "ANOMALY FLAGS" in sys
    assert "FOLLOWUPS" in sys
    assert "GRANT-FRAME HANDLING" in sys
    # And the fetch-failure phrasing — so the Phase 4 work on
    # human-friendly unavailable-variable messaging stays in place.
    assert "could be a transient API problem" in sys


# ---------------------------------------------------------------------------
# Citation variable enrichment — plain-English labels on Sources
# ---------------------------------------------------------------------------

def test_citations_include_every_planned_variable():
    geo = _geo("Test")
    concept = ExtractedConcept(text="poverty rate")
    entry = _entry("poverty rate", "B17001")
    plan = _make_plan(geo, concept, entry, 2023)
    # Extend the plan so the only call references TWO variables.
    plan.calls[0].api_call.variables = ["B17001_001E", "B17001_002E"]

    aggregated = AggregatedResult(values=[])
    llm = _ScriptedLLM([{
        "prose": "p", "key_findings": [], "caveats": [],
    }])
    out = synthesize_answer(
        "q", plan.intent, plan, aggregated, llm,
    )
    assert len(out.citations) == 1
    cit = out.citations[0]
    assert {v.variable_id for v in cit.variables} == {
        "B17001_001E", "B17001_002E",
    }
    # Before enrichment, labels are None.
    assert all(v.label is None for v in cit.variables)


def test_enrich_citation_variables_attaches_labels():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE variables (
            table_id TEXT, dataset TEXT, year INTEGER,
            variable_id TEXT, label_natural TEXT,
            PRIMARY KEY (table_id, dataset, year, variable_id)
        )
    """)
    conn.executemany(
        "INSERT INTO variables VALUES (?,?,?,?,?)",
        [
            ("B17001", "acs/acs5", 2023, "B17001_001E",
             "Population for whom poverty status is determined"),
            ("B17001", "acs/acs5", 2023, "B17001_002E",
             "Population whose income is below the poverty level"),
        ],
    )
    conn.commit()

    cit = Citation(
        table_id="B17001", dataset="acs/acs5", year=2023,
        variables=[
            CitedVariable(variable_id="B17001_001E"),
            CitedVariable(variable_id="B17001_002E"),
        ],
    )
    enriched = enrich_citation_variables([cit], conn)
    labels = {v.variable_id: v.label for v in enriched[0].variables}
    assert labels["B17001_001E"] == (
        "Population for whom poverty status is determined"
    )
    assert labels["B17001_002E"] == (
        "Population whose income is below the poverty level"
    )


def test_enrich_citation_variables_preserves_unknowns():
    """Variables missing from the DB keep variable_id but label=None
    — renderer falls back to showing the id alone."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE variables (
            table_id TEXT, dataset TEXT, year INTEGER,
            variable_id TEXT, label_natural TEXT,
            PRIMARY KEY (table_id, dataset, year, variable_id)
        )
    """)
    conn.commit()

    cit = Citation(
        table_id="B99999", dataset="acs/acs5", year=2023,
        variables=[CitedVariable(variable_id="B99999_001E")],
    )
    enriched = enrich_citation_variables([cit], conn)
    assert enriched[0].variables[0].variable_id == "B99999_001E"
    assert enriched[0].variables[0].label is None


def test_enrich_citation_variables_noop_without_db():
    cit = Citation(
        table_id="B19013", dataset="acs/acs5", year=2023,
        variables=[CitedVariable(variable_id="B19013_001E")],
    )
    enriched = enrich_citation_variables([cit], None)
    # Passes through unchanged rather than blowing up.
    assert enriched[0].variables[0].variable_id == "B19013_001E"
    assert enriched[0].variables[0].label is None
