"""Tests for the scoped clarifier nodes."""
from __future__ import annotations

from typing import Any

import pytest

from scripts.chatbot.llm_client import LLMCallError
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.nodes.clarifier import (
    ClarificationOption, ClarificationResult,
    apply_clarification, clarify_scope,
    parse_clarification_response, render_clarification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Returns pre-canned dicts from extract() in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        self.calls.append({
            "system_prompt": system_prompt, "user_text": user_text,
        })
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _geo(
    name: str, level: str = "neighborhood", conf: float = 0.82,
    tracts: int = 33,
) -> ResolvedGeography:
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region",
    ) else level
    return ResolvedGeography(
        geo_id=f"{name.upper()}_ID", geo_level=level, geo_type=level,
        display_name=name, tract_geoids=[f"t{i}" for i in range(tracts)],
        county_geoid=None,
        api_for_clause=f"{level}:*", api_in_clause="",
        confidence=conf, assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type="administrative"),
    )


def _intent(query_text="buckhead", concept_texts=("poverty",)) -> ExtractedIntent:
    return ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text=query_text, ref_type="neighborhood")],
        concepts=[ExtractedConcept(text=t) for t in concept_texts],
    )


# ---------------------------------------------------------------------------
# parse_clarification_response
# ---------------------------------------------------------------------------

def test_parse_bare_letter_selects_option():
    opts = [ClarificationOption(label="a"),
            ClarificationOption(label="b"),
            ClarificationOption(label="c")]
    for raw, idx in (("a", 0), ("A", 0), ("b", 1), ("(c)", 2), ("B.", 1),
                     ("a)", 0), (" a ", 0)):
        opt, extra = parse_clarification_response(raw, opts)
        assert opt is opts[idx]
        assert extra == ""


def test_parse_letter_with_extra_freeform_extracts_both():
    opts = [ClarificationOption(label="a"),
            ClarificationOption(label="b")]
    opt, extra = parse_clarification_response(
        "a — but only for pre-2020 data", opts,
    )
    assert opt is opts[0]
    assert extra == "— but only for pre-2020 data"


def test_parse_pure_freeform_returns_no_option():
    opts = [ClarificationOption(label="a"),
            ClarificationOption(label="b")]
    opt, extra = parse_clarification_response(
        "just give me all of north atlanta's tracts", opts,
    )
    assert opt is None
    assert extra == "just give me all of north atlanta's tracts"


def test_parse_out_of_range_letter_is_freeform():
    opts = [ClarificationOption(label="a")]   # only one option
    # "b" or "c" when only (a) exists -> treat as freeform
    opt, extra = parse_clarification_response("b", opts)
    assert opt is None
    assert extra == "b"


def test_parse_empty_response_returns_nothing():
    opts = [ClarificationOption(label="a")]
    opt, extra = parse_clarification_response("", opts)
    assert opt is None
    assert extra == ""


# ---------------------------------------------------------------------------
# render_clarification
# ---------------------------------------------------------------------------

def test_render_emits_lettered_options_and_freeform_footer():
    res = ClarificationResult(
        needs_clarification=True, scope="geo",
        question="Which Buckhead did you mean?",
        options=[
            ClarificationOption(label="All of Buckhead (~90k)"),
            ClarificationOption(label="North Buckhead only (~19k)"),
        ],
    )
    rendered = render_clarification(res)
    assert "Which Buckhead did you mean?" in rendered
    assert "(a) All of Buckhead" in rendered
    assert "(b) North Buckhead only" in rendered
    assert "type your own response" in rendered.lower()


def test_render_handles_no_options_open_ended():
    res = ClarificationResult(
        needs_clarification=True, scope="concept",
        question="Which slice of poverty matters for your grant?",
    )
    rendered = render_clarification(res)
    assert "Which slice" in rendered
    assert "type your own response" in rendered.lower()


# ---------------------------------------------------------------------------
# apply_clarification — geo scope
# ---------------------------------------------------------------------------

def test_apply_geo_clarification_replaces_first_geo_ref():
    intent = _intent(query_text="Buckhead")
    opt = ClarificationOption(
        label="North Buckhead only", new_geo_text="North Buckhead",
    )
    new_intent, new_query = apply_clarification(
        scope="geo", intent=intent, option=opt,
        freeform_extra="", query="poverty in buckhead",
    )
    assert new_intent.geo_refs[0].text == "North Buckhead"
    # No freeform → query unchanged
    assert new_query == "poverty in buckhead"


def test_apply_geo_clarification_with_freeform_appends_to_query():
    intent = _intent(query_text="Buckhead")
    opt = ClarificationOption(
        label="North Buckhead", new_geo_text="North Buckhead",
    )
    new_intent, new_query = apply_clarification(
        scope="geo", intent=intent, option=opt,
        freeform_extra="ignoring the commercial core",
        query="poverty in buckhead",
    )
    assert new_intent.geo_refs[0].text == "North Buckhead"
    assert "ignoring the commercial core" in new_query


def test_apply_geo_clarification_pure_freeform_sets_geo_text():
    intent = _intent(query_text="Atlanta")
    new_intent, new_query = apply_clarification(
        scope="geo", intent=intent, option=None,
        freeform_extra="just Fulton County proper",
        query="poverty in atlanta",
    )
    assert new_intent.geo_refs[0].text == "just Fulton County proper"


def test_apply_geo_clarification_can_change_ref_type():
    intent = _intent()
    opt = ClarificationOption(
        label="Atlanta MSA", new_geo_text="Atlanta MSA",
        new_geo_ref_type="administrative",
    )
    new_intent, _ = apply_clarification(
        scope="geo", intent=intent, option=opt,
        freeform_extra="", query="q",
    )
    assert new_intent.geo_refs[0].text == "Atlanta MSA"
    assert new_intent.geo_refs[0].ref_type == "administrative"


def test_apply_geo_clarification_coerces_city_to_administrative():
    """Regression: the LLM had emitted ``new_geo_ref_type: "city"``
    on a real reviewer clarification, which isn't in the RefType
    literal set and crashed the whole pipeline with a pydantic
    ValidationError. It should now coerce 'city' → 'administrative'
    silently."""
    intent = _intent()
    opt = ClarificationOption(
        label="Atlanta city, Georgia",
        new_geo_text="Atlanta city",
        new_geo_ref_type="city",              # LLM-invented value
    )
    new_intent, _ = apply_clarification(
        scope="geo", intent=intent, option=opt,
        freeform_extra="", query="q",
    )
    assert new_intent.geo_refs[0].text == "Atlanta city"
    assert new_intent.geo_refs[0].ref_type == "administrative"


def test_apply_geo_clarification_drops_unknown_ref_type():
    """When the LLM emits a ref_type we can't map to any legal
    value, we keep the original geo's ref_type rather than crash."""
    # _intent() seeds ref_type='neighborhood'; the unknown value
    # should be dropped, leaving that original type intact.
    intent = _intent()
    opt = ClarificationOption(
        label="???", new_geo_text="Somewhere",
        new_geo_ref_type="zangief_zone",
    )
    new_intent, _ = apply_clarification(
        scope="geo", intent=intent, option=opt,
        freeform_extra="", query="q",
    )
    assert new_intent.geo_refs[0].text == "Somewhere"
    assert new_intent.geo_refs[0].ref_type == "neighborhood"


# ---------------------------------------------------------------------------
# apply_clarification — concept scope
# ---------------------------------------------------------------------------

def test_apply_concept_clarification_replaces_concepts():
    intent = _intent(concept_texts=("poverty",))
    opt = ClarificationOption(
        label="Child poverty",
        new_concepts=["child poverty rate"],
    )
    new_intent, _ = apply_clarification(
        scope="concept", intent=intent, option=opt,
        freeform_extra="", query="q",
    )
    assert [c.text for c in new_intent.concepts] == ["child poverty rate"]


def test_apply_concept_clarification_handles_multiple_new_concepts():
    intent = _intent(concept_texts=("housing",))
    opt = ClarificationOption(
        label="Full housing profile",
        new_concepts=["rent burden", "owner-occupancy rate",
                      "median home value"],
    )
    new_intent, _ = apply_clarification(
        scope="concept", intent=intent, option=opt,
        freeform_extra="", query="q",
    )
    assert len(new_intent.concepts) == 3
    assert "rent burden" in [c.text for c in new_intent.concepts]


def test_apply_concept_clarification_pure_freeform_becomes_new_concept():
    intent = _intent(concept_texts=("help me",))
    new_intent, _ = apply_clarification(
        scope="concept", intent=intent, option=None,
        freeform_extra="senior isolation and poverty",
        query="q",
    )
    assert [c.text for c in new_intent.concepts] == [
        "senior isolation and poverty",
    ]


# ---------------------------------------------------------------------------
# apply_clarification — other scope
# ---------------------------------------------------------------------------

def test_apply_other_clarification_appends_addendum():
    intent = _intent()
    opt = ClarificationOption(
        label="Last 5 years",
        query_addendum="using 2019 to 2024 data",
    )
    _, new_query = apply_clarification(
        scope="other", intent=intent, option=opt,
        freeform_extra="", query="poverty in dekalb",
    )
    assert "using 2019 to 2024 data" in new_query
    assert "poverty in dekalb" in new_query       # original preserved


def test_apply_other_clarification_combines_option_and_freeform():
    intent = _intent()
    opt = ClarificationOption(
        label="Last 5 years", query_addendum="using 2019-2024",
    )
    _, new_query = apply_clarification(
        scope="other", intent=intent, option=opt,
        freeform_extra="weighted by child population",
        query="q",
    )
    assert "using 2019-2024" in new_query
    assert "weighted by child population" in new_query


# ---------------------------------------------------------------------------
# clarify_scope — LLM call error handling
# ---------------------------------------------------------------------------

def test_clarify_scope_llm_error_returns_no_clarification():
    class _FailingLLM:
        def extract(self, **kw):
            raise LLMCallError("fake failure")

    result = clarify_scope(
        scope="geo", query="poverty in buckhead",
        intent=_intent(), resolved_geos=[_geo("Buckhead")],
        llm=_FailingLLM(),
    )
    assert result.needs_clarification is False
    assert result.scope == "geo"


def test_clarify_scope_non_dict_output_returns_no_clarification():
    class _StringLLM:
        def extract(self, **kw):
            return "not a dict"

    result = clarify_scope(
        scope="concept", query="help me",
        intent=_intent(concept_texts=("help me",)),
        resolved_geos=[_geo("DeKalb County", "county", 0.97, 0)],
        llm=_StringLLM(),
    )
    assert result.needs_clarification is False


def test_clarify_scope_happy_path_returns_populated_result():
    llm = _ScriptedLLM([{
        "needs_clarification": True,
        "scope": "geo",
        "question": "Which Buckhead did you mean?",
        "options": [
            {"label": "All of Buckhead (~90k)",
             "new_geo_text": "Buckhead area"},
            {"label": "North Buckhead only",
             "new_geo_text": "North Buckhead"},
        ],
        "rationale": "confidence was 0.82; multiple sub-areas possible",
    }])
    result = clarify_scope(
        scope="geo", query="poverty in buckhead",
        intent=_intent(), resolved_geos=[_geo("Buckhead")],
        llm=llm,
    )
    assert result.needs_clarification is True
    assert result.scope == "geo"
    assert len(result.options) == 2
    assert result.options[0].new_geo_text == "Buckhead area"


def test_clarify_scope_forces_scope_field():
    """LLM echoes a wrong scope → our code overrides."""
    llm = _ScriptedLLM([{
        "needs_clarification": False,
        "scope": "wrong_scope_the_llm_hallucinated",
    }])
    result = clarify_scope(
        scope="concept", query="q",
        intent=_intent(), resolved_geos=[_geo("x")], llm=llm,
    )
    assert result.scope == "concept"


# ---------------------------------------------------------------------------
# End-to-end sanity: parse + apply round-trip
# ---------------------------------------------------------------------------

def test_round_trip_letter_pick_applies_expected_patch():
    """User picks (a), we look up its option, apply the patch."""
    result = ClarificationResult(
        needs_clarification=True, scope="geo",
        question="Which Buckhead?",
        options=[
            ClarificationOption(
                label="All of Buckhead", new_geo_text="Buckhead area",
            ),
            ClarificationOption(
                label="North Buckhead only",
                new_geo_text="North Buckhead",
            ),
        ],
    )
    # User types "b"
    option, freeform = parse_clarification_response("b", result.options)
    assert option is result.options[1]
    assert freeform == ""

    intent = _intent(query_text="Buckhead")
    new_intent, new_query = apply_clarification(
        scope="geo", intent=intent, option=option,
        freeform_extra=freeform, query="poverty in buckhead",
    )
    assert new_intent.geo_refs[0].text == "North Buckhead"
    assert new_query == "poverty in buckhead"   # no freeform → unchanged


def test_clarify_scope_hard_timeout_returns_no_clarification():
    """A hung LLM call must not stall the pipeline. The clarifier
    wraps its LLM call in a wall-clock timeout; on timeout it returns
    a no-clarification result and the orchestrator proceeds."""
    import time as _time

    class _HangingLLM:
        def extract(self, **kw):
            _time.sleep(5)      # longer than our test timeout below
            return {"needs_clarification": False, "scope": kw.get("scope", "geo")}

    result = clarify_scope(
        scope="geo", query="ambiguous",
        intent=_intent(), resolved_geos=[_geo("X")],
        llm=_HangingLLM(),
        hard_timeout_s=0.5,
    )
    assert result.needs_clarification is False
    assert result.scope == "geo"


# ---------------------------------------------------------------------------
# should_run_clarifiers — observable-ambiguity gate
# ---------------------------------------------------------------------------

def _concept_with_hint(text="median household income"):
    return ExtractedConcept(text=text, canonical_hint=text)


def test_gate_skips_clear_unambiguous_query():
    """Top geo conf 0.97, concept pinned, no trend/vague markers → skip."""
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb County", ref_type="administrative")],
        concepts=[_concept_with_hint()],
    )
    resolved = [_geo("DeKalb County", "county", conf=0.97, tracts=0)]
    run, reason = should_run_clarifiers(
        query="median household income in DeKalb County",
        intent=intent, resolved_geos=resolved,
    )
    assert run is False
    assert "clear" in reason.lower() or "conf" in reason.lower()


def test_gate_triggers_on_low_confidence_geo():
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="Buckhead", ref_type="neighborhood")],
        concepts=[_concept_with_hint("poverty rate")],
    )
    resolved = [_geo("Buckhead", "neighborhood", conf=0.82, tracts=33)]
    run, reason = should_run_clarifiers(
        query="poverty rate in Buckhead",
        intent=intent, resolved_geos=resolved,
    )
    assert run is True
    assert "0.82" in reason or "confidence" in reason.lower()


def test_gate_triggers_on_unpinned_concept():
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb County", ref_type="administrative")],
        concepts=[ExtractedConcept(text="poverty", canonical_hint=None)],
    )
    resolved = [_geo("DeKalb County", "county", conf=0.97, tracts=0)]
    run, reason = should_run_clarifiers(
        query="poverty in DeKalb County",
        intent=intent, resolved_geos=resolved,
    )
    assert run is True
    assert "canonical_hint" in reason


def test_gate_does_not_trigger_on_trend_when_geo_and_concept_clear():
    """Trend markers alone don't trigger the clarifier anymore — the
    year-window default is fine when geo + concept are confident.
    Only geo / concept ambiguity should fire the gate."""
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb County", ref_type="administrative")],
        concepts=[_concept_with_hint("child poverty rate")],
        years=[],
    )
    resolved = [_geo("DeKalb County", "county", conf=0.97, tracts=0)]
    run, _ = should_run_clarifiers(
        query="how has child poverty changed over time in DeKalb",
        intent=intent, resolved_geos=resolved,
    )
    assert run is False


def test_gate_does_not_trigger_on_vague_phrasing_alone():
    """Vague phrasing like 'help me' / 'data on' is no longer a
    trigger when geo + concept are well-resolved. The downstream
    pipeline handles the default output cleanly."""
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb County", ref_type="administrative")],
        concepts=[_concept_with_hint("median household income")],
    )
    resolved = [_geo("DeKalb County", "county", conf=0.97, tracts=0)]
    run, _ = should_run_clarifiers(
        query="help me with some data on DeKalb County",
        intent=intent, resolved_geos=resolved,
    )
    assert run is False


def test_gate_triggers_on_multi_geo():
    from scripts.chatbot.nodes.clarifier import should_run_clarifiers
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="DeKalb County", ref_type="administrative"),
            ExtractedGeoRef(text="Fulton County", ref_type="administrative"),
        ],
        concepts=[_concept_with_hint()],
    )
    resolved = [_geo("DeKalb County", "county", conf=0.97, tracts=0)]
    run, reason = should_run_clarifiers(
        query="compare median income in DeKalb and Fulton",
        intent=intent, resolved_geos=resolved,
    )
    assert run is True
    assert "geo refs" in reason.lower()
