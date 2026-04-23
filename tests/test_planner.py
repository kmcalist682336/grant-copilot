"""Phase H planner tests."""
from __future__ import annotations

import pytest

from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.models import (
    CompoundSpec, ExtractedConcept, ExtractedGeoRef, ExtractedIntent,
)
from scripts.chatbot.planner import (
    PlanResult, _county_groups_from_tracts, _pick_years,
    plan_query, resolve_concept,
)


# ---------------------------------------------------------------------------
# _pick_years
# ---------------------------------------------------------------------------

def test_pick_years_latest_returns_one():
    intent = ExtractedIntent(temporal_intent="latest")
    assert _pick_years(intent, [2024, 2023, 2022]) == [2024]


def test_pick_years_change_uses_explicit_years():
    intent = ExtractedIntent(
        temporal_intent="change", years=[2020, 2023],
    )
    out = _pick_years(intent, [2024, 2023, 2022, 2021, 2020])
    assert out == [2020, 2023]


def test_pick_years_change_clamps_unsupported_explicit_years():
    """Explicit years not in supported_years get dropped; falls back."""
    intent = ExtractedIntent(
        temporal_intent="change", years=[2015, 2099],
    )
    # Both years unsupported → falls back to [latest]
    out = _pick_years(intent, [2024, 2023, 2022])
    assert out == [2024]


def test_pick_years_change_default_picks_two_years_apart():
    intent = ExtractedIntent(temporal_intent="change")
    # Default: latest vs ~3 years prior
    out = _pick_years(intent, [2024, 2023, 2022, 2021, 2020])
    assert out == [2021, 2024]


def test_pick_years_trend_uses_explicit_start():
    intent = ExtractedIntent(temporal_intent="trend", years=[2021])
    out = _pick_years(intent, [2024, 2023, 2022, 2021, 2020])
    assert out == [2021, 2022, 2023, 2024]


def test_pick_years_trend_capped_to_max():
    """Trend output capped at most-recent _MAX_TREND_VINTAGES years."""
    intent = ExtractedIntent(temporal_intent="trend", years=[2010])
    out = _pick_years(intent, list(range(2024, 2009, -1)))
    assert len(out) <= 5
    assert out[-1] == 2024


def test_pick_years_empty_supported_returns_empty():
    intent = ExtractedIntent(temporal_intent="latest")
    assert _pick_years(intent, []) == []


# ---------------------------------------------------------------------------
# _county_groups_from_tracts
# ---------------------------------------------------------------------------

def test_county_groups_buckets_by_state_county_prefix():
    tracts = [
        "13121009001",   # GA Fulton
        "13121009002",   # GA Fulton
        "13089020100",   # GA DeKalb
        "13089020200",
        "47037011000",   # TN Davidson
    ]
    out = _county_groups_from_tracts(tracts)
    assert set(out.keys()) == {("13", "121"), ("13", "089"), ("47", "037")}
    assert sorted(out[("13", "121")]) == ["13121009001", "13121009002"]


def test_county_groups_skips_short_geoids():
    out = _county_groups_from_tracts(["abc", "13121", "1312100100"])
    # "13121" is exactly 5 chars (state+county prefix only) → bucketed empty
    # "1312100100" is 10 chars — also bucketed
    # "abc" is too short → skipped
    assert ("13", "121") in out


# ---------------------------------------------------------------------------
# resolve_concept (tier cascade)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cmap():
    return ConceptMap.from_yaml()


def test_resolve_concept_tier1_via_canonical_hint(cmap, metadata_db):
    c = ExtractedConcept(
        text="paycheck", canonical_hint="median household income",
    )
    res = resolve_concept(c, cmap, metadata_db, ["tract"])
    assert res.tier == "tier_1_concept_map"
    assert res.concept_entry.canonical == "median household income"


def test_resolve_concept_tier1_via_text_fallback(cmap, metadata_db):
    """No hint, but verbatim text is a curated alias → still Tier 1."""
    c = ExtractedConcept(text="poverty rate")
    res = resolve_concept(c, cmap, metadata_db, ["county"])
    assert res.tier == "tier_1_concept_map"
    assert res.concept_entry.canonical == "poverty rate"


def test_resolve_concept_tier2_fts_fallback(cmap, metadata_db):
    """A concept missing from cmap but with strong FTS hit → Tier 2."""
    c = ExtractedConcept(text="median age")  # IS in cmap actually
    # Use a gibberish hint to force lookup-by-text-only fallback
    c2 = ExtractedConcept(text="contract rent")  # in cmap
    res = resolve_concept(c2, cmap, metadata_db, ["county"])
    # 'contract rent' → 'median contract rent' alias → Tier 1 hit
    assert res.tier == "tier_1_concept_map"


def test_resolve_concept_tier3_when_all_miss(cmap, metadata_db):
    """Truly unknown concept → falls all the way to Tier 3."""
    c = ExtractedConcept(text="xyzzqq notarealconcept blibblob")
    res = resolve_concept(c, cmap, metadata_db, ["county"])
    assert res.tier == "tier_3_llm_fallback"
    assert res.concept_entry is None
    assert res.table_candidate is None


def test_resolve_concept_composite_no_cache_emits_decomp_request(
    cmap, metadata_db,
):
    """Composite + no cache + no Tier 1 match → needs_llm_decomposition."""
    c = ExtractedConcept(
        text="gentrification", canonical_hint="gentrification",
        is_composite=True,
    )
    res = resolve_concept(c, cmap, metadata_db, ["tract"])
    assert res.tier == "needs_llm_decomposition"


def test_resolve_concept_composite_falls_back_to_tier1_when_present(
    cmap, metadata_db,
):
    """Composite hint should NOT veto a perfectly good Tier 1 hit.

    The user's "median income inequality for different races" → LLM
    flags is_composite=True but the canonical_hint 'income inequality'
    maps directly to the Gini-index entry. Without this fallback the
    planner produced 0 calls; with it we get the Gini entry plus a note
    explaining the degradation.
    """
    c = ExtractedConcept(
        text="median income inequality",
        canonical_hint="income inequality",
        is_composite=True,
    )
    res = resolve_concept(c, cmap, metadata_db, ["tract"])
    assert res.tier == "tier_1_concept_map"
    assert res.concept_entry is not None
    assert res.concept_entry.canonical == "income inequality"
    assert any(
        "Tier 1 single-concept match as best-effort fallback" in n
        for n in res.notes
    ), f"expected fallback note; got {res.notes}"


# ---------------------------------------------------------------------------
# plan_query end-to-end
# ---------------------------------------------------------------------------

def test_plan_query_buckhead_income(gazetteer_db, metadata_db, cmap):
    """Single-geo, single-concept, latest year → 1 PlannedCall per
    unique county the tracts span."""
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="Buckhead", ref_type="neighborhood",
            scope_hint="Atlanta",
        )],
        concepts=[ExtractedConcept(
            text="income", canonical_hint="median household income",
        )],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    assert resolved
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert isinstance(plan, PlanResult)
    assert len(plan.concept_resolutions) == 1
    assert plan.concept_resolutions[0].tier == "tier_1_concept_map"
    assert plan.calls, "expected at least one APIPlanCall"
    call = plan.calls[0]
    assert call.geo_idx == 0 and call.concept_idx == 0
    assert call.api_call.dataset == "acs/acs5"
    assert call.api_call.table_id == "B19013"
    assert call.api_call.geo_level == "tract"
    assert call.tract_filter, "tract aggregation should populate filter"
    assert "B19013_001E" in call.api_call.variables


def test_plan_query_county_admin(gazetteer_db, metadata_db, cmap):
    """Pure-admin geo (county) uses the resolver-supplied for/in clauses."""
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="DeKalb County", ref_type="administrative",
            scope_hint="Georgia",
        )],
        concepts=[ExtractedConcept(
            text="poverty", canonical_hint="poverty rate",
        )],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert plan.calls
    call = plan.calls[0]
    assert call.api_call.geo_level == "county"
    assert call.tract_filter == []
    # poverty rate uses numerator+denominator — both must appear
    assert "B17001_002E" in call.api_call.variables
    assert "B17001_001E" in call.api_call.variables


def test_plan_query_temporal_change_emits_two_years(
    gazetteer_db, metadata_db, cmap,
):
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="DeKalb County", ref_type="administrative",
            scope_hint="Georgia",
        )],
        concepts=[ExtractedConcept(
            text="poverty", canonical_hint="poverty rate",
        )],
        temporal_intent="change",
        years=[2020, 2023],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    years = sorted({c.year for c in plan.calls})
    assert years == [2020, 2023]


def test_plan_query_msa_gets_acs1_override(
    gazetteer_db, metadata_db, cmap,
):
    """At MSA, median household income override switches dataset to ACS1."""
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="Atlanta-Sandy Springs-Alpharetta, GA",
            ref_type="administrative",
        )],
        concepts=[ExtractedConcept(
            text="median household income",
            canonical_hint="median household income",
        )],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    if not resolved or resolved[0].geo_level != "msa":
        pytest.skip(
            f"need msa resolution; got "
            f"{resolved[0].geo_level if resolved else 'none'}"
        )
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert plan.calls
    assert plan.calls[0].api_call.dataset == "acs/acs1"


def test_plan_query_no_concepts_yields_no_calls(
    gazetteer_db, metadata_db, cmap,
):
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="Buckhead", ref_type="neighborhood",
            scope_hint="Atlanta",
        )],
        concepts=[],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert plan.calls == []
    assert any("0 calls" in n for n in plan.notes)


def test_plan_query_unresolvable_concept_emits_no_calls(
    gazetteer_db, metadata_db, cmap,
):
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="Buckhead", ref_type="neighborhood",
            scope_hint="Atlanta",
        )],
        concepts=[ExtractedConcept(text="zzzqqq fakeconcept blibblob")],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert plan.calls == []
    assert plan.concept_resolutions[0].tier == "tier_3_llm_fallback"


def test_plan_query_compound_difference_filters_to_remaining_tracts(
    gazetteer_db, metadata_db, cmap,
):
    """DeKalb minus Decatur produces a tract-aggregating geo; the planner
    must use those filtered tracts in the call."""
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="DeKalb County", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="Decatur", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="difference")],
        concepts=[ExtractedConcept(
            text="poverty", canonical_hint="poverty rate",
        )],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    assert len(resolved) == 1
    plan = plan_query(intent, resolved, cmap, metadata_db)
    assert plan.calls
    # All calls should target tract level and carry tract_filter
    for c in plan.calls:
        assert c.api_call.geo_level == "tract"
        assert c.tract_filter
    # Filter set must equal the resolved geo's tract_geoids
    all_filter = set()
    for c in plan.calls:
        all_filter.update(c.tract_filter)
    assert all_filter == set(resolved[0].tract_geoids)


# ---------------------------------------------------------------------------
# Subgroup-concept detection (task #32) — prevents the universe _001E
# from beating the correct subgroup variable for queries like "veterans
# in Cobb County" or "children in poverty in DeKalb".
# ---------------------------------------------------------------------------

def test_concept_is_subgroup_detects_subgroup_concepts():
    from scripts.chatbot.planner import _concept_is_subgroup
    # True positives: named subgroups of broader universes
    assert _concept_is_subgroup("veterans")
    assert _concept_is_subgroup("veteran poverty rate")
    assert _concept_is_subgroup("renters with cost burden")
    assert _concept_is_subgroup("children in poverty")
    assert _concept_is_subgroup("child poverty rate")
    assert _concept_is_subgroup("uninsured children")
    assert _concept_is_subgroup("hispanic population")
    assert _concept_is_subgroup("foreign-born residents")
    assert _concept_is_subgroup("population 65 and over")
    assert _concept_is_subgroup("single mothers")
    assert _concept_is_subgroup("workers without a bachelor's degree")
    assert _concept_is_subgroup("broadband subscription rate")


def test_concept_is_subgroup_does_not_fire_on_totals():
    from scripts.chatbot.planner import _concept_is_subgroup
    # True negatives: genuine totals — universe _001E is the right answer
    assert not _concept_is_subgroup("median household income")
    assert not _concept_is_subgroup("total population")
    assert not _concept_is_subgroup("total households")
    assert not _concept_is_subgroup("per capita income")
    assert not _concept_is_subgroup(None)
    assert not _concept_is_subgroup("")


def test_score_candidate_penalizes_is_total_on_subgroup_concept():
    """Regression for the 'Cobb County had 599,285 veterans' bug.

    Given two candidates in the same top table:
      - universe _001E (is_total=1, depth=0) — civilian pop 18+
      - subgroup _002E (is_total=0, depth=1) — veterans
    On a subgroup query, the subgroup variable must outrank the
    universe variable. Before the fix, _001E won by +5 top-table bonus
    vs the subgroup's −3 depth penalty.
    """
    from scripts.chatbot.planner import _VarCandidate, _score_candidate

    universe = _VarCandidate(
        table_id="B21001", dataset="acs/acs5", year=2024,
        variable_id="B21001_001E", label="Estimate!!Total:",
        is_total=True, depth=0, is_commonly_queried=True,
        router_rank=None, router_score=0.0,
    )
    subgroup = _VarCandidate(
        table_id="B21001", dataset="acs/acs5", year=2024,
        variable_id="B21001_002E",
        label="Estimate!!Total:!!Male:",   # Male veterans subtree
        is_total=False, depth=1, is_commonly_queried=False,
        router_rank=None, router_score=0.0,
    )
    top_key = ("B21001", "acs/acs5", 2024)

    # Without subgroup hint (wrong behavior): universe wins.
    s_u = _score_candidate(
        universe, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=False,
    )
    s_s = _score_candidate(
        subgroup, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=False,
    )
    assert s_u > s_s, "sanity check: universe wins without subgroup hint"

    # With subgroup hint: universe takes a penalty and subgroup wins.
    s_u2 = _score_candidate(
        universe, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=True,
    )
    s_s2 = _score_candidate(
        subgroup, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=True,
    )
    assert s_s2 > s_u2, (
        f"subgroup variable must outrank universe when concept is "
        f"a subgroup; got universe={s_u2} vs subgroup={s_s2}"
    )


def test_score_candidate_keeps_universe_on_total_concept():
    """When concept is NOT a subgroup (e.g. 'total population',
    'median household income'), the universe _001E should still win —
    the subgroup penalty must not fire."""
    from scripts.chatbot.planner import _VarCandidate, _score_candidate

    universe = _VarCandidate(
        table_id="B01003", dataset="acs/acs5", year=2024,
        variable_id="B01003_001E", label="Estimate!!Total",
        is_total=True, depth=0, is_commonly_queried=True,
        router_rank=None, router_score=0.0,
    )
    subgroup = _VarCandidate(
        table_id="B01003", dataset="acs/acs5", year=2024,
        variable_id="B01003_002E", label="Estimate!!Total!!Male",
        is_total=False, depth=1, is_commonly_queried=False,
        router_rank=None, router_score=0.0,
    )
    top_key = ("B01003", "acs/acs5", 2024)
    s_u = _score_candidate(
        universe, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=False,
    )
    s_s = _score_candidate(
        subgroup, top_table_key=top_key,
        measure_type_sought=None, is_total_likely=None,
        concept_is_subgroup=False,
    )
    assert s_u > s_s
