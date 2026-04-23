"""Tests for nodes/comparators.py — J.1 comparator chain synthesis."""
from __future__ import annotations

import pytest

from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent,
)
from scripts.chatbot.nodes.comparators import comparator_geos
from scripts.chatbot.planner import plan_query


# ---------------------------------------------------------------------------
# Comparator chain shape
# ---------------------------------------------------------------------------

def test_neighborhood_geo_gets_full_chain(gazetteer_db):
    """Buckhead (neighborhood) → county + MSA + state + US."""
    intent = ExtractedIntent(geo_refs=[
        ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                        scope_hint="Atlanta"),
    ])
    resolved = resolve_intent(intent, gazetteer_db)
    geo = resolved[0]
    pairs = comparator_geos(geo, gazetteer_db, max_comparators=4)
    assert len(pairs) == 4
    levels = [role.split(".", 1)[1] for _, role in pairs]
    assert levels == ["county", "msa", "state", "us"]
    # County row should be Fulton (Buckhead's primary county)
    county_geo = pairs[0][0]
    assert county_geo.geo_level == "county"
    assert county_geo.geo_id == "13121"
    # MSA should be Atlanta-Sandy Springs
    msa_geo = pairs[1][0]
    assert msa_geo.geo_level == "msa"
    assert msa_geo.geo_id == "12060"
    # State row → Georgia
    state_geo = pairs[2][0]
    assert state_geo.geo_level == "state"
    assert state_geo.geo_id == "13"
    # US row is synthesized (geo_id="1")
    us_geo = pairs[3][0]
    assert us_geo.geo_level == "us"
    assert us_geo.geo_id == "1"
    assert us_geo.api_for_clause == "us:1"


def test_county_geo_skips_county_in_chain(gazetteer_db):
    """DeKalb County → MSA + state + US (no self-county)."""
    intent = ExtractedIntent(geo_refs=[
        ExtractedGeoRef(text="DeKalb County", ref_type="administrative",
                        scope_hint="Georgia"),
    ])
    resolved = resolve_intent(intent, gazetteer_db)
    geo = resolved[0]
    pairs = comparator_geos(geo, gazetteer_db, max_comparators=4)
    levels = [role.split(".", 1)[1] for _, role in pairs]
    assert levels == ["msa", "state", "us"]


def test_state_geo_only_gets_us_comparator(gazetteer_db):
    """Georgia → US only."""
    intent = ExtractedIntent(geo_refs=[
        ExtractedGeoRef(text="Georgia", ref_type="administrative"),
    ])
    resolved = resolve_intent(intent, gazetteer_db)
    geo = resolved[0]
    pairs = comparator_geos(geo, gazetteer_db, max_comparators=4)
    levels = [role.split(".", 1)[1] for _, role in pairs]
    assert levels == ["us"]


def test_max_comparators_caps_chain(gazetteer_db):
    intent = ExtractedIntent(geo_refs=[
        ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                        scope_hint="Atlanta"),
    ])
    resolved = resolve_intent(intent, gazetteer_db)
    pairs = comparator_geos(resolved[0], gazetteer_db, max_comparators=2)
    assert len(pairs) == 2
    levels = [role.split(".", 1)[1] for _, role in pairs]
    assert levels == ["county", "msa"]


def test_max_comparators_zero_returns_empty(gazetteer_db):
    intent = ExtractedIntent(geo_refs=[
        ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                        scope_hint="Atlanta"),
    ])
    resolved = resolve_intent(intent, gazetteer_db)
    pairs = comparator_geos(resolved[0], gazetteer_db, max_comparators=0)
    assert pairs == []


def test_us_geo_returns_no_comparators():
    """A US-level primary has nothing to compare against."""
    from scripts.chatbot.models import ResolvedGeography
    us = ResolvedGeography(
        geo_id="1", geo_level="us", geo_type="us",
        display_name="United States",
        api_for_clause="us:1", confidence=1.0,
        data_level_available="us",
        source_ref=ExtractedGeoRef(text="USA", ref_type="national"),
    )
    # No db needed for this case.
    pairs = comparator_geos(us, db=None)        # type: ignore[arg-type]
    assert pairs == []


# ---------------------------------------------------------------------------
# Planner integration via extra_geos
# ---------------------------------------------------------------------------

def test_planner_emits_comparator_calls_when_extra_geos_supplied(
    gazetteer_db, metadata_db, cmap=None,
):
    """Pass comparator pairs through plan_query.extra_geos and verify
    the resulting plan emits one call per (concept × extra geo) with
    role="comparator.<level>"."""
    from scripts.chatbot.concept_map import ConceptMap
    cmap = ConceptMap.from_yaml()

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
    extras = comparator_geos(resolved[0], gazetteer_db, max_comparators=4)
    plan = plan_query(
        intent, resolved, cmap, metadata_db, extra_geos=extras,
    )
    # All resolved_geos should be present (primary + comparators).
    assert len(plan.resolved_geos) == 1 + len(extras)
    # Role distribution: 1 primary + n comparators per concept.
    primary_calls = [c for c in plan.calls if c.role == "primary"]
    comparator_calls = [c for c in plan.calls if c.role.startswith("comparator.")]
    assert primary_calls, "expected at least one primary call"
    assert len(comparator_calls) >= 1
    # Comparator roles should match the chain.
    comp_roles = sorted({c.role for c in comparator_calls})
    assert any(r.endswith("msa") for r in comp_roles)
    assert any(r.endswith("state") for r in comp_roles)
    assert any(r.endswith("us") for r in comp_roles)


def test_planner_extra_geos_default_empty(gazetteer_db, metadata_db):
    """No extra_geos passed → behaves identically to legacy plan_query."""
    from scripts.chatbot.concept_map import ConceptMap
    cmap = ConceptMap.from_yaml()
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
    assert all(c.role == "primary" for c in plan.calls)
    assert len(plan.resolved_geos) == 1
