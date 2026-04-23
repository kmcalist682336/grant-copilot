"""Tests for nodes/trend.py — J.6 prior-period expansion."""
from __future__ import annotations

import pytest

from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent,
)
from scripts.chatbot.nodes.concept_rewriter import ConceptRewrite
from scripts.chatbot.nodes.trend import prior_period_calls
from scripts.chatbot.planner import plan_query


def _rewrite(text: str, *, trend_eligible: bool = True) -> ConceptRewrite:
    return ConceptRewrite(
        original_text=text, embedding_text=text,
        measure_type_sought="percentage",
        units_expected="percent",
        is_total_likely=False,
        disparity_eligible=False,
        trend_eligible=trend_eligible,
        is_composite=False,
    )


@pytest.fixture
def cmap():
    return ConceptMap.from_yaml()


def test_prior_period_emits_for_trend_eligible_concept(
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
        temporal_intent="latest",
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    rewrites = [_rewrite("poverty", trend_eligible=True)]
    extras = prior_period_calls(plan, rewrites, intent, metadata_db)
    assert extras, "expected prior_period calls"
    for c in extras:
        assert c.role == "prior_period"
        assert c.year < plan.calls[0].year
        # URL uses the prior year
        assert f"/data/{c.year}/" in c.api_call.url


def test_prior_period_skips_non_trend_eligible(
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
        temporal_intent="latest",
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    rewrites = [_rewrite("poverty", trend_eligible=False)]
    extras = prior_period_calls(plan, rewrites, intent, metadata_db)
    assert extras == []


def test_prior_period_skips_when_intent_already_trend(
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
        temporal_intent="trend",
        years=[2020],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap, metadata_db)
    rewrites = [_rewrite("poverty", trend_eligible=True)]
    extras = prior_period_calls(plan, rewrites, intent, metadata_db)
    assert extras == []


def test_prior_period_only_for_primary_role(
    gazetteer_db, metadata_db, cmap,
):
    """Comparator/sub calls should not get their own prior_period —
    we only expand from primaries."""
    from scripts.chatbot.nodes.comparators import comparator_geos
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(
            text="DeKalb County", ref_type="administrative",
            scope_hint="Georgia",
        )],
        concepts=[ExtractedConcept(
            text="poverty", canonical_hint="poverty rate",
        )],
        temporal_intent="latest",
    )
    resolved = resolve_intent(intent, gazetteer_db)
    extras_geo = comparator_geos(resolved[0], gazetteer_db)
    plan = plan_query(
        intent, resolved, cmap, metadata_db, extra_geos=extras_geo,
    )
    rewrites = [_rewrite("poverty", trend_eligible=True)]
    trend_extras = prior_period_calls(plan, rewrites, intent, metadata_db)
    # Should equal exactly the count of primary calls (1 per geo×concept,
    # minus duplicates by table+year).
    primary_calls = [c for c in plan.calls if c.role == "primary"]
    assert len(trend_extras) == len(primary_calls)
    assert all(c.role == "prior_period" for c in trend_extras)
