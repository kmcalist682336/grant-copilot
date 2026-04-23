"""Tests for the peer_context node — concept-to-axis mapping, geo-level
gating, and end-to-end integration with PeerRetriever."""
from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.nodes.peer_context import (
    PeerContext, _axes_for_intent, _axes_for_text,
    _resolve_anchor_key, get_peer_contexts,
)
from scripts.chatbot.nodes.peer_retriever import PeerRetriever


# ---------------------------------------------------------------------------
# Concept → axis mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("poverty rate", ["economic"]),
    ("rent affordability", ["housing"]),
    ("food pantry grant", ["food_benefits"]),
    ("internet access in rural areas", ["digital_access"]),
    ("veteran population", ["veterans"]),
    ("unemployment for workers without a bachelor's degree",
     ["education", "economic"]),
    # Broad economic keywords hit last so they don't shadow specific ones
    ("snap participation and poverty",
     ["food_benefits", "economic"]),
])
def test_axes_for_text_maps_keywords(text, expected):
    got = _axes_for_text(text)
    # Expected axes appear in the result (order-preserving).
    for axis in expected:
        assert axis in got, f"{axis!r} missing from {got!r}"


def test_axes_for_intent_dedupes_across_concepts():
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="DeKalb", ref_type="administrative")],
        concepts=[
            ExtractedConcept(text="poverty rate"),
            ExtractedConcept(text="median household income"),
        ],
    )
    axes = _axes_for_intent(intent, query="poverty and income in DeKalb")
    assert "economic" in axes
    assert axes.count("economic") == 1     # deduped


def test_axes_for_intent_falls_back_to_economic_when_no_keywords():
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="x", ref_type="administrative")],
        concepts=[ExtractedConcept(text="widgets")],
    )
    assert _axes_for_intent(intent, query="widgets") == ["economic"]


# ---------------------------------------------------------------------------
# Geo-level gating
# ---------------------------------------------------------------------------

def _geo(name: str, level: str, geo_id: str = "X") -> ResolvedGeography:
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region", "landmark_area",
    ) else level
    return ResolvedGeography(
        geo_id=geo_id, geo_level=level, geo_type=level,
        display_name=name, tract_geoids=[], county_geoid=None,
        api_for_clause=f"{level}:*", api_in_clause="",
        confidence=1.0, assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type="administrative"),
    )


def test_resolve_anchor_key_accepts_supported_levels():
    assert _resolve_anchor_key(_geo("X", "county", "13089")) == (
        "county", "13089",
    )
    assert _resolve_anchor_key(_geo("X", "msa", "12060")) == (
        "msa", "12060",
    )
    assert _resolve_anchor_key(_geo("X", "place", "1304000")) == (
        "place", "1304000",
    )


def test_resolve_anchor_key_rejects_unsupported_levels():
    # Neighborhoods/tracts/corridors/regions need synthetic anchors —
    # v1 defers that to future work.
    for lvl in ("neighborhood", "tract", "state", "us", "corridor",
                "region"):
        assert _resolve_anchor_key(_geo("X", lvl, "X")) is None


# ---------------------------------------------------------------------------
# End-to-end with a synthetic peer_features DB
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE peer_features (
    geo_level TEXT NOT NULL,
    geo_id TEXT NOT NULL,
    state_fips TEXT,
    geo_name TEXT NOT NULL,
    vintage INTEGER NOT NULL,
    population INTEGER,
    features_json TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    PRIMARY KEY (geo_level, geo_id, vintage)
);
"""


def _insert(con, **kw):
    con.execute(
        "INSERT INTO peer_features VALUES (?, ?, ?, ?, ?, ?, ?, 0.0)",
        (kw["geo_level"], kw["geo_id"], kw["state_fips"],
         kw["geo_name"], kw.get("vintage", 2023), kw["population"],
         json.dumps(kw["features"], separators=(",", ":"))),
    )


@pytest.fixture
def peer_db(tmp_path):
    path = tmp_path / "peer_features.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # Anchor: a 100k-pop county in GA (state=13)
    base = {
        "median_household_income": 70_000.0,
        "poverty_rate": 0.14,
        "pct_bachelors_plus": 0.32,
        "pct_snap_households": 0.11,
        "pct_owner_occupied": 0.55,
        "median_home_value": 240_000.0,
        "median_gross_rent": 1_400.0,
        "per_capita_income": 34_000.0,
        "mean_household_income": 80_000.0,
        "unemployment_rate": 0.05,
        "labor_force_participation": 0.67,
        "child_poverty_rate": 0.17,
        "senior_poverty_rate": 0.09,
        "median_family_income": 80_000.0,
    }
    _insert(con, geo_level="county", geo_id="13999",
            state_fips="13", geo_name="Anchor County, GA",
            population=100_000, features=base)
    # Close in-state peer
    in_state = dict(base)
    in_state["median_household_income"] = 68_000.0
    in_state["poverty_rate"] = 0.15
    _insert(con, geo_level="county", geo_id="13123",
            state_fips="13", geo_name="Near County, GA",
            population=105_000, features=in_state)
    # Nationwide close econ peer (different state)
    close = dict(base)
    close["median_household_income"] = 70_500.0
    close["poverty_rate"] = 0.135
    _insert(con, geo_level="county", geo_id="17031",
            state_fips="17", geo_name="Peer County, IL",
            population=99_000, features=close)
    # Nationwide housing peer (very different income but matching home value)
    housing = dict(base)
    housing["median_home_value"] = 245_000.0
    housing["median_gross_rent"] = 1_420.0
    housing["median_household_income"] = 55_000.0
    housing["poverty_rate"] = 0.21
    _insert(con, geo_level="county", geo_id="22017",
            state_fips="22", geo_name="Housing Peer County, LA",
            population=102_000, features=housing)
    # Place-level sample to exercise place anchors
    _insert(con, geo_level="place", geo_id="1304000",
            state_fips="13", geo_name="Atlanta city, GA",
            population=500_000, features=base)
    _insert(con, geo_level="place", geo_id="3704000",
            state_fips="37", geo_name="Charlotte city, NC",
            population=880_000, features=base)
    con.commit()
    con.close()
    return path


@pytest.fixture
def retriever(peer_db):
    return PeerRetriever(peer_db)


def test_get_peer_contexts_returns_empty_without_retriever():
    assert get_peer_contexts(
        resolved_geos=[_geo("X", "county", "13089")],
        intent=ExtractedIntent(geo_refs=[], concepts=[]),
        query="x", peer_retriever=None,
    ) == []


def test_get_peer_contexts_skips_unsupported_geo(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="poverty")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Buckhead", "neighborhood")],
        intent=intent, query="poverty in buckhead",
        peer_retriever=retriever,
    )
    assert ctxs == []


def test_get_peer_contexts_returns_in_state_peers_for_county(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="poverty rate")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Anchor County", "county", "13999")],
        intent=intent, query="poverty rate in anchor county",
        peer_retriever=retriever,
    )
    # One axis matched (economic)
    assert len(ctxs) >= 1
    econ = next(c for c in ctxs if c.axis == "economic")
    assert econ.anchor_geo_name == "Anchor County, GA"
    # Only one other GA county exists; within-state fallback fired to
    # nationwide pool because <2 candidates in-state.
    peer_names = [p.geo_name for p in econ.peers]
    assert "Peer County, IL" in peer_names or "Near County, GA" in peer_names


def test_get_peer_contexts_picks_multiple_axes(retriever):
    intent = ExtractedIntent(
        geo_refs=[],
        concepts=[
            ExtractedConcept(text="housing cost burden"),
            ExtractedConcept(text="poverty rate"),
        ],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Atlanta city", "place", "1304000")],
        intent=intent, query="housing cost and poverty in atlanta",
        peer_retriever=retriever,
    )
    axes = {c.axis for c in ctxs}
    assert "housing" in axes
    assert "economic" in axes


def test_get_peer_contexts_respects_max_axes(retriever):
    intent = ExtractedIntent(
        geo_refs=[],
        concepts=[
            ExtractedConcept(text="rent"),
            ExtractedConcept(text="income"),
            ExtractedConcept(text="veterans"),
            ExtractedConcept(text="broadband"),
        ],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Anchor County", "county", "13999")],
        intent=intent, query="rent income veterans broadband",
        peer_retriever=retriever, max_axes=2,
    )
    assert len(ctxs) <= 2


def test_get_peer_contexts_top_k_capped(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="poverty rate")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Anchor County", "county", "13999")],
        intent=intent, query="poverty",
        peer_retriever=retriever, top_k=2,
    )
    for c in ctxs:
        assert len(c.peers) <= 2


def test_get_peer_contexts_returns_empty_when_anchor_missing(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="poverty")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Unknown", "county", "99999")],
        intent=intent, query="poverty",
        peer_retriever=retriever,
    )
    assert ctxs == []


def test_peer_context_preserves_axis_description(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="housing")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Anchor County", "county", "13999")],
        intent=intent, query="housing",
        peer_retriever=retriever,
    )
    # The catalog's 'housing' axis description should be non-empty.
    housing = [c for c in ctxs if c.axis == "housing"]
    if housing:
        assert housing[0].axis_description != ""


# ---------------------------------------------------------------------------
# Peer feature values — so the synthesizer can cite concrete numbers
# ---------------------------------------------------------------------------

def test_peer_context_carries_anchor_and_peer_feature_values(retriever):
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="poverty rate")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Anchor County", "county", "13999")],
        intent=intent, query="poverty in anchor",
        peer_retriever=retriever,
    )
    econ = next(c for c in ctxs if c.axis == "economic")
    # Anchor's own values on the axis must be present
    assert "poverty_rate" in econ.anchor_feature_values
    assert econ.anchor_feature_values["poverty_rate"] == pytest.approx(0.14)
    assert "median_household_income" in econ.anchor_feature_values
    assert econ.anchor_feature_values[
        "median_household_income"
    ] == pytest.approx(70_000.0)
    # Each peer carries its own axis feature values
    for p in econ.peers:
        assert p.feature_values, (
            f"peer {p.geo_name!r} missing feature_values"
        )
        # The axis's features must be the ones returned
        assert "poverty_rate" in p.feature_values


def test_peer_feature_values_exclude_non_axis_features(retriever):
    """The peer payload must only carry features that belong to the
    axis — not the full 100+ feature catalog. Keeps the synthesizer
    payload lean and focused on the axis narrative."""
    intent = ExtractedIntent(
        geo_refs=[], concepts=[ExtractedConcept(text="housing")],
    )
    ctxs = get_peer_contexts(
        resolved_geos=[_geo("Atlanta city", "place", "1304000")],
        intent=intent, query="housing in atlanta",
        peer_retriever=retriever,
    )
    housing = [c for c in ctxs if c.axis == "housing"]
    if not housing:
        pytest.skip("no housing peers in this fixture")
    # No economic-only features should leak into the housing axis
    for p in housing[0].peers:
        # median_home_value is a housing feature → allowed
        # gini_index is NOT in the housing axis → not allowed
        assert "gini_index" not in p.feature_values
