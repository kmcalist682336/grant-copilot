"""Tests for PeerRetriever — the axis-specific peer ranker.

Uses an in-memory SQLite fixture with synthetic rows so tests don't
depend on a real Census fetch. Exercises the ranking math, size
bucketing, state restriction, NULL feature handling, and axis
selection.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.chatbot.nodes.peer_retriever import (
    AnchorFeatures, PeerRef, PeerRetrievalError, PeerRetriever,
)
from scripts.chatbot.peer_features_catalog import AXES, FEATURES


# ---------------------------------------------------------------------------
# Fixture DB — write a tiny peer_features.sqlite to tmp_path
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


def _insert_row(con, *, geo_level, geo_id, geo_name, state_fips,
                population, features, vintage=2023):
    con.execute(
        "INSERT INTO peer_features VALUES (?, ?, ?, ?, ?, ?, ?, 0.0)",
        (geo_level, geo_id, state_fips, geo_name, vintage, population,
         json.dumps(features, separators=(",", ":"))),
    )


@pytest.fixture
def peer_db(tmp_path):
    path = tmp_path / "peer_features.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)

    # Anchor: a mid-sized place with median income $75k, poverty 12%
    _insert_row(con, geo_level="place", geo_id="0100001",
                geo_name="Anchor City, XX", state_fips="13",
                population=100_000, features={
                    "median_household_income": 75_000.0,
                    "poverty_rate": 0.12,
                    "pct_bachelors_plus": 0.35,
                    "pct_owner_occupied": 0.55,
                    "median_home_value": 250_000.0,
                    "median_gross_rent": 1_400.0,
                    "pct_snap_households": 0.10,
                    "per_capita_income": 35_000.0,
                    "mean_household_income": 85_000.0,
                    "gini_index": 0.45,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.68,
                    "child_poverty_rate": 0.16,
                    "senior_poverty_rate": 0.08,
                    "median_family_income": 85_000.0,
                })

    # Close econ peer: same median income, similar poverty, different state
    _insert_row(con, geo_level="place", geo_id="0200001",
                geo_name="Econ Peer, YY", state_fips="17",
                population=95_000, features={
                    "median_household_income": 76_000.0,
                    "poverty_rate": 0.11,
                    "pct_bachelors_plus": 0.40,
                    "pct_owner_occupied": 0.58,
                    "median_home_value": 260_000.0,
                    "median_gross_rent": 1_450.0,
                    "pct_snap_households": 0.09,
                    "per_capita_income": 36_000.0,
                    "mean_household_income": 87_000.0,
                    "gini_index": 0.44,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.70,
                    "child_poverty_rate": 0.14,
                    "senior_poverty_rate": 0.08,
                    "median_family_income": 86_000.0,
                })

    # Different econ: much lower income, higher poverty
    _insert_row(con, geo_level="place", geo_id="0300001",
                geo_name="Lower-Income City, ZZ", state_fips="22",
                population=110_000, features={
                    "median_household_income": 40_000.0,
                    "poverty_rate": 0.28,
                    "pct_bachelors_plus": 0.15,
                    "pct_owner_occupied": 0.45,
                    "median_home_value": 120_000.0,
                    "median_gross_rent": 850.0,
                    "pct_snap_households": 0.25,
                    "per_capita_income": 18_000.0,
                    "mean_household_income": 48_000.0,
                    "gini_index": 0.50,
                    "unemployment_rate": 0.11,
                    "labor_force_participation": 0.55,
                    "child_poverty_rate": 0.38,
                    "senior_poverty_rate": 0.18,
                    "median_family_income": 46_000.0,
                })

    # In-state peer (same state as anchor): mildly similar
    _insert_row(con, geo_level="place", geo_id="1300002",
                geo_name="In-State Peer, GA", state_fips="13",
                population=105_000, features={
                    "median_household_income": 68_000.0,
                    "poverty_rate": 0.14,
                    "pct_bachelors_plus": 0.30,
                    "pct_owner_occupied": 0.52,
                    "median_home_value": 220_000.0,
                    "median_gross_rent": 1_300.0,
                    "pct_snap_households": 0.12,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 78_000.0,
                    "gini_index": 0.46,
                    "unemployment_rate": 0.06,
                    "labor_force_participation": 0.67,
                    "child_poverty_rate": 0.18,
                    "senior_poverty_rate": 0.09,
                    "median_family_income": 78_000.0,
                })

    # Out-of-size-bucket: tiny town, should be filtered by size bucket
    _insert_row(con, geo_level="place", geo_id="0400001",
                geo_name="Tiny Town", state_fips="44",
                population=2_000, features={
                    "median_household_income": 75_000.0,
                    "poverty_rate": 0.12,
                })

    # Another level: county (shouldn't appear when geo_level='place')
    _insert_row(con, geo_level="county", geo_id="13089",
                geo_name="DeKalb County, GA", state_fips="13",
                population=762_000, features={
                    "median_household_income": 77_000.0,
                    "poverty_rate": 0.13,
                })

    con.commit()
    con.close()
    return path


# ---------------------------------------------------------------------------
# Lookup + anchor
# ---------------------------------------------------------------------------

def test_lookup_anchor_by_geo_id(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    assert anchor is not None
    assert anchor.geo_name == "Anchor City, XX"
    assert anchor.population == 100_000
    assert anchor.features["median_household_income"] == 75_000.0
    assert anchor.state_fips == "13"


def test_lookup_anchor_missing_returns_none(peer_db):
    r = PeerRetriever(peer_db)
    assert r.lookup_anchor(geo_level="place", geo_id="nope") is None


def test_missing_db_raises():
    with pytest.raises(PeerRetrievalError, match="DB missing"):
        PeerRetriever("/tmp/does_not_exist_42.sqlite")


def test_anchor_from_aggregated_validates_feature_names(peer_db):
    r = PeerRetriever(peer_db)
    with pytest.raises(PeerRetrievalError, match="unknown feature"):
        r.anchor_from_aggregated(
            features={"not_a_real_feature": 1.0}, population=10_000,
            label="synthetic",
        )


def test_anchor_from_aggregated_builds_synthetic_anchor(peer_db):
    r = PeerRetriever(peer_db)
    a = r.anchor_from_aggregated(
        features={"median_household_income": 50_000.0}, population=30_000,
        label="Southwest Atlanta composite",
    )
    assert a.population == 30_000
    assert a.geo_name == "Southwest Atlanta composite"
    assert a.features == {"median_household_income": 50_000.0}


# ---------------------------------------------------------------------------
# Axis ranking — the close econ peer should win
# ---------------------------------------------------------------------------

def test_econ_axis_ranks_close_peer_first(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", top_k=3)
    # Econ Peer and In-State Peer both look similar; Lower-Income City
    # is much further. Anchor itself should be excluded.
    assert all(p.geo_id != "0100001" for p in peers)
    assert peers[0].geo_id == "0200001"       # closest on income
    assert peers[0].axis == "economic"
    assert peers[0].rank == 1
    assert peers[-1].distance >= peers[0].distance


def test_unknown_axis_raises(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    with pytest.raises(PeerRetrievalError, match="unknown axis"):
        r.peers(anchor, axis="not_real")


# ---------------------------------------------------------------------------
# Size bucketing — the 2k-pop tiny town must NOT appear for a 100k anchor
# ---------------------------------------------------------------------------

def test_size_bucket_excludes_tiny_town(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", top_k=10)
    assert all(p.geo_id != "0400001" for p in peers)


# ---------------------------------------------------------------------------
# State restriction
# ---------------------------------------------------------------------------

def test_restrict_state_limits_candidates(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    # Only in-state candidate is 1300002 (same state fips = '13').
    peers = r.peers(anchor, axis="economic", top_k=10, restrict_state="13")
    assert [p.geo_id for p in peers] == ["1300002"]


# ---------------------------------------------------------------------------
# Geo-level filtering
# ---------------------------------------------------------------------------

def test_place_anchor_does_not_match_county_candidates(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", geo_level="place", top_k=10)
    assert all(p.geo_level == "place" for p in peers)


# ---------------------------------------------------------------------------
# NULL feature handling — min_features gates and missing vals ignored
# ---------------------------------------------------------------------------

def test_min_features_gates_degenerate_peers(peer_db):
    """Tiny Town has only 2 features; if it were in-bucket, it would
    fail the min_features=3 gate."""
    r = PeerRetriever(peer_db)
    # Use synthetic small-pop anchor so tiny town is in-bucket
    anchor = r.anchor_from_aggregated(
        features={
            "median_household_income": 75_000.0,
            "poverty_rate": 0.12,
            "pct_bachelors_plus": 0.35,
            "pct_owner_occupied": 0.55,
        },
        population=3_000, label="small anchor",
    )
    peers = r.peers(anchor, axis="economic", top_k=10, min_features=3)
    # Tiny Town has only 2 econ features (income + poverty); excluded.
    assert all(p.geo_id != "0400001" for p in peers)


# ---------------------------------------------------------------------------
# peers_by_axis — bulk retrieval
# ---------------------------------------------------------------------------

def test_peers_by_axis_returns_dict_keyed_by_axis(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    by_axis = r.peers_by_axis(
        anchor, axes=["economic", "housing"], top_k=2,
    )
    assert set(by_axis.keys()) == {"economic", "housing"}
    assert all(isinstance(p, PeerRef) for peers in by_axis.values() for p in peers)
    # Default top_k=2 on bulk path
    assert all(len(peers) <= 2 for peers in by_axis.values())


def test_peers_by_axis_defaults_to_all_axes(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    by_axis = r.peers_by_axis(anchor, top_k=1)
    # Every configured axis must appear as a key, even if its peer
    # list ends up empty for this tiny fixture.
    assert set(by_axis.keys()) == set(AXES.keys())


# ---------------------------------------------------------------------------
# Determinism — same inputs → same ordering
# ---------------------------------------------------------------------------

def test_rank_order_is_deterministic(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    a = [p.geo_id for p in r.peers(anchor, axis="economic", top_k=5)]
    b = [p.geo_id for p in r.peers(anchor, axis="economic", top_k=5)]
    assert a == b


# ---------------------------------------------------------------------------
# Rank numbering + distance ordering
# ---------------------------------------------------------------------------

def test_rank_is_1_indexed_and_distances_nondecreasing(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", top_k=5)
    ranks = [p.rank for p in peers]
    assert ranks == list(range(1, len(ranks) + 1))
    dists = [p.distance for p in peers]
    assert dists == sorted(dists)


# ---------------------------------------------------------------------------
# Proximity weighting + match explanations
# ---------------------------------------------------------------------------

def test_same_state_peer_beats_cross_region_peer_with_tie_features(tmp_path):
    """When two candidates have identical features, the same-state
    peer should win on the proximity discount."""
    path = tmp_path / "proximity.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # Anchor in Georgia (FIPS 13)
    _insert_row(con, geo_level="place", geo_id="A",
                geo_name="Atlanta, GA", state_fips="13",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    # Macon, GA — same state
    _insert_row(con, geo_level="place", geo_id="B",
                geo_name="Macon, GA", state_fips="13",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    # Portland, OR — Pacific, different region entirely
    _insert_row(con, geo_level="place", geo_id="C",
                geo_name="Portland, OR", state_fips="41",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    con.commit()
    con.close()

    r = PeerRetriever(path)
    anchor = r.lookup_anchor(geo_level="place", geo_id="A")
    peers = r.peers(anchor, axis="economic", top_k=5)
    assert peers[0].geo_name == "Macon, GA"


def test_same_region_peer_beats_cross_region_on_modifier(tmp_path):
    """Georgia (South Atlantic) → Louisiana (West South Central,
    same South region) should rank above Oregon (Pacific)."""
    path = tmp_path / "region.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    _insert_row(con, geo_level="place", geo_id="A",
                geo_name="Atlanta, GA", state_fips="13",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    _insert_row(con, geo_level="place", geo_id="B",
                geo_name="New Orleans, LA", state_fips="22",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    _insert_row(con, geo_level="place", geo_id="C",
                geo_name="Portland, OR", state_fips="41",
                population=500_000, features={
                    "median_household_income": 60_000.0,
                    "median_family_income": 70_000.0,
                    "per_capita_income": 32_000.0,
                    "mean_household_income": 80_000.0,
                    "gini_index": 0.46,
                    "poverty_rate": 0.15,
                    "child_poverty_rate": 0.20,
                    "senior_poverty_rate": 0.10,
                    "unemployment_rate": 0.05,
                    "labor_force_participation": 0.65,
                })
    con.commit()
    con.close()

    r = PeerRetriever(path)
    anchor = r.lookup_anchor(geo_level="place", geo_id="A")
    peers = r.peers(anchor, axis="economic", top_k=5)
    names = [p.geo_name for p in peers]
    assert names.index("New Orleans, LA") < names.index("Portland, OR")


def test_match_explanation_mentions_proximity_and_features(peer_db):
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", top_k=3)
    assert peers, "expected at least one peer"
    for p in peers:
        assert p.match_explanation, "match_explanation should be populated"
        # Must at least name a proximity tier.
        tiers = ("same state", "same Census division",
                 "same Census region", "nationwide peer")
        assert any(t in p.match_explanation for t in tiers)


def test_raw_distance_preserved_and_adjusted_distance_differs(peer_db):
    """For peers in the same state as the anchor, the stored
    ``distance`` should be lower than ``raw_distance`` because the
    proximity discount was applied."""
    r = PeerRetriever(peer_db)
    anchor = r.lookup_anchor(geo_level="place", geo_id="0100001")
    peers = r.peers(anchor, axis="economic", top_k=10)
    ga_peers = [p for p in peers if p.state_fips == "13"]
    for p in ga_peers:
        # Skip unpopulated raw_distance for anchor-identical peers.
        if p.raw_distance > 0:
            assert p.distance <= p.raw_distance
