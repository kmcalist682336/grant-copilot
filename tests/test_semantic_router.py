"""Tests for semantic_router.

Build a tiny synthetic FAISS index + cards table so router logic can be
validated without depending on the full ~5M-card production index.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import pytest

from scripts.chatbot.semantic_router import (
    Embedder, RouterConfig, SemanticRouter,
)


# ---------------------------------------------------------------------------
# Mock embedder — deterministic 4-dim vectors so we can hand-craft hits
# ---------------------------------------------------------------------------

class _DictEmbedder(Embedder):
    """Maps known text → fixed unit vector. Unknown text → zero vector."""

    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = {k: np.array(v, dtype=np.float32) for k, v in mapping.items()}

    def embed_one(self, text: str) -> np.ndarray:
        if text in self._mapping:
            return self._mapping[text]
        return np.zeros(4, dtype=np.float32)


def _unit(*dims: float) -> list[float]:
    """Unit-normalize so L2 distances stay in the [0,2] range."""
    v = np.array(dims, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / (norm if norm > 0 else 1.0)).tolist()


# ---------------------------------------------------------------------------
# Fixture factory: build a small metadata DB + FAISS index on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_router(tmp_path):
    """Build:
      - cards.sqlite with 8 hand-crafted cards
      - embeddings.faiss with matching 4-dim vectors
    Returns (SemanticRouter, embedder, path)."""

    # Synthetic cards: 4 "poverty" cards (table B17001), 3 "income" cards
    # (table B19013), 1 negative "rent" card on B17001 (avoid_when).
    # Each card has a unique vector pointing in its concept's direction.
    cards = [
        # rowid, type, text, table_id, dataset, year, var_id, weight, polarity, vec
        (1,  "table_question_canonical", "How many people are poor in this area?",
         "B17001", "acs/acs5", 2023, None, 1.5, "positive", _unit(1, 0, 0, 0)),
        (2,  "var_question",             "Total population below poverty",
         "B17001", "acs/acs5", 2023, "B17001_002E", 2.0, "positive", _unit(0.95, 0.05, 0, 0)),
        (3,  "var_question",             "Poverty count for area",
         "B17001", "acs/acs5", 2023, "B17001_002E", 2.0, "positive", _unit(0.9, 0.1, 0, 0)),
        (4,  "table_description",        "Poverty Status table description",
         "B17001", "acs/acs5", 2023, None, 1.0, "positive", _unit(0.8, 0.2, 0, 0)),
        (5,  "table_question_canonical", "What is the median household income?",
         "B19013", "acs/acs5", 2023, None, 1.5, "positive", _unit(0, 1, 0, 0)),
        (6,  "var_question",             "Median household income value",
         "B19013", "acs/acs5", 2023, "B19013_001E", 2.0, "positive", _unit(0, 0.95, 0.05, 0)),
        (7,  "table_description",        "Median Household Income description",
         "B19013", "acs/acs5", 2023, None, 1.0, "positive", _unit(0, 0.9, 0.1, 0)),
        # NEGATIVE card on B17001 — penalizes when query embeds near "rent"
        (8,  "table_avoid_when",         "Avoid this table for rent-related questions",
         "B17001", "acs/acs5", 2023, None, 1.0, "negative", _unit(0, 0, 0, 1)),
    ]

    db_path = tmp_path / "metadata.db"
    db = sqlite3.connect(db_path)
    # Post-Phase-2.5 schema: years_available + geo_levels_available JSON
    # arrays sit on the card row itself so the router can filter by
    # coverage without joining a separate table. target_year stays as
    # a legacy column (NULL for new-shape question cards).
    db.execute("""
        CREATE TABLE cards (
            rowid INTEGER PRIMARY KEY,
            card_type TEXT, text TEXT,
            target_table_id TEXT, target_dataset TEXT, target_year INTEGER,
            target_variable_id TEXT,
            weight REAL, polarity TEXT,
            years_available TEXT NOT NULL DEFAULT '[]',
            geo_levels_available TEXT NOT NULL DEFAULT '[]',
            measure_type TEXT, subject_area TEXT, l1_domains TEXT,
            is_commonly_queried INTEGER, is_total INTEGER, depth INTEGER,
            embedding_version TEXT
        )
    """)
    # Coverage table kept for legacy shape-compat (no writers left in
    # production, but some tools still read it). Filter no longer joins
    # against it — per-card JSON is authoritative.
    db.execute("""
        CREATE TABLE coverage (
            table_id TEXT, dataset TEXT, year INTEGER, geo_level TEXT,
            status TEXT, row_count INTEGER, source TEXT, checked_at REAL
        )
    """)
    # Per-card coverage metadata. Matches what the swap path writes in
    # production: a union of coverage across every vintage the card
    # was built from.
    coverage_by_target = {
        "B17001": {"years": [2023], "levels": ["tract", "county"]},
        "B19013": {"years": [2023], "levels": ["county"]},
    }
    for c in cards:
        cov = coverage_by_target.get(c[3], {"years": [], "levels": []})
        db.execute(
            "INSERT INTO cards (rowid, card_type, text, target_table_id, "
            "target_dataset, target_year, target_variable_id, weight, "
            "polarity, years_available, geo_levels_available) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            c[:9] + (
                json.dumps(cov["years"]),
                json.dumps(cov["levels"]),
            ),
        )
    db.commit()
    db.close()

    # Build FAISS index — flat L2, 4-dim, IDMap2 wrapper for explicit IDs
    base = faiss.IndexFlatL2(4)
    idx = faiss.IndexIDMap2(base)
    vectors = np.array([c[9] for c in cards], dtype=np.float32)
    ids = np.array([c[0] for c in cards], dtype=np.int64)
    idx.add_with_ids(vectors, ids)
    index_path = tmp_path / "embeddings.faiss"
    faiss.write_index(idx, str(index_path))

    # Build the embedder mapping common test queries
    embedder = _DictEmbedder({
        "poverty in atlanta":     _unit(1, 0, 0, 0),
        "median household income": _unit(0, 1, 0, 0),
        "rent affordability":      _unit(0, 0, 0, 1),
        "income or poverty":       _unit(0.7, 0.7, 0, 0),
    })

    router = SemanticRouter(
        index_path, db_path, embedder,
        config=RouterConfig(over_fetch_factor=10, min_aggregate_score=0.0),
    )
    return router


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------

def test_poverty_query_picks_b17001(tiny_router):
    """Pure poverty query → B17001 should outrank B19013."""
    res = tiny_router.route("poverty in atlanta", top_k=5)
    assert res.top_tables, "expected at least one table candidate"
    top = res.top_tables[0]
    assert top.target_table_id == "B17001"
    # Best-hit cosine on the matching card should be near 1.0
    assert top.best_hit.cosine > 0.9
    # Multiple cards from B17001 should have hit
    assert top.n_positive_hits >= 3


def test_poverty_query_picks_correct_variable(tiny_router):
    """Variable-level result for poverty should pick B17001_002E."""
    res = tiny_router.route("poverty in atlanta", top_k=5)
    assert res.top_variables
    assert res.top_variables[0].target_variable_id == "B17001_002E"


def test_income_query_picks_b19013(tiny_router):
    res = tiny_router.route("median household income", top_k=5)
    assert res.top_tables[0].target_table_id == "B19013"
    assert res.top_variables[0].target_variable_id == "B19013_001E"


def test_negative_card_demotes_b17001_for_rent_query(tiny_router):
    """Querying near the avoid_when card should demote B17001."""
    res = tiny_router.route("rent affordability", top_k=5)
    # B17001 may still appear (its other cards are roughly orthogonal to
    # the rent vector) but its aggregate must be NEGATIVE because of
    # the strong negative card hit.
    b17001 = next(
        (t for t in res.top_tables if t.target_table_id == "B17001"), None,
    )
    if b17001 is not None:
        # The avoid_when card scored ~1.0 cosine, weighted -1.0
        assert b17001.aggregate_score < 0
        assert b17001.n_negative_hits == 1


def test_convergent_evidence_boosts_table(tiny_router):
    """A query equally close to two concepts (poverty + income) should
    rank both tables, but the one with more matching cards wins."""
    res = tiny_router.route("income or poverty", top_k=5)
    assert len(res.top_tables) >= 2
    # B17001 has 4 positive cards near the poverty axis;
    # B19013 has 3 near the income axis. Both should appear.
    table_ids = [t.target_table_id for t in res.top_tables]
    assert "B17001" in table_ids
    assert "B19013" in table_ids


def test_coverage_filter_drops_uncovered_table(tiny_router):
    """Requiring tract coverage should drop B19013 (county-only)."""
    res = tiny_router.route(
        "median household income", top_k=5,
        require_coverage_at=["tract"],
    )
    table_ids = [t.target_table_id for t in res.top_tables]
    assert "B19013" not in table_ids
    # B17001 has tract coverage, so it should survive any incidental hit


def test_empty_query_returns_empty(tiny_router):
    res = tiny_router.route("", top_k=5)
    assert res.top_tables == []
    assert res.top_variables == []


def test_unknown_query_returns_empty_or_low_scores(tiny_router):
    """A query the embedder doesn't know returns the zero vector → all
    cards equidistant → uniform scores. With min_aggregate_score floor
    we still expect SOMETHING but it shouldn't crash."""
    res = tiny_router.route("totally unrelated jibberish", top_k=5)
    # Just sanity — no crash, plausible structure
    assert isinstance(res.top_tables, list)


def test_top_k_caps_results(tiny_router):
    res = tiny_router.route("poverty in atlanta", top_k=1)
    assert len(res.top_tables) <= 1
    assert len(res.top_variables) <= 1


def test_route_includes_evidence(tiny_router):
    res = tiny_router.route("poverty in atlanta", top_k=5)
    assert res.raw_hits
    assert res.n_hits_retrieved >= 1
    # First raw hit should match the highest-cosine card (rowid=1 or 2)
    assert res.raw_hits[0].cosine > 0.9


# ---------------------------------------------------------------------------
# Table-family bias (Phase 2a)
# ---------------------------------------------------------------------------

def test_table_family_weight_lookup():
    """Verify prefix-longest-match + PR/race suffix composition."""
    from scripts.chatbot.semantic_router import (
        _table_family_weight, RouterConfig,
    )
    cfg = RouterConfig()
    w = cfg.table_family_weights
    assert _table_family_weight("B17001", w) == 1.00
    assert _table_family_weight("C17001", w) == 0.90
    assert _table_family_weight("S1701", w) == 0.60
    assert _table_family_weight("DP03", w) == 0.20
    assert _table_family_weight("CP05", w) == 0.20
    assert _table_family_weight("K200101", w) == 0.40
    # PR suffix multiplies — even B-series PR gets zeroed.
    assert _table_family_weight("B17001PR", w) == 0.0
    assert _table_family_weight("S1701PR", w) == 0.0
    # Race suffix multiplies — single trailing A-I after digits.
    assert _table_family_weight("B17001A", w) == pytest.approx(0.30)
    assert _table_family_weight("B19013H", w) == pytest.approx(0.30)
    assert _table_family_weight("C19013A", w) == pytest.approx(0.27)
    assert _table_family_weight("S0701B", w) == pytest.approx(0.18)
    # Race-suffix regex matches only B/C/S prefixes, so DP03A is not
    # treated as a race variant — it just gets the DP bias of 0.20.
    assert _table_family_weight("DP03A", w) == pytest.approx(0.20)
    # Variable IDs (table_id with _NNNE suffix) NOT race-suffix candidates.
    assert _table_family_weight("B19013_001E", w) == 1.0
    # Unknown prefix → 1.0 (no change).
    assert _table_family_weight("XYZ", w) == 1.0


def test_bias_demotes_profile_table_when_scores_would_tie(tmp_path):
    """Build a router whose raw scores tie between a B-series and an
    S-series table. With bias enabled, B should win."""
    import sqlite3
    import time
    import faiss
    import numpy as np
    from scripts.chatbot.semantic_router import (
        Embedder, RouterConfig, SemanticRouter,
    )

    # Two cards hit the same query equally; distinguished only by
    # their table prefix.
    class _OneEmb(Embedder):
        def embed_one(self, text):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    cards = [
        (1, "table_question_canonical", "poverty", "B17001", "acs/acs5",
         2023, None, 1.0, "positive",
         [1.0, 0.0, 0.0, 0.0]),
        (2, "table_question_canonical", "poverty", "S1701", "acs/acs5",
         2023, None, 1.0, "positive",
         [1.0, 0.0, 0.0, 0.0]),
    ]
    db_path = tmp_path / "m.db"
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE cards (
            rowid INTEGER PRIMARY KEY,
            card_type TEXT, text TEXT,
            target_table_id TEXT, target_dataset TEXT, target_year INTEGER,
            target_variable_id TEXT,
            weight REAL, polarity TEXT,
            years_available TEXT NOT NULL DEFAULT '[]',
            geo_levels_available TEXT NOT NULL DEFAULT '[]',
            measure_type TEXT, subject_area TEXT, l1_domains TEXT,
            is_commonly_queried INTEGER, is_total INTEGER, depth INTEGER,
            embedding_version TEXT
        )
    """)
    db.execute("""
        CREATE TABLE coverage (
            table_id TEXT, dataset TEXT, year INTEGER, geo_level TEXT,
            status TEXT, row_count INTEGER, source TEXT, checked_at REAL
        )
    """)
    for c in cards:
        db.execute(
            "INSERT INTO cards (rowid,card_type,text,target_table_id,"
            "target_dataset,target_year,target_variable_id,weight,polarity,"
            "years_available,geo_levels_available) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            c[:9] + (json.dumps([2023]), json.dumps(["tract"])),
        )
    db.commit()
    db.close()

    base = faiss.IndexFlatL2(4)
    idx = faiss.IndexIDMap2(base)
    vecs = np.array([c[9] for c in cards], dtype=np.float32)
    ids = np.array([c[0] for c in cards], dtype=np.int64)
    idx.add_with_ids(vecs, ids)
    idx_path = tmp_path / "idx.faiss"
    faiss.write_index(idx, str(idx_path))

    # Bias ON — B17001 should outrank S1701.
    r_on = SemanticRouter(
        idx_path, db_path, _OneEmb(),
        config=RouterConfig(
            over_fetch_factor=10, min_aggregate_score=-10.0,
        ),
    )
    res_on = r_on.route("poverty", top_k=5)
    assert res_on.top_tables[0].target_table_id == "B17001"

    # Bias OFF — ties break on insertion order (effectively random);
    # just assert both are present and have equal scores.
    r_off = SemanticRouter(
        idx_path, db_path, _OneEmb(),
        config=RouterConfig(
            over_fetch_factor=10, min_aggregate_score=-10.0,
            table_family_bias=False,
        ),
    )
    res_off = r_off.route("poverty", top_k=5)
    assert {t.target_table_id for t in res_off.top_tables} == {"B17001", "S1701"}
    assert res_off.top_tables[0].aggregate_score == \
           res_off.top_tables[1].aggregate_score


def test_bias_excludes_puerto_rico_variants(tmp_path):
    """PR-suffix tables should score ~0 under default bias."""
    import sqlite3, time, faiss
    import numpy as np
    from scripts.chatbot.semantic_router import (
        Embedder, RouterConfig, SemanticRouter,
    )

    class _OneEmb(Embedder):
        def embed_one(self, text):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Two identical cards on a real table and a PR variant.
    cards = [
        (1, "table_question_canonical", "poverty", "S1701", "acs/acs5",
         2023, None, 1.0, "positive", [1.0, 0.0, 0.0, 0.0]),
        (2, "table_question_canonical", "poverty", "S1701PR", "acs/acs5",
         2023, None, 1.0, "positive", [1.0, 0.0, 0.0, 0.0]),
    ]
    db_path = tmp_path / "m.db"
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE cards (
            rowid INTEGER PRIMARY KEY, card_type TEXT, text TEXT,
            target_table_id TEXT, target_dataset TEXT, target_year INTEGER,
            target_variable_id TEXT, weight REAL, polarity TEXT,
            years_available TEXT NOT NULL DEFAULT '[]',
            geo_levels_available TEXT NOT NULL DEFAULT '[]',
            measure_type TEXT, subject_area TEXT, l1_domains TEXT,
            is_commonly_queried INTEGER, is_total INTEGER, depth INTEGER,
            embedding_version TEXT
        )
    """)
    db.execute("""
        CREATE TABLE coverage (
            table_id TEXT, dataset TEXT, year INTEGER, geo_level TEXT,
            status TEXT, row_count INTEGER, source TEXT, checked_at REAL
        )
    """)
    for c in cards:
        db.execute(
            "INSERT INTO cards (rowid,card_type,text,target_table_id,"
            "target_dataset,target_year,target_variable_id,weight,polarity,"
            "years_available,geo_levels_available) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            c[:9] + (json.dumps([2023]), json.dumps(["tract"])),
        )
    db.commit()
    db.close()

    base = faiss.IndexFlatL2(4)
    idx = faiss.IndexIDMap2(base)
    vecs = np.array([c[9] for c in cards], dtype=np.float32)
    ids = np.array([c[0] for c in cards], dtype=np.int64)
    idx.add_with_ids(vecs, ids)
    idx_path = tmp_path / "idx.faiss"
    faiss.write_index(idx, str(idx_path))

    r = SemanticRouter(
        idx_path, db_path, _OneEmb(),
        config=RouterConfig(
            over_fetch_factor=10, min_aggregate_score=-10.0,
        ),
    )
    res = r.route("poverty", top_k=5)
    pr = [t for t in res.top_tables if t.target_table_id.endswith("PR")]
    for p in pr:
        # weight 0 → aggregate_score multiplied to 0 (or very close)
        assert p.aggregate_score == 0.0
    non_pr = [t for t in res.top_tables if not t.target_table_id.endswith("PR")]
    assert non_pr, "non-PR variant should still appear"
    assert non_pr[0].aggregate_score > 0
