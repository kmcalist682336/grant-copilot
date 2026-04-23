"""Tests for the universe-aware variable picker.

Uses a tiny synthetic metadata DB + in-memory FAISS index + mock
embedder so every scoring path runs offline without Vertex or FAISS
file IO.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures — synthetic cards DB + FAISS IDMap + UniverseEmbeddings
# ---------------------------------------------------------------------------

class _DictEmbedder:
    """Returns a pre-defined vector per input text."""
    def __init__(self, mapping: dict[str, np.ndarray]):
        self._m = mapping

    def embed_one(self, text: str) -> np.ndarray:
        return self._m[text]


def _unit(*xs) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class _RoutedStub:
    """Mimics SemanticRouter.RoutedResult enough for the picker."""
    def __init__(self, tables):
        self.top_tables = tables


class _TargetStub:
    def __init__(self, table_id, dataset, aggregate_score):
        self.target_table_id = table_id
        self.target_dataset = dataset
        self.aggregate_score = aggregate_score


@pytest.fixture
def picker_fixture(tmp_path):
    """Build everything the picker needs in isolation."""
    import faiss
    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )

    # 4-D embedding space:
    #   axis 0 = poverty
    #   axis 1 = veterans
    #   axis 2 = housing
    #   axis 3 = children
    VEC_POVERTY = _unit(1, 0, 0, 0)
    VEC_VETERANS = _unit(0, 1, 0, 0)
    VEC_HOUSING = _unit(0, 0, 1, 0)
    VEC_CHILDREN = _unit(0, 0, 0, 1)

    # Universe embeddings for three tables:
    #   B17001 — poverty universe
    #   B21003 — veterans universe
    #   B25070 — housing universe
    # Universes embedded on the appropriate axis.
    ue_path = tmp_path / "universe_embeddings.npz"
    keys = np.array(["B17001|acs/acs5", "B21003|acs/acs5",
                      "B25070|acs/acs5"], dtype=object)
    vectors = np.vstack([VEC_POVERTY, VEC_VETERANS, VEC_HOUSING])
    np.savez(ue_path, keys=keys, vectors=vectors.astype(np.float32))
    ue = UniverseEmbeddings.load(ue_path, expected_dim=None)

    # Metadata DB — variables table + cards table (with embedded
    # var_label_natural cards we'll reference from the FAISS index via
    # their rowid).
    db_path = tmp_path / "meta.db"
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE variables (
            table_id TEXT, dataset TEXT, year INTEGER,
            variable_id TEXT, label TEXT, label_natural TEXT,
            is_total INTEGER, depth INTEGER,
            kind TEXT, is_commonly_queried INTEGER, earns_questions INTEGER,
            concept TEXT, description TEXT, column_name TEXT
        )
    """)
    db.execute("""
        CREATE TABLE cards (
            rowid INTEGER PRIMARY KEY,
            card_type TEXT, text TEXT,
            target_table_id TEXT, target_dataset TEXT,
            target_variable_id TEXT,
            weight REAL, polarity TEXT,
            years_available TEXT NOT NULL DEFAULT '[]',
            geo_levels_available TEXT NOT NULL DEFAULT '[]',
            measure_type TEXT, subject_area TEXT, l1_domains TEXT,
            is_commonly_queried INTEGER, is_total INTEGER, depth INTEGER,
            embedded_at REAL, embedding_version TEXT
        )
    """)
    db.execute("""
        CREATE TABLE denominator_pairs (
            table_id TEXT, dataset TEXT, year INTEGER,
            numerator_var TEXT, denominator_var TEXT
        )
    """)

    # Variables: B21001 contains both _001E (universe — civ pop 18+)
    # and _002E (veterans subgroup). B21003 contains _001E (veterans
    # total) and _002E (WWII veterans).
    def ins_var(t, v, label, is_total=0, depth=0):
        db.execute(
            "INSERT INTO variables "
            "(table_id,dataset,year,variable_id,label,label_natural,"
            " is_total,depth,kind,is_commonly_queried) "
            "VALUES (?,?,2024,?,?,?,?,?,'estimate',0)",
            (t, "acs/acs5", v, label, label, is_total, depth),
        )

    ins_var("B21001", "B21001_001E",
            "Estimate!!Total", is_total=1, depth=0)
    ins_var("B21001", "B21001_002E",
            "Estimate!!Total!!Male:", is_total=0, depth=1)
    ins_var("B21003", "B21003_001E",
            "Estimate!!Total", is_total=1, depth=0)
    ins_var("B17001", "B17001_001E",
            "Estimate!!Total", is_total=1, depth=0)
    ins_var("B17001", "B17001_002E",
            "Estimate!!Total!!Income below poverty level",
            is_total=0, depth=1)
    ins_var("B25070", "B25070_001E",
            "Estimate!!Total", is_total=1, depth=0)

    # Card rows — map each variable to a rowid; those rowids live in
    # FAISS at a vector aligned with its subject axis.
    # Mapping chosen so that:
    #   B21001_001E label points at "poverty" axis (wrong universe!)
    #     — this simulates the real-world failure: a generic label
    #       like "Estimate!!Total" doesn't embed well and ends up
    #       near the dominant concept the label happens to contain.
    #     But the UNIVERSE embedding will correct it.
    #   B21003_001E points at VETERANS — correct universe.
    # For this test we only care that the universe stage steers to
    # B21003 when the query is "veterans". Inside B21003 there's one
    # is_total variable so it wins trivially.
    def ins_card(rowid, tid, vid, vec_axis):
        db.execute(
            "INSERT INTO cards "
            "(rowid,card_type,text,target_table_id,target_dataset,"
            " target_variable_id,weight,polarity,embedded_at) "
            "VALUES (?,'var_label_natural',?,?,?,?,1.0,'positive',0.0)",
            (rowid, f"label for {vid}", tid, "acs/acs5", vid),
        )

    # rowid-to-vector mapping
    labels_to_vec = {
        100: ("B21001", "B21001_001E", VEC_POVERTY),   # mislabel
        101: ("B21001", "B21001_002E", VEC_VETERANS),
        200: ("B21003", "B21003_001E", VEC_VETERANS),
        300: ("B17001", "B17001_001E", VEC_POVERTY),
        301: ("B17001", "B17001_002E", VEC_POVERTY),
        400: ("B25070", "B25070_001E", VEC_HOUSING),
    }
    for rid, (t, v, _vec) in labels_to_vec.items():
        ins_card(rid, t, v, None)
    db.commit()
    db.close()

    # Build a 4-D Flat IDMap2 FAISS index so reconstruct works.
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(4))
    ids = np.array(list(labels_to_vec), dtype=np.int64)
    vecs = np.vstack([labels_to_vec[i][2] for i in ids]).astype(np.float32)
    index.add_with_ids(vecs, ids)

    embedder = _DictEmbedder({
        "poverty rate": VEC_POVERTY,
        "how many veterans are there": VEC_VETERANS,
        "housing cost burden": VEC_HOUSING,
    })

    # Separate read-only connection for the picker (it must not share
    # with our test-scope writer connection).
    ro = sqlite3.connect(db_path)
    ro.row_factory = sqlite3.Row

    picker = UniversePicker(
        universe_emb=ue, faiss_index=index, metadata_db=ro,
        embedder=embedder,
    )
    return picker, embedder


# ---------------------------------------------------------------------------
# Table rescore — universe match should dominate when router is torn
# ---------------------------------------------------------------------------

def test_veterans_query_picks_veterans_table_over_civ_pop_table(picker_fixture):
    """The bug this whole rewrite fixes: B21001 (civilian pop 18+ is
    the universe) scores high on the router, but B21003 (veterans IS
    the universe) is semantically correct for 'how many veterans'.
    Universe-cosine must flip them."""
    picker, _ = picker_fixture
    # Router's best guess puts B21001 and B21003 close together but
    # B21001 has a slightly higher aggregate (a realistic pattern).
    routed = _RoutedStub([
        _TargetStub("B21001", "acs/acs5", 5.2),
        _TargetStub("B21003", "acs/acs5", 4.8),
    ])
    res = picker.pick(query_text="how many veterans are there",
                      routed=routed)
    assert res is not None
    table_id, dataset, variables_obj = res
    assert table_id == "B21003", (
        f"universe picker should prefer B21003 (veterans universe) "
        f"over B21001 (civ pop 18+ universe); got {table_id}"
    )
    # B21003 has a single is_total variable → it wins inside.
    assert variables_obj.value == "B21003_001E"


def test_housing_query_picks_housing_table(picker_fixture):
    picker, _ = picker_fixture
    routed = _RoutedStub([
        _TargetStub("B25070", "acs/acs5", 3.0),
        _TargetStub("B17001", "acs/acs5", 2.0),
    ])
    res = picker.pick(query_text="housing cost burden", routed=routed)
    assert res is not None
    assert res[0] == "B25070"


def test_poverty_query_picks_poverty_table(picker_fixture):
    picker, _ = picker_fixture
    routed = _RoutedStub([
        _TargetStub("B17001", "acs/acs5", 2.0),
        _TargetStub("B21001", "acs/acs5", 5.0),
    ])
    res = picker.pick(query_text="poverty rate", routed=routed)
    assert res is not None
    assert res[0] == "B17001"


# ---------------------------------------------------------------------------
# Graceful fallbacks
# ---------------------------------------------------------------------------

def test_pick_returns_none_for_empty_routed(picker_fixture):
    picker, _ = picker_fixture
    res = picker.pick(query_text="poverty rate",
                      routed=_RoutedStub([]))
    assert res is None


def test_pick_returns_none_when_embedder_rejects_query(picker_fixture):
    picker, embedder = picker_fixture
    # Unseen query text → _DictEmbedder raises KeyError; the picker's
    # embed happens before any table scoring, so the call surfaces
    # the exception. Exercise the error path explicitly so the
    # contract is clear.
    routed = _RoutedStub([_TargetStub("B17001", "acs/acs5", 1.0)])
    with pytest.raises(KeyError):
        picker.pick(query_text="unseen query", routed=routed)


def test_pick_uses_floor_when_universe_missing(picker_fixture, tmp_path):
    """When a table has no universe embedding, the picker falls back
    on the router's prior × universe floor, so it still produces a
    result rather than failing."""
    import faiss
    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )
    # Empty universe embeddings index
    ue_path = tmp_path / "empty_ue.npz"
    np.savez(ue_path,
             keys=np.array([], dtype=object),
             vectors=np.empty((0, 4), dtype=np.float32))
    ue = UniverseEmbeddings.load(ue_path, expected_dim=None)

    picker_orig, embedder = picker_fixture
    picker = UniversePicker(
        universe_emb=ue,
        faiss_index=picker_orig._index,
        metadata_db=picker_orig._db,
        embedder=embedder,
    )
    routed = _RoutedStub([_TargetStub("B17001", "acs/acs5", 2.0)])
    res = picker.pick(query_text="poverty rate", routed=routed)
    assert res is not None
    assert res[0] == "B17001"


# ---------------------------------------------------------------------------
# Variable-level filters
# ---------------------------------------------------------------------------

def test_subgroup_restricted_label_penalty_applies():
    """The race-restricted label heuristic should match known racial-
    stratification markers and miss everything else."""
    from scripts.chatbot.nodes.universe_picker import (
        _is_subgroup_restricted,
    )
    assert _is_subgroup_restricted("Estimate!!Total!!Black alone")
    assert _is_subgroup_restricted("Hispanic or Latino")
    assert _is_subgroup_restricted("Two or more races")
    assert not _is_subgroup_restricted("Estimate!!Total")
    assert not _is_subgroup_restricted("Estimate!!Male")
    assert not _is_subgroup_restricted("")
    assert not _is_subgroup_restricted(None)


def test_table_family_penalty_zeroes_pr_variants():
    from scripts.chatbot.nodes.universe_picker import (
        _table_family_penalty,
    )
    assert _table_family_penalty("B17001PR") == 0.0
    assert _table_family_penalty("S1701PR") == 0.0
    # Race-suffix pattern: single A-I after digits on B/C/S
    assert _table_family_penalty("B17001A") == 0.4
    assert _table_family_penalty("S0701B") == 0.4
    # Unrestricted tables stay at 1.0
    assert _table_family_penalty("B17001") == 1.0
    assert _table_family_penalty("CP03") == 1.0
    # Variable IDs are NOT race-suffix candidates
    assert _table_family_penalty("B19013_001E") == 1.0


# ---------------------------------------------------------------------------
# UniverseEmbeddings load contract
# ---------------------------------------------------------------------------

def test_universe_embeddings_cosine_returns_none_for_missing_key(picker_fixture):
    picker, _ = picker_fixture
    query = _unit(1, 0, 0, 0)
    assert picker._ue.cosine(query, "NOT_A_REAL_KEY|xx") is None
    # In-index key returns a finite cosine
    c = picker._ue.cosine(query, "B17001|acs/acs5")
    assert c is not None
    assert 0.999 <= c <= 1.001   # query axis 0, universe axis 0


def test_universe_embeddings_rejects_wrong_dimension(tmp_path):
    from scripts.chatbot.nodes.universe_picker import UniverseEmbeddings
    p = tmp_path / "bad_dim.npz"
    np.savez(
        p, keys=np.array(["x"], dtype=object),
        vectors=np.zeros((1, 5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dim"):
        UniverseEmbeddings.load(p, expected_dim=768)
