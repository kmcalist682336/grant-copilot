"""Regression guard: runtime embedding calls must stay bounded.

Content embeddings (variable labels, table universes, cards, peer
features) MUST be pre-computed into FAISS / .npz files and looked up
at query time via FAISS ``reconstruct`` or direct numpy indexing —
never regenerated at query time. An earlier bug embedded every
candidate variable's label inside a 1000-iteration loop, adding 40+
seconds to every query.

This test exercises the UniversePicker (the most call-heavy runtime
embed consumer) and asserts that a single pick() call fires no more
than one query-embedding call, regardless of how many candidate
variables are in the top tables. Re-introducing a content-embed-in-
loop pattern will blow the budget and break this test.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from scripts.chatbot.semantic_router import (
    RuntimeEmbedBudgetExceeded, _CountingEmbedder,
)


class _DictEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]):
        self._m = mapping

    def embed_one(self, text: str) -> np.ndarray:
        return self._m[text]


def _unit(*xs) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# CountingEmbedder behavior
# ---------------------------------------------------------------------------

def test_counting_embedder_tracks_call_count():
    base = _DictEmbedder({"a": _unit(1, 0), "b": _unit(0, 1)})
    counter = _CountingEmbedder(base)
    assert counter.call_count == 0
    counter.embed_one("a")
    counter.embed_one("b")
    assert counter.call_count == 2


def test_counting_embedder_raises_on_budget_exceeded():
    base = _DictEmbedder({"a": _unit(1, 0)})
    counter = _CountingEmbedder(base, max_calls=2, session_label="test")
    counter.embed_one("a")
    counter.embed_one("a")
    with pytest.raises(RuntimeEmbedBudgetExceeded, match="test"):
        counter.embed_one("a")


def test_counting_embedder_reset_clears_counter():
    base = _DictEmbedder({"a": _unit(1, 0)})
    counter = _CountingEmbedder(base, max_calls=1)
    counter.embed_one("a")
    with pytest.raises(RuntimeEmbedBudgetExceeded):
        counter.embed_one("a")
    counter.reset()
    # Now allowed again
    counter.embed_one("a")
    assert counter.call_count == 1


# ---------------------------------------------------------------------------
# UniversePicker: the hot-path regression guard
# ---------------------------------------------------------------------------

def _build_picker_with_many_vars(tmp_path, n_vars_per_table: int = 50):
    """Build a picker where the kept tables have lots of variables, so
    any regression that embeds per-variable at runtime would blow the
    embedder budget."""
    import faiss
    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )

    # Simple 3-D embedding space; axis 0 = "poverty".
    VEC_POVERTY = _unit(1, 0, 0)

    ue_path = tmp_path / "universe.npz"
    keys = np.array(["B17001|acs/acs5", "B25070|acs/acs5"], dtype=object)
    vectors = np.vstack([VEC_POVERTY, _unit(0, 1, 0)])
    np.savez(ue_path, keys=keys, vectors=vectors.astype(np.float32))
    ue = UniverseEmbeddings.load(ue_path, expected_dim=None)

    db = sqlite3.connect(tmp_path / "meta.db")
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

    # Populate each of two tables with many variables. Half of them
    # get embedded cards (rowid in FAISS); the other half do NOT.
    # A regression that runtime-embeds non-cached labels would hit
    # the budget on this half.
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(3))
    ids, vecs = [], []
    next_rowid = 1
    for table in ("B17001", "B25070"):
        for i in range(n_vars_per_table):
            vid = f"{table}_{i:03d}E"
            label = f"Variable {table} {i} label text"
            db.execute(
                "INSERT INTO variables "
                "(table_id,dataset,year,variable_id,label,label_natural,"
                " is_total,depth,kind,is_commonly_queried) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (table, "acs/acs5", 2024, vid, label, label,
                 1 if i == 0 else 0, 0 if i == 0 else 1,
                 "estimate", 0),
            )
            # Only even-indexed variables get a card (simulates
            # stage-7 label_natural coverage gaps on real tables).
            if i % 2 == 0:
                rowid = next_rowid
                next_rowid += 1
                db.execute(
                    "INSERT INTO cards (rowid, card_type, text, "
                    " target_table_id, target_dataset, "
                    " target_variable_id, weight, polarity, "
                    " embedded_at) "
                    "VALUES (?,'var_label_natural',?,?,?,?,1.0,'positive',0.0)",
                    (rowid, label, table, "acs/acs5", vid),
                )
                ids.append(rowid)
                vecs.append(_unit(0.5 + i * 0.01, 0.5, 0.0))
    db.commit()
    db.close()

    index.add_with_ids(
        np.vstack(vecs).astype(np.float32),
        np.array(ids, dtype=np.int64),
    )

    embedder = _DictEmbedder({"poverty rate": VEC_POVERTY})
    counter = _CountingEmbedder(embedder, max_calls=3,
                                 session_label="picker")

    ro = sqlite3.connect(tmp_path / "meta.db")
    ro.row_factory = sqlite3.Row
    picker = UniversePicker(
        universe_emb=ue, faiss_index=index, metadata_db=ro,
        embedder=counter,
    )
    return picker, counter


class _RoutedStub:
    def __init__(self, tables):
        self.top_tables = tables


class _TargetStub:
    def __init__(self, table_id, dataset, aggregate_score):
        self.target_table_id = table_id
        self.target_dataset = dataset
        self.aggregate_score = aggregate_score


def test_picker_respects_runtime_embed_budget(tmp_path):
    """A single pick() call must not exceed the per-session embed
    budget, regardless of how many candidate variables are in the
    kept tables. Tight budget (max_calls=3) catches any regression
    that re-introduces the loop-embed pattern."""
    picker, counter = _build_picker_with_many_vars(
        tmp_path, n_vars_per_table=50,
    )
    routed = _RoutedStub([_TargetStub("B17001", "acs/acs5", 5.0)])
    # Must complete without raising — budget is 3, actual calls
    # should be exactly 1 (the query embed).
    picker.pick(query_text="poverty rate", routed=routed)
    assert counter.call_count == 1, (
        f"single pick() should fire exactly one query embed; "
        f"got {counter.call_count}. If this count is > 1, a new "
        f"code path in UniversePicker is calling embed_one on "
        f"content rather than looking it up from the FAISS index."
    )


def test_picker_cache_prevents_query_re_embedding(tmp_path):
    """Second pick() on the same query_text must hit the query
    embedding cache — no new runtime embed call."""
    picker, counter = _build_picker_with_many_vars(
        tmp_path, n_vars_per_table=20,
    )
    routed = _RoutedStub([_TargetStub("B17001", "acs/acs5", 5.0)])
    picker.pick(query_text="poverty rate", routed=routed)
    first = counter.call_count
    picker.pick(query_text="poverty rate", routed=routed)
    assert counter.call_count == first, (
        f"second pick() on same query must hit the per-concept cache; "
        f"got {counter.call_count - first} additional call(s)"
    )


def test_picker_does_not_embed_unindexed_variable_labels(tmp_path):
    """Variables without an embedded var_label_natural card must be
    skipped, not runtime-embedded. Tight budget verifies."""
    picker, counter = _build_picker_with_many_vars(
        tmp_path, n_vars_per_table=100,   # half without cards = 50 no-card vars
    )
    routed = _RoutedStub([_TargetStub("B17001", "acs/acs5", 5.0)])
    # Under the old fallback-to-embed-one logic, picker would have
    # attempted 50+ embed_one() calls here. With the guard at 3, this
    # would raise.
    picker.pick(query_text="poverty rate", routed=routed)
    assert counter.call_count == 1
