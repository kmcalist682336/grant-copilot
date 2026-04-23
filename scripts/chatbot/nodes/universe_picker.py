"""Universe-aware variable picker — semantic alternative to the
weight-stacking composer in :mod:`scripts.chatbot.planner`.

Two-stage flow:

  1. **Table rescore**: for every candidate table from the semantic
     router, compute ``cos(query_embedding, universe_embedding) *
     router_prior``. The pre-computed universe embeddings live at
     ``data/metadata/universe_embeddings.npz``; they capture each
     table's ``universe_plain`` text from stage 4, i.e. who the table
     counts. Queries like "veterans in Cobb" prefer tables whose
     universe IS veterans (B21003) over tables where veterans is a
     subgroup (B21001).

  2. **Variable pick**: within the top-ranked tables, score each
     candidate variable by ``cos(query_embedding, label_natural_embedding)``.
     We reuse the already-embedded ``var_label_natural`` cards from the
     live FAISS index via ``index.reconstruct(rowid)`` — no Vertex call
     per candidate.

     Minimal filters apply: race/PR suffix gets a 0.4× multiplier;
     is_total gets a small 1.05× prior. No keyword lists, no
     hand-tuned +5/-8 constants.

Fallback: when the picker can't score (missing universe embedding, no
embedded label cards, etc.), it returns ``None`` and the planner falls
back to the legacy ``_compose_from_router`` flow. Safe to turn on
incrementally.

Public API:
    UniversePicker
        pick(query_text, routed, metadata_db, data_level)
            → Optional[tuple[table_id, dataset, ConceptVariables]]
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from scripts.chatbot.concept_map import ConceptVariables

logger = logging.getLogger(__name__)


_INVALID_DENOMINATORS = {
    "GEO_ID", "NAME", "STATE", "COUNTY", "TRACT", "PLACE",
    "state", "county", "tract", "place", "name", "geo_id",
}

# Race-stratified table suffix: single A-I after 4-5 digits on B/C/S
# prefix. These should not win a generic concept query.
_RACE_SUFFIX_RE = re.compile(r"^[BCS]\d{4,5}[A-I]$")

# Embedding dim must match whatever Vertex text-embedding-005 emits
# and what FAISS stores. 768 for text-embedding-005.
EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ‖v‖ with a small epsilon to avoid div-by-zero on a
    degenerate zero vector. Input may be 1-D (single) or 2-D (N×D)."""
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-9)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _table_family_penalty(table_id: str) -> float:
    """Minimal filter layer: gently penalize race-stratified variants.
    Everything else stays at 1.0 — the router's semantic signal has
    already filtered poorly-matching tables."""
    if _RACE_SUFFIX_RE.match(table_id):
        return 0.4
    if table_id.endswith("PR"):
        return 0.0
    return 1.0


def _is_subgroup_restricted(label: Optional[str]) -> bool:
    """Match the filter in planner._is_subgroup_restricted — a label
    that names a race/sex/disability restriction inside a table should
    not outrank the overall variable. Returns True for labels like
    'Estimate!!Total!!Female:!!Black alone' but NOT for 'Estimate!!
    Total' or 'Estimate!!Total:!!Male:!!Under 18 years' (the latter
    is a normal subgroup of the top-level universe)."""
    if not label:
        return False
    t = label.lower()
    markers = (
        "black alone", "white alone", "asian alone",
        "american indian", "native hawaiian",
        "two or more races", "some other race",
        "hispanic or latino", "not hispanic",
    )
    return any(m in t for m in markers)


_LEAF_SUBGROUP_MARKERS = (
    # age brackets commonly appearing in ACS leaf labels
    "18 to 19", "20 to 24", "25 to 29", "30 to 34", "35 to 39",
    "40 to 44", "45 to 54", "55 to 59", "60 to 61", "62 to 64",
    "65 to 66", "67 to 69", "70 to 74", "75 years and over",
    "75 to 84", "85 years",
    "under 5", "under 18", "under 6",
    "5 to 9", "5 to 17", "10 to 14", "15 to 17",
    # sex markers when they're the terminal segment of the label
    "!!male:!!", "!!female:!!",
    # household composition leaves
    "with own children under 18", "with related children",
    # educational-leaf markers
    "less than 9th grade", "9th to 12th grade",
)


def _is_leaf_subgroup(label: Optional[str]) -> bool:
    """True when the label contains an age bracket, sex marker, or
    other 'leaf' restriction that narrows the variable to a specific
    demographic slice. Leaf subgroups should lose to the table's
    overall/universe variable on generic queries ('how many veterans').
    """
    if not label:
        return False
    t = label.lower()
    return any(m in t for m in _LEAF_SUBGROUP_MARKERS)


# ---------------------------------------------------------------------------
# Universe embeddings loader
# ---------------------------------------------------------------------------

@dataclass
class UniverseEmbeddings:
    """Load pre-computed universe embeddings keyed on 'table_id|dataset'."""

    path: Path
    _keys: list[str]
    _key_to_idx: dict[str, int]
    _vectors: np.ndarray     # (N, EMBED_DIM), L2-normalized

    @classmethod
    def load(
        cls, path: Path, *, expected_dim: Optional[int] = EMBED_DIM,
    ) -> "UniverseEmbeddings":
        """Load from a numpy .npz file. ``expected_dim`` is a sanity
        check for production (text-embedding-005 is 768-D); tests can
        pass ``expected_dim=None`` to use a smaller synthetic vector
        space."""
        data = np.load(path, allow_pickle=True)
        keys = list(data["keys"])
        vectors = data["vectors"].astype(np.float32)
        if vectors.size and expected_dim is not None and \
                vectors.shape[1] != expected_dim:
            raise ValueError(
                f"universe embeddings dim {vectors.shape[1]} != "
                f"expected {expected_dim}"
            )
        return cls(
            path=path, _keys=keys,
            _key_to_idx={k: i for i, k in enumerate(keys)},
            _vectors=vectors,
        )

    def cosine(self, query_vec: np.ndarray, key: str) -> Optional[float]:
        """Cosine(query, universe[key]). Returns None when the key
        isn't in the index. query_vec must already be L2-normalized."""
        idx = self._key_to_idx.get(key)
        if idx is None:
            return None
        return float(query_vec @ self._vectors[idx])


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------

class UniversePicker:
    """Universe-aware semantic picker. One instance per SemanticRouter;
    shares its FAISS index and metadata DB connection.

    Construction:
        picker = UniversePicker(
            universe_emb=UniverseEmbeddings.load(...),
            faiss_index=semantic_router.index,
            metadata_db=semantic_router.metadata_db,
            embedder=semantic_router.embedder,
        )

    Query-time cost: one Vertex embed for the query (already paid by
    the router), plus ~2-20 FAISS ``reconstruct`` calls (no network).
    No Vertex calls for candidate scoring.
    """

    def __init__(
        self,
        *,
        universe_emb: UniverseEmbeddings,
        faiss_index,
        metadata_db: sqlite3.Connection,
        embedder,
        top_tables_to_rescore: int = 10,
        top_tables_to_keep: int = 2,
        subgroup_suffix_penalty: float = 0.4,
        leaf_subgroup_penalty: float = 0.3,
        is_total_bonus: float = 1.3,
        min_universe_floor: float = 0.30,
    ):
        self._ue = universe_emb
        self._index = faiss_index
        self._db = metadata_db
        self._embedder = embedder
        self._top_n_rescore = top_tables_to_rescore
        self._top_n_keep = top_tables_to_keep
        self._subgroup_penalty = subgroup_suffix_penalty
        self._leaf_penalty = leaf_subgroup_penalty
        self._is_total_bonus = is_total_bonus
        # Per-concept query-embedding cache so a query with 5
        # comparator geos doesn't embed the same concept text 5 times.
        # Cleared explicitly via ``clear_cache`` if callers want to
        # force a refresh; otherwise lives for the picker's lifetime.
        self._query_emb_cache: dict[str, np.ndarray] = {}
        # Per-(table,dataset) candidate-variable cache — the SQL query
        # and its FAISS-reconstruct results are idempotent, so
        # memoizing avoids hundreds of repeated reads when the planner
        # calls us across comparator geos.
        self._var_cache: dict[tuple[str, str], list[dict]] = {}
        # FAISS reconstruct cache keyed on card rowid. The dominant
        # cost inside pick() is ~200 reconstruct() calls on a 4 GB
        # HNSWSQ index (~100 ms each). Caching by rowid drops the
        # second-and-subsequent pick() call for the same query from
        # 20+ s to milliseconds.
        self._label_vec_cache: dict[int, np.ndarray] = {}
        # When a table has no universe embedding, use this floor so the
        # router's prior still carries it — otherwise it gets zeroed
        # out and the picker has nothing to fall back to.
        self._univ_floor = min_universe_floor

    # ------------------------------------------------------------------
    # Stage 1 — rescore router's top tables by universe-similarity
    # ------------------------------------------------------------------

    def _rescore_tables(
        self, query_vec: np.ndarray, routed,
    ) -> list[tuple[float, object]]:
        """Return [(combined_score, target), …] sorted desc."""
        out: list[tuple[float, object]] = []
        for t in list(routed.top_tables)[: self._top_n_rescore]:
            key = f"{t.target_table_id}|{t.target_dataset}"
            u_cos = self._ue.cosine(query_vec, key)
            if u_cos is None:
                u_cos = self._univ_floor
            else:
                # Cosine of L2-normalized vectors ∈ [-1, 1]; clip
                # negative values to 0 so they can't dominate.
                u_cos = max(0.0, u_cos)
            # Router's aggregate_score is unbounded; cap at 10 for a
            # [0,1] prior. A score of 10 is already very strong.
            router_prior = min(float(t.aggregate_score), 10.0) / 10.0
            family = _table_family_penalty(t.target_table_id)
            combined = router_prior * u_cos * family
            out.append((combined, t))
        out.sort(key=lambda s: s[0], reverse=True)
        return out

    # ------------------------------------------------------------------
    # Stage 2 — pick the best variable within the kept tables
    # ------------------------------------------------------------------

    def _candidate_variables(
        self,
        table_id: str, dataset: str,
    ) -> list[dict]:
        """Return the table's estimate variables with their cached
        label-natural embedding rowid, if one exists. Memoized per
        (table_id, dataset) so repeat calls across comparator geos
        don't re-query the metadata DB."""
        cache_key = (table_id, dataset)
        cached = self._var_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self._db.execute(
            """
            SELECT v.variable_id, v.label_natural, v.label,
                   v.is_total, v.depth, v.is_commonly_queried,
                   c.rowid AS card_rowid
              FROM variables v
              LEFT JOIN cards c
                ON c.card_type = 'var_label_natural'
               AND c.target_table_id = v.table_id
               AND c.target_dataset  = v.dataset
               AND c.target_variable_id = v.variable_id
             WHERE v.table_id = ?
               AND v.dataset  = ?
               AND v.kind     = 'estimate'
            """,
            (table_id, dataset),
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            if r["variable_id"] in _INVALID_DENOMINATORS:
                continue
            out.append({
                "variable_id": r["variable_id"],
                "label": r["label_natural"] or r["label"] or "",
                "is_total": bool(r["is_total"]),
                "depth": r["depth"],
                "is_commonly_queried": bool(r["is_commonly_queried"]),
                "card_rowid": r["card_rowid"],
            })
        self._var_cache[cache_key] = out
        return out

    def _get_label_vec(
        self, card_rowid: Optional[int], fallback_label: str,
    ) -> Optional[np.ndarray]:
        """Retrieve the variable's label embedding.

        Preferred path: pull the already-stored ``var_label_natural``
        vector out of FAISS via ``reconstruct(rowid)`` — costs nothing
        after the first call (results are cached on ``_label_vec_cache``).

        No runtime Vertex fallback: tables like CP03 have 1k+ variables,
        most without pre-embedded labels, and per-call Vertex embeds
        dominated query latency (~40ms × 1k vars = 40s+ per pick).
        When a variable has no embedded label, we return None and the
        variable gets skipped from scoring — the remaining scored
        candidates are almost always sufficient. If the entire top
        tables have no embeddings, the picker returns None and the
        caller falls back to the legacy composer.
        """
        if card_rowid is None:
            return None
        rid = int(card_rowid)
        cached = self._label_vec_cache.get(rid)
        if cached is not None:
            return cached
        try:
            vec = self._index.reconstruct(rid)
        except Exception as e:
            logger.debug("reconstruct(%s) failed: %s", card_rowid, e)
            return None
        normed = _l2_normalize(np.asarray(vec, dtype=np.float32))
        self._label_vec_cache[rid] = normed
        return normed

    def _score_variable(
        self, query_vec: np.ndarray, var: dict,
    ) -> Optional[float]:
        lvec = self._get_label_vec(var["card_rowid"], var["label"])
        if lvec is None:
            return None
        cos = float(query_vec @ lvec)
        # Clip negative cosines to 0 so they can't beat a near-zero
        # match when filters multiply in.
        cos = max(0.0, cos)
        if _is_subgroup_restricted(var["label"]):
            cos *= self._subgroup_penalty
        # Age-bracket / sex-marker leaf labels lose to the overall
        # variable on generic subject queries. A label like "Total
        # Male 75 years and over Veteran" matches 'veterans' strongly
        # on embedding but is obviously the wrong answer to
        # "how many veterans are there". The _001E universe should win.
        if _is_leaf_subgroup(var["label"]):
            cos *= self._leaf_penalty
        if var["is_total"]:
            cos *= self._is_total_bonus
        return cos

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def pick(
        self,
        *,
        query_text: str,
        routed,
        data_level: Optional[str] = None,
    ) -> Optional[tuple[str, str, ConceptVariables]]:
        """Return ``(table_id, dataset, ConceptVariables)`` or None.

        Returning None lets the caller fall back to the legacy
        ``_compose_from_router`` picker — which is the intended
        behavior when the universe embeddings don't cover a table
        vintage or when the router handed us nothing.
        """
        import time as _time
        _t0 = _time.time()
        if not routed or not getattr(routed, "top_tables", None):
            return None

        _t_embed = _time.time()
        query_vec = self._query_emb_cache.get(query_text)
        cache_hit = query_vec is not None
        if query_vec is None:
            q = self._embedder.embed_one(query_text)
            query_vec = _l2_normalize(np.asarray(q, dtype=np.float32))
            self._query_emb_cache[query_text] = query_vec
        _t_embed_elapsed = _time.time() - _t_embed

        _t_rescore = _time.time()
        ranked = self._rescore_tables(query_vec, routed)
        _t_rescore_elapsed = _time.time() - _t_rescore
        if not ranked or ranked[0][0] <= 0:
            return None
        kept = ranked[: self._top_n_keep]

        # Score every candidate variable across the kept tables;
        # highest cosine wins, with table-level ties broken by which
        # table ranked higher at stage 1.
        _t_vars = _time.time()
        n_vars_examined = 0
        n_reconstruct = 0
        best: Optional[tuple[float, str, str, dict]] = None
        for rank_idx, (table_score, target) in enumerate(kept):
            table_id = target.target_table_id
            dataset = target.target_dataset
            vars_ = self._candidate_variables(table_id, dataset)
            for v in vars_:
                n_vars_examined += 1
                if v["card_rowid"] is not None:
                    n_reconstruct += 1
                cos = self._score_variable(query_vec, v)
                if cos is None:
                    continue
                # Compose with table score so stronger-table
                # variables win ties.
                combined = cos * (1.0 + table_score)
                if best is None or combined > best[0]:
                    best = (combined, table_id, dataset, v)
        _t_vars_elapsed = _time.time() - _t_vars
        _t_total = _time.time() - _t0
        if _t_total > 1.0:
            kept_desc = ",".join(
                f"{t.target_table_id}/{t.target_dataset}"
                for _, t in kept
            )
            logger.warning(
                "[SLOW PICK] concept=%r total=%.2fs  "
                "embed=%.2fs(cache=%s)  rescore=%.2fs  "
                "vars_stage=%.2fs  n_vars=%d  n_reconstruct=%d  "
                "kept=[%s]  label_vec_cache_size=%d",
                query_text, _t_total, _t_embed_elapsed,
                "HIT" if cache_hit else "MISS",
                _t_rescore_elapsed, _t_vars_elapsed,
                n_vars_examined, n_reconstruct, kept_desc,
                len(self._label_vec_cache),
            )

        if best is None:
            return None
        _, winner_table, winner_dataset, winner_var = best
        logger.info(
            "universe_picker: %s / %s  label=%s  is_total=%s",
            winner_table, winner_var["variable_id"],
            (winner_var.get("label") or "")[:60],
            winner_var.get("is_total"),
        )

        # Denominator pair lookup — identical semantics to the legacy
        # composer. When the winning variable is flagged is_total we
        # use it as `value`; otherwise we check if stage 3 identified
        # a denominator pair and compose a ratio.
        if winner_var["is_total"]:
            return winner_table, winner_dataset, ConceptVariables(
                value=winner_var["variable_id"],
            )
        pair = self._db.execute(
            "SELECT denominator_var FROM denominator_pairs "
            "WHERE table_id = ? AND dataset = ? AND numerator_var = ?",
            (winner_table, winner_dataset, winner_var["variable_id"]),
        ).fetchone()
        if pair and pair[0] and pair[0] not in _INVALID_DENOMINATORS:
            return winner_table, winner_dataset, ConceptVariables(
                numerator=winner_var["variable_id"],
                denominator=pair[0],
            )
        return winner_table, winner_dataset, ConceptVariables(
            value=winner_var["variable_id"],
        )
