"""K.4 — semantic routing layer.

Replaces the Tier 1/2/3 cascade with a single vector-retrieval pass over
the unified card index built in K.2-K.3. Embeds the user concept, ANN
searches FAISS, joins the cards SQLite for structured filters, applies
weights and polarity, aggregates per (table, variable) target, and
returns a ranked candidate set.

The router is **pure retrieval**. Variable-role composition (value vs
numerator+denominator vs components) lives in the planner so this
module can stay test-isolated.

Public API:
    SemanticRouter
        .route(query, *, top_k=10, ...) → RoutedResult
    RoutedHit
    RoutedTarget
    RoutedResult
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parse helpers for the per-card coverage arrays
# ---------------------------------------------------------------------------

def _parse_json_int_list(raw) -> list[int]:
    """Parse a JSON array of integers stored as TEXT; return [] on
    NULL / malformed / non-list values. Used for years_available."""
    if raw is None:
        return []
    try:
        v = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(v, list):
        return []
    out: list[int] = []
    for x in v:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return sorted(set(out))


def _parse_json_str_list(raw) -> list[str]:
    """Parse a JSON array of strings stored as TEXT; return [] on
    NULL / malformed / non-list values. Used for geo_levels_available."""
    if raw is None:
        return []
    try:
        v = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(v, list):
        return []
    return sorted({str(x) for x in v if x is not None})


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class RoutedHit(BaseModel):
    """One card returned by FAISS, with its joined metadata + scoring.

    Schema note: post-Phase-2.5 swap, question-type cards carry
    ``target_year=None`` and instead expose ``years_available`` /
    ``geo_levels_available`` as JSON arrays. Legacy callers that saw
    ``target_year: int`` now see ``Optional[int]``; pick the year at
    plan time from intent + ``years_available``.
    """

    model_config = ConfigDict(extra="ignore")

    rowid: int
    card_type: str
    text: str
    target_table_id: str
    target_dataset: str
    target_year: Optional[int] = None
    target_variable_id: Optional[str] = None
    cosine: float                    # raw similarity ∈ [-1, 1]
    weight: float                    # per-card weight from build_cards
    polarity: str                    # 'positive' | 'negative'
    weighted_score: float            # cosine * weight, negated if polarity=='negative'
    # structured signals
    measure_type: Optional[str] = None
    subject_area: Optional[str] = None
    l1_domains: Optional[str] = None
    is_commonly_queried: Optional[int] = None
    is_total: Optional[int] = None
    depth: Optional[int] = None
    # Coverage (post-reorg schema). Parsed from the per-card JSON
    # columns at hydrate time so the filter step doesn't need to join
    # against the coverage table.
    years_available: list[int] = Field(default_factory=list)
    geo_levels_available: list[str] = Field(default_factory=list)


class RoutedTarget(BaseModel):
    """One table OR variable target with the aggregated score across all
    its hit cards. The aggregate is the SUM of weighted scores from
    every card whose target matches — convergent evidence boosts rank.

    Aggregation keys on ``(table, dataset, variable?)`` only — years are
    no longer part of the key because question cards are year-agnostic
    in the post-swap schema. Callers who need a specific year consult
    ``years_available`` (the union across all merged hits).
    """

    model_config = ConfigDict(extra="ignore")

    target_table_id: str
    target_dataset: str
    target_year: Optional[int] = None          # legacy — NULL for new cards
    target_variable_id: Optional[str] = None   # None = table-level target

    aggregate_score: float
    n_positive_hits: int
    n_negative_hits: int
    best_hit: RoutedHit                   # the highest-cosine matching card
    matched_card_types: list[str] = Field(default_factory=list)
    # Union of the per-card coverage arrays across all hits that fed
    # this target. Used by the planner to pick the live year/level.
    years_available: list[int] = Field(default_factory=list)
    geo_levels_available: list[str] = Field(default_factory=list)


class RoutedResult(BaseModel):
    """Full router output for one query."""

    model_config = ConfigDict(extra="ignore")

    query: str
    top_tables: list[RoutedTarget] = Field(default_factory=list)
    top_variables: list[RoutedTarget] = Field(default_factory=list)
    raw_hits: list[RoutedHit] = Field(default_factory=list)
    n_hits_retrieved: int = 0
    n_hits_after_filter: int = 0


# ---------------------------------------------------------------------------
# Embedder protocol — accept anything with .embed_one(text) -> np.ndarray
# ---------------------------------------------------------------------------
#
# INVARIANT: runtime embedding calls are for QUERIES ONLY.
#
# Every content item that needs an embedding — table universes,
# variable labels, card text, peer features — is pre-computed by
# tools/build_embedding_index, tools/build_universe_embeddings, etc.
# Embedding those at runtime defeats the whole point of the FAISS
# index. An earlier bug in UniversePicker called embed_one(label) in
# a 1000-variable loop on every query, burning 40+ seconds per pick;
# see commit history / task #33.
#
# The `RuntimeEmbedBudgetExceeded` check in _CountingEmbedder below
# is the guard against that class of regression. Tests wire it up
# with a tight budget (<=N calls per pick); production disables the
# hard limit but still records the call count for observability.


class RuntimeEmbedBudgetExceeded(RuntimeError):
    """Raised when a runtime embedder exceeds its per-session budget.

    Content embeddings must be pre-computed, not generated at query
    time — if a new code path triggers this, the fix is almost always
    to pre-embed the content offline (see tools/build_*.py) and look
    the vector up via FAISS reconstruct(rowid), not to raise the
    budget.
    """


class _CountingEmbedder:
    """Wraps an underlying embedder with a call counter + optional
    per-session budget. Use via ``CallBudget.session`` context
    manager to make regressions loud.

    Not exposed directly — callers instantiate VertexEmbedder /
    construct mock embedders; this wrapper is applied internally by
    the orchestrator when observability is requested.
    """

    def __init__(
        self, inner, *, max_calls: Optional[int] = None,
        session_label: str = "session",
    ):
        self._inner = inner
        self._max = max_calls
        self._session_label = session_label
        self.call_count = 0

    def reset(self) -> None:
        self.call_count = 0

    def embed_one(self, text: str) -> np.ndarray:
        self.call_count += 1
        if self._max is not None and self.call_count > self._max:
            raise RuntimeEmbedBudgetExceeded(
                f"{self._session_label}: exceeded runtime embed "
                f"budget of {self._max} calls. Content embeddings "
                f"must be pre-computed, not generated at query time. "
                f"Check for loops calling embed_one on table / "
                f"variable / description text."
            )
        return self._inner.embed_one(text)


class Embedder:
    """Minimal protocol the router calls. Production = Vertex; tests = mock."""
    def embed_one(self, text: str) -> np.ndarray: ...


class VertexEmbedder:
    """Wraps google-genai for single-query embeds at runtime.

    See the INVARIANT note above: call this at runtime only for user
    queries, never to embed content we already have stored. For bulk
    content embedding, use ``tools/build_embedding_index.VertexEmbedder``
    (different class, supports region rotation + batching).

    ``timeout_seconds`` is enforced via a ThreadPoolExecutor wrap —
    google-genai's embed_content doesn't honor a per-call timeout on
    its own, so without this a stuck Vertex embed would hang the
    whole query (we call this once per rewritten concept on every
    user query, so it's a hot-path liability).
    """

    def __init__(self, project: str, location: str,
                 model: str = "text-embedding-005", dim: int = 768,
                 timeout_seconds: float = 30.0):
        from google import genai
        from google.genai import types
        self._client = genai.Client(
            vertexai=True, project=project, location=location,
        )
        self._types = types
        self._model = model
        self._dim = dim
        self._timeout_seconds = timeout_seconds
        # Observability counter — orchestrator / tests may read this
        # to enforce the "runtime embeds are bounded" invariant.
        self.call_count = 0

    def embed_one(self, text: str) -> np.ndarray:
        import threading
        self.call_count += 1

        holder: dict = {"resp": None, "error": None}

        def _invoke():
            try:
                holder["resp"] = self._client.models.embed_content(
                    model=self._model,
                    contents=[text],
                    config=self._types.EmbedContentConfig(
                        output_dimensionality=self._dim,
                    ),
                )
            except BaseException as e:                 # noqa: BLE001
                holder["error"] = e

        worker = threading.Thread(
            target=_invoke, name="vertex-embed", daemon=True,
        )
        worker.start()
        worker.join(timeout=self._timeout_seconds)
        if worker.is_alive():
            raise RuntimeError(
                f"Vertex embed timed out after "
                f"{self._timeout_seconds}s for text[:40]={text[:40]!r}"
            )
        if holder["error"] is not None:
            raise holder["error"]
        resp = holder["resp"]
        return np.array(resp.embeddings[0].values, dtype=np.float32)


# ---------------------------------------------------------------------------
# SemanticRouter
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Tunable knobs. Defaults are sensible PoC starting points; the K.6
    eval is expected to refine them."""
    # FAISS over-fetch — we ANN top_k * over_fetch_factor candidates
    # before SQLite-side filtering, since negatives + topic facets may
    # remove a lot.
    over_fetch_factor: int = 10

    # If query embedding distance to a card is worse than this floor we
    # drop it before scoring (defensive; FAISS HNSW returns L2 distance).
    max_l2_distance: float = 1.5

    # Aggregate score floor — targets below this are dropped from results.
    min_aggregate_score: float = 0.05

    # Table-family bias multipliers. The Census table catalog has
    # overlapping "canonical detailed" (B-series) vs. "profile"
    # (DP/S/CP/K) families for many concepts. Profile tables have
    # simpler user-facing question phrasings, so embedding retrieval
    # systematically over-ranks them for common queries ("poverty
    # rate", "median income"). These multipliers re-balance the
    # aggregate scores so B-series wins the tie.
    #
    # Keys are prefix strings matched against the start of table_id.
    # Special key "_PR_SUFFIX" matches table_ids ending in "PR"
    # (Puerto Rico-only tables; excluded for GA-scoped v1).
    #
    # NOTE: prefixes are matched longest-first, so "DP" beats "D" and
    # "CP" beats "C". Missing prefix → weight 1.0 (no change).
    table_family_bias: bool = True
    table_family_weights: dict = field(default_factory=lambda: {
        "B":           1.00,   # Detailed Tables — canonical
        "C":           0.90,   # Collapsed Tables
        "S":           0.60,   # Subject Tables (profile)
        # Profile tables (DP*, CP*) are pre-computed summaries. Their
        # variables cover a subset of each subject area, often only at
        # MSA+/state/US geographies, and for queries where a B-table
        # exists they tend to be worse answers. The v1 card corpus has
        # enough redundant cards on these tables that even a 0.40 bias
        # doesn't always flip the ranking; the 0.20–0.25 range below
        # aligns the final order with the B-table-first preference
        # without eliminating profile tables entirely.
        "DP":          0.20,   # Data Profiles (down from 0.50)
        "CP":          0.20,   # Comparison Profiles (down from 0.40)
        "K":           0.40,   # Special Tabulations
        "_PR_SUFFIX":  0.0,    # Puerto Rico variants (GA v1: excluded)
        # Race/ethnicity-stratified table variants. Census appends a
        # single A-I suffix to denote a single-race-or-Hispanic
        # restriction:
        #   A=White, B=Black, C=AIAN, D=Asian, E=NHPI, F=Other,
        #   G=Two+, H=White-non-Hispanic, I=Hispanic.
        # These should not outrank the overall (no-suffix) table for
        # generic queries — only fire when the user explicitly asks
        # about race breakdowns (J.5 disparity territory, deferred).
        "_RACE_SUFFIX": 0.30,
    })


# ---------------------------------------------------------------------------
# Table-family bias helpers
# ---------------------------------------------------------------------------

_RACE_SUFFIX_RE = re.compile(r"^[BCS]\d{4,5}[A-I]$")


def _table_family_weight(
    table_id: str, weights: dict,
) -> float:
    """Return the bias multiplier for one table_id.

    Matching rules (most-specific wins; multipliers compose):
      - PR-suffix Puerto Rico variants → '_PR_SUFFIX' (multiplied
        with prefix weight).
      - Race/ethnicity-stratified tables (B19013A-I, B17001A-I,
        S0701A-I, etc.) → '_RACE_SUFFIX' (multiplied with prefix
        weight). Detected via the pattern <prefix><digits><A-I>.
      - Longest-prefix match against the non-underscored keys
        (so 'DP' wins over 'D').
      - Unrecognized → 1.0 (no change).
    """
    weight = 1.0
    # Prefix lookup
    best_key: Optional[str] = None
    for key in weights:
        if key.startswith("_"):
            continue
        if table_id.startswith(key):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is not None:
        weight *= weights[best_key]
    # PR suffix multiplier
    if table_id.endswith("PR"):
        pr = weights.get("_PR_SUFFIX")
        if pr is not None:
            weight *= pr
    # Race suffix multiplier (single trailing letter A-I after digits)
    if _RACE_SUFFIX_RE.match(table_id):
        race = weights.get("_RACE_SUFFIX")
        if race is not None:
            weight *= race
    return weight


def _apply_table_family_bias(
    targets: dict, weights: dict,
) -> None:
    """Multiply each target's aggregate_score by its family weight in
    place. No-op for a target whose prefix isn't in ``weights``.
    """
    for target in targets.values():
        w = _table_family_weight(target.target_table_id, weights)
        if w != 1.0:
            target.aggregate_score *= w


class SemanticRouter:
    """Vector-first retrieval over the unified card index."""

    def __init__(
        self,
        index_path: Path,
        metadata_db_path: Path,
        embedder: Embedder,
        *,
        config: Optional[RouterConfig] = None,
    ):
        """Args:
            index_path: FAISS index file (built by K.3).
            metadata_db_path: search_index.db — contains cards, coverage,
                tables, variables (single unified store).
            embedder: anything with .embed_one(text) -> np.ndarray.
                Production = VertexEmbedder; tests = mock.
            config: RouterConfig overrides.
        """
        import faiss
        self._faiss = faiss
        self.index = faiss.read_index(str(index_path))
        self.metadata_db = sqlite3.connect(
            metadata_db_path, check_same_thread=False,
        )
        self.metadata_db.row_factory = sqlite3.Row
        self.embedder = embedder
        self.config = config or RouterConfig()
        logger.info(
            "SemanticRouter loaded: %s vectors, dim=%d",
            self.index.ntotal, self.index.d,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        *,
        top_k: int = 10,
        l1_domain_filter: Optional[list[str]] = None,
        require_coverage_at: Optional[list[str]] = None,
        boost_commonly_queried: bool = True,
    ) -> RoutedResult:
        """Search → filter → aggregate → rank. See module docstring."""
        if not query.strip():
            return RoutedResult(query=query)

        # 1. Embed query
        query_vec = self.embedder.embed_one(query).reshape(1, -1)

        # 2. ANN
        n_fetch = top_k * self.config.over_fetch_factor
        D, I = self.index.search(query_vec, k=n_fetch)
        D = D[0]   # (n_fetch,)
        I = I[0]   # (n_fetch,)

        # 3. Hydrate hits with metadata via SQLite JOIN
        hits = self._hydrate(I, D)
        n_total = len(hits)

        # 4. Drop hits beyond the hard distance floor
        hits = [h for h in hits if h.cosine >= -self.config.max_l2_distance]

        # 5. Apply structured filters
        if l1_domain_filter:
            wanted = set(l1_domain_filter)
            hits = [
                h for h in hits
                if h.l1_domains and any(d in h.l1_domains for d in wanted)
            ]
        if require_coverage_at:
            hits = self._filter_by_coverage(hits, require_coverage_at)

        # 6. Apply commonly_queried boost (variable hits only)
        if boost_commonly_queried:
            for h in hits:
                if h.target_variable_id and h.is_commonly_queried:
                    h.weighted_score *= 1.10

        # 7. Aggregate per target
        table_targets, var_targets = self._aggregate(hits)

        # 7b. Table-family bias — demote profile / PR variants so they
        #     can't outrank canonical detailed tables for common queries.
        if self.config.table_family_bias:
            _apply_table_family_bias(
                table_targets, self.config.table_family_weights,
            )
            _apply_table_family_bias(
                var_targets, self.config.table_family_weights,
            )

        # 8. Sort + cap
        top_tables = sorted(
            table_targets.values(), key=lambda t: -t.aggregate_score,
        )[:top_k]
        top_variables = sorted(
            var_targets.values(), key=lambda t: -t.aggregate_score,
        )[:top_k]

        # 9. Drop targets that fell below floor
        top_tables = [
            t for t in top_tables
            if t.aggregate_score >= self.config.min_aggregate_score
        ]
        top_variables = [
            t for t in top_variables
            if t.aggregate_score >= self.config.min_aggregate_score
        ]

        return RoutedResult(
            query=query,
            top_tables=top_tables,
            top_variables=top_variables,
            raw_hits=hits[:50],            # cap raw evidence for response size
            n_hits_retrieved=n_total,
            n_hits_after_filter=len(hits),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hydrate(
        self, faiss_ids: np.ndarray, faiss_distances: np.ndarray,
    ) -> list[RoutedHit]:
        """JOIN the cards table to get per-card metadata + compute weighted
        score. ``faiss_ids`` may contain -1 for empty slots."""
        valid = [
            (int(rid), float(dist))
            for rid, dist in zip(faiss_ids, faiss_distances)
            if rid >= 0
        ]
        if not valid:
            return []
        rowids = [v[0] for v in valid]
        # Bulk fetch metadata. The post-swap schema adds per-card
        # years_available / geo_levels_available JSON arrays so we can
        # filter by coverage without hitting the coverage table.
        placeholders = ",".join("?" * len(rowids))
        rows = self.metadata_db.execute(
            f"SELECT rowid, card_type, text, target_table_id, target_dataset, "
            f"       target_year, target_variable_id, weight, polarity, "
            f"       measure_type, subject_area, l1_domains, "
            f"       is_commonly_queried, is_total, depth, "
            f"       years_available, geo_levels_available "
            f"FROM cards WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        meta_by_rowid = {int(r["rowid"]): r for r in rows}

        # FAISS returns L2 distances for our HNSW SQ index. Convert to
        # cosine-like similarity: cosine = 1 - L2² / 2 (only valid for
        # unit-normalized vectors, which Vertex embeddings approximately
        # are). We treat the L2 distance directly as a relevance signal:
        # smaller = better. To make the rest of the code think in
        # similarity space, flip sign so larger = better.
        hits: list[RoutedHit] = []
        for rid, dist in valid:
            meta = meta_by_rowid.get(rid)
            if meta is None:
                continue
            similarity = max(0.0, 1.0 - dist / 2.0)   # ≈ cosine for unit vecs
            weight = float(meta["weight"])
            polarity = meta["polarity"]
            weighted = similarity * weight
            if polarity == "negative":
                weighted = -weighted
            # target_year is NULL on post-swap question cards; keep it
            # optional rather than coerce to 0.
            raw_year = meta["target_year"]
            target_year = int(raw_year) if raw_year is not None else None
            # Parse JSON coverage arrays; default to empty list on
            # missing / malformed values.
            years_available = _parse_json_int_list(
                meta["years_available"] if "years_available" in meta.keys()
                else None
            )
            geo_levels_available = _parse_json_str_list(
                meta["geo_levels_available"]
                if "geo_levels_available" in meta.keys() else None
            )
            hits.append(RoutedHit(
                rowid=rid,
                card_type=meta["card_type"],
                text=meta["text"],
                target_table_id=meta["target_table_id"],
                target_dataset=meta["target_dataset"],
                target_year=target_year,
                target_variable_id=meta["target_variable_id"],
                cosine=similarity,
                weight=weight,
                polarity=polarity,
                weighted_score=weighted,
                measure_type=meta["measure_type"],
                subject_area=meta["subject_area"],
                l1_domains=meta["l1_domains"],
                is_commonly_queried=meta["is_commonly_queried"],
                is_total=meta["is_total"],
                depth=meta["depth"],
                years_available=years_available,
                geo_levels_available=geo_levels_available,
            ))
        return hits

    def _filter_by_coverage(
        self, hits: list[RoutedHit], required_levels: list[str],
    ) -> list[RoutedHit]:
        """Keep only hits whose card advertises every required geo level
        via its ``geo_levels_available`` JSON array.

        Post-Phase-2.5 swap, question cards carry per-card coverage
        metadata inline (years_available + geo_levels_available), so we
        no longer join the ``coverage`` table at retrieval time — the
        filter becomes an in-Python set containment check.

        Fallback: if a hit has an empty ``geo_levels_available`` (older
        card type without coverage metadata, or suppressed during
        import), we pass it through rather than drop it. The planner's
        APIPlanCall construction still catches truly-broken targets
        when the Census API returns 404.
        """
        if not hits or not required_levels:
            return list(hits)
        required_set = set(required_levels)
        out: list[RoutedHit] = []
        for h in hits:
            if not h.geo_levels_available:
                # No declared coverage — don't silently drop; let the
                # planner discover unsupported combos at fetch time.
                out.append(h)
                continue
            if required_set.issubset(set(h.geo_levels_available)):
                out.append(h)
        return out

    def _aggregate(
        self, hits: list[RoutedHit],
    ) -> tuple[dict, dict]:
        """Group hits by table-level and variable-level targets; sum
        weighted scores within each group.

        Keys are ``(table_id, dataset)`` for table targets and
        ``(table_id, dataset, variable_id)`` for variable targets —
        year is no longer part of the key because post-swap cards are
        year-agnostic. The merged ``years_available`` /
        ``geo_levels_available`` unions across contributing hits tell
        the planner which specific year + geo_level combinations are
        actually live.
        """
        table_acc: dict[tuple, dict] = defaultdict(lambda: {
            "score": 0.0, "n_pos": 0, "n_neg": 0,
            "best_hit": None, "card_types": set(),
            "years": set(), "levels": set(),
        })
        var_acc: dict[tuple, dict] = defaultdict(lambda: {
            "score": 0.0, "n_pos": 0, "n_neg": 0,
            "best_hit": None, "card_types": set(),
            "years": set(), "levels": set(),
        })

        for h in hits:
            t_key = (h.target_table_id, h.target_dataset)
            ta = table_acc[t_key]
            ta["score"] += h.weighted_score
            ta["card_types"].add(h.card_type)
            ta["years"].update(h.years_available)
            ta["levels"].update(h.geo_levels_available)
            if h.polarity == "positive":
                ta["n_pos"] += 1
            else:
                ta["n_neg"] += 1
            if ta["best_hit"] is None or h.cosine > ta["best_hit"].cosine:
                ta["best_hit"] = h

            if h.target_variable_id:
                v_key = t_key + (h.target_variable_id,)
                va = var_acc[v_key]
                va["score"] += h.weighted_score
                va["card_types"].add(h.card_type)
                va["years"].update(h.years_available)
                va["levels"].update(h.geo_levels_available)
                if h.polarity == "positive":
                    va["n_pos"] += 1
                else:
                    va["n_neg"] += 1
                if va["best_hit"] is None or h.cosine > va["best_hit"].cosine:
                    va["best_hit"] = h

        def _target(k, v, is_var: bool) -> RoutedTarget:
            return RoutedTarget(
                target_table_id=k[0], target_dataset=k[1],
                target_year=None,
                target_variable_id=k[2] if is_var else None,
                aggregate_score=v["score"],
                n_positive_hits=v["n_pos"],
                n_negative_hits=v["n_neg"],
                best_hit=v["best_hit"],
                matched_card_types=sorted(v["card_types"]),
                years_available=sorted(v["years"]),
                geo_levels_available=sorted(v["levels"]),
            )

        table_targets = {
            k: _target(k, v, is_var=False)
            for k, v in table_acc.items() if v["best_hit"] is not None
        }
        var_targets = {
            k: _target(k, v, is_var=True)
            for k, v in var_acc.items() if v["best_hit"] is not None
        }
        return table_targets, var_targets
