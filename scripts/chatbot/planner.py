"""Phase H planner: ExtractedIntent + ResolvedGeographies → APIPlanCalls.

For each (resolved_geo, concept) pair the planner runs the tier cascade
to find a Census table+variable set, then constructs the right API
calls for the geo's level + the requested temporal scope.

Tier cascade per concept:
  - 0.5  composite-concept decomposition cache (DecompositionCache).
         Cache hit → recurse on each sub-concept. Cache miss on a
         composite → emit a placeholder telling the orchestrator
         (Phase I) to call the LLM decomposer.
  - 1    Tier-1 concept_map (curated). Lookup canonical_hint, then
         verbatim text. Cheapest path.
  - 2    FTS metadata search. Acceptance threshold lets weak matches
         fall through to Tier 3.
  - 3    LLM fallback. Stubbed here — orchestrator wires the LLM call
         when this tier surfaces.

Year selection:
  - latest: latest year supported by the effective (table, dataset)
            at every required geo_level (uses ``find_supported_years``).
  - change: intent.years[0] and intent.years[-1] if explicitly given;
            else (latest, latest-3) as a default.
  - trend:  every supported year from intent.years[0] (or latest-4)
            up to and including latest. Capped at 5 vintages.

API call construction:
  - For tract-aggregating geos (neighborhoods, corridors, regions,
    landmark_areas, and compound-derived results): emit one call per
    unique (state, county) prefix found in ``tract_geoids``. Aggregator
    filters the response back to the original tract list.
  - For pure admin geos (county/state/place/msa/us): use the resolver-
    supplied ``api_for_clause`` + ``api_in_clause`` directly.
  - Each call inherits the shared TTL from chatbot.yaml: 30d for ACS5,
    7d for ACS1, 1d otherwise.

The planner does NOT make any HTTP calls or read data — it produces the
plan only. The orchestrator (Phase I) wires planner → CensusCaller →
aggregator together.

Public API:
    PlannedCall                    — one (call, geo, concept, role) bundle
    ConceptResolution              — tier-cascade outcome for one concept
    PlanResult                     — the planner's full output
    plan_query(intent, resolved, cmap, metadata_db, decomp_cache)
                                   — entry point
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from typing import Literal, Optional
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.census_caller import APIPlanCall
from scripts.chatbot.concept_map import (
    ConceptEntry, ConceptMap, ConceptVariables, ResolvedConcept,
)
from scripts.chatbot.decomposition_cache import (
    Decomposition, DecompositionCache,
)
from scripts.chatbot.metadata_search import (
    TableCandidate, find_supported_years, search_tables,
)
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedIntent, ResolvedGeography,
)
# Semantic router is the K.5 replacement for the Tier 1/2/3 cascade.
# Imported lazily inside route_concept_semantic to avoid faiss as a hard
# dependency for unit tests that exercise the legacy path.
if False:                                       # pragma: no cover
    from scripts.chatbot.semantic_router import RoutedResult, SemanticRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / config defaults
# ---------------------------------------------------------------------------

# How many vintages a 'trend' query expands to before we cap.
_MAX_TREND_VINTAGES = 5

# FTS5 BM25 cutoff for accepting a Tier-2 hit. Lower = better match;
# anything worse than this triggers Tier 3 fallback.
_TIER2_BM25_CEILING = -3.0

# TTLs (seconds) by dataset family. Mirrors chatbot.yaml's intent — ACS5
# vintages are immutable once published; ACS1 refreshes faster.
_TTL_ACS5 = 30 * 24 * 60 * 60
_TTL_ACS1 = 7 * 24 * 60 * 60
_TTL_OTHER = 1 * 24 * 60 * 60

# Dataset-level identifier / predicate variables that must NEVER be used as
# a denominator. Stage3's auto-pairing occasionally emits these; if one
# slips through as denominator the aggregator ends up dividing by a non-
# numeric string and silently drops every row. Belt-and-suspenders with
# the cleanup in the metadata DB.
_INVALID_DENOMINATORS = frozenset({
    "GEO_ID", "NAME", "STATE", "COUNTY", "SUMLEVEL", "GEOCOMP",
    "GEO_ID.annotation", "POPGROUP", "YEAR", "SUBJECT", "VINTAGE",
    "FIRMPDEMP_S", "VAL", "NAICS2017", "NAICS2017_LABEL",
})


def _ttl_for_dataset(dataset: str) -> int:
    if dataset.startswith("acs/acs5"):
        return _TTL_ACS5
    if dataset.startswith("acs/acs1"):
        return _TTL_ACS1
    return _TTL_OTHER


# Maps ResolvedGeography.geo_level → the Census-API geo_level we'd query.
# Aggregating levels (neighborhood, corridor, region, landmark_area)
# all roll up tract data.
_GEO_LEVEL_TO_DATA_LEVEL = {
    "tract": "tract",
    "neighborhood": "tract",
    "region": "tract",
    "corridor": "tract",
    "landmark_area": "tract",
    "county": "county",
    "place": "place",
    "msa": "msa",
    "state": "state",
    "us": "us",
    "school_district_unified": "school_district_unified",
    "zcta": "zcta",
}


def data_level_for(geo: ResolvedGeography) -> str:
    """Census-API geo_level the planner should request for this resolution.

    A compound result (intersect/directional/difference over an admin
    primary) keeps the primary's ``geo_level`` (often 'county' or
    'place') but carries a filtered ``tract_geoids`` list. Those must
    be queried at tract level — the API call has to fetch tracts and
    aggregator filters to the list. We detect that by looking for
    ``tract_geoids`` on a non-tract geo_level.
    """
    if geo.tract_geoids and geo.geo_level not in ("tract",):
        return "tract"
    return _GEO_LEVEL_TO_DATA_LEVEL.get(geo.geo_level, geo.geo_level)


# ---------------------------------------------------------------------------
# Pydantic outputs
# ---------------------------------------------------------------------------

ResolutionTier = Literal[
    "tier_0_5_decomposition",
    "tier_1_concept_map",
    "tier_2_fts",
    "tier_3_llm_fallback",
    "needs_llm_decomposition",
    "unresolved",
    # K.5 — single semantic-routing path replacing the Tier 1/2/3 cascade
    # when a SemanticRouter is provided to plan_query.
    "tier_semantic",
]


class ConceptResolution(BaseModel):
    """Outcome of running one ExtractedConcept through the tier cascade."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    concept: ExtractedConcept
    tier: ResolutionTier

    # Set when tier == 'tier_1_concept_map'
    concept_entry: Optional[ConceptEntry] = None

    # Set when tier == 'tier_2_fts'. The table the planner selected; the
    # aggregator pulls the canonical _001E (or first non-GEO) for value.
    table_candidate: Optional[TableCandidate] = None

    # Set when tier == 'tier_0_5_decomposition'. Sub-concepts each get
    # their own ConceptResolution recursively.
    decomposition: Optional[Decomposition] = None
    sub_resolutions: list["ConceptResolution"] = Field(default_factory=list)

    # Set when tier == 'tier_semantic'. The router result is what the
    # composer uses to pick the (table, dataset, variables) triple.
    # Typed loosely as `dict | None` here to avoid pulling faiss into
    # Pydantic's schema for the legacy-path tests; the orchestrator and
    # downstream code treat it as a RoutedResult.
    routed_result: Optional[object] = None

    # Set when tier == 'tier_semantic'. The ConceptRewrite that drove
    # retrieval. Carried so the composer can use IntentFacets
    # (measure_type_sought, is_total_likely) to disambiguate sibling
    # variables in the same table.
    rewrite: Optional[object] = None

    notes: list[str] = Field(default_factory=list)


class PlannedCall(BaseModel):
    """One API call together with the (geo, concept, year, role) it serves.

    ``role`` is 'primary' for direct concept calls, 'sub_<i>' for
    decomposition sub-concept calls.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    api_call: APIPlanCall
    geo_idx: int = Field(
        ..., description="Index into PlanResult.resolved_geos this call serves.",
    )
    concept_idx: int = Field(
        ..., description="Index into PlanResult.intent.concepts this call serves.",
    )
    year: int
    role: str = "primary"
    # Effective (after by_geo_level overrides) variable layout, so the
    # aggregator knows whether to take a value, ratio, or components.
    variables: ConceptVariables
    # tract_geoids the aggregator should restrict the response to —
    # empty when the geo is pure admin.
    tract_filter: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    """Top-level planner output."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    intent: ExtractedIntent
    resolved_geos: list[ResolvedGeography]
    concept_resolutions: list[ConceptResolution]
    calls: list[PlannedCall]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tier cascade
# ---------------------------------------------------------------------------

def _tier1_lookup(
    cmap: ConceptMap, concept: ExtractedConcept,
) -> tuple[Optional[ConceptEntry], Optional[str]]:
    """Try canonical_hint then verbatim text; return (entry, key_used)."""
    if concept.canonical_hint:
        hit = cmap.lookup(concept.canonical_hint)
        if hit is not None:
            return hit, concept.canonical_hint
    hit = cmap.lookup(concept.text)
    if hit is not None:
        return hit, concept.text
    return None, None


def _tier2_lookup(
    metadata_db: sqlite3.Connection,
    concept: ExtractedConcept,
    data_levels: list[str],
) -> Optional[TableCandidate]:
    """FTS metadata search; only returns if BM25 clears _TIER2_BM25_CEILING."""
    candidates = search_tables(
        metadata_db, concept.text,
        require_coverage_at=data_levels if data_levels else None,
        top_k=5,
    )
    if not candidates:
        return None
    top = candidates[0]
    if top.bm25_score > _TIER2_BM25_CEILING:
        return None
    return top


# ---------------------------------------------------------------------------
# K.5 — semantic routing path
# ---------------------------------------------------------------------------

def route_concept_semantic(
    concept: ExtractedConcept,
    semantic_router: object,                     # SemanticRouter
    data_levels: list[str],
    *,
    top_k: int = 10,
) -> ConceptResolution:
    """K.5 replacement for the tier cascade.

    One vector retrieval pass against the embedding index. The
    canonical_hint (when set) is preferred as the query text since it's
    the LLM's normalized concept name; fall back to the verbatim user
    span otherwise.
    """
    text = (concept.canonical_hint or concept.text or "").strip()
    if not text:
        return ConceptResolution(
            concept=concept, tier="unresolved",
            notes=["empty concept text — cannot route"],
        )
    routed = semantic_router.route(
        text, top_k=top_k,
        require_coverage_at=data_levels if data_levels else None,
    )
    if not routed.top_tables and not routed.top_variables:
        return ConceptResolution(
            concept=concept, tier="unresolved",
            routed_result=routed,
            notes=[
                f"semantic router returned no candidates with coverage at "
                f"{data_levels}"
            ],
        )
    notes = []
    if routed.top_variables:
        tv = routed.top_variables[0]
        notes.append(
            f"top variable {tv.target_dataset}/{tv.target_table_id}/"
            f"{tv.target_variable_id} score={tv.aggregate_score:.2f}"
        )
    elif routed.top_tables:
        tt = routed.top_tables[0]
        notes.append(
            f"table-only hit {tt.target_dataset}/{tt.target_table_id} "
            f"score={tt.aggregate_score:.2f}"
        )
    return ConceptResolution(
        concept=concept,
        tier="tier_semantic",
        routed_result=routed,
        notes=notes,
    )


# Tokens in a variable's natural-language label that mark it as
# restricted to a specific demographic / household / nativity / health
# subgroup. The composer de-prioritizes such variables unless the
# rewriter flagged the concept as a disparity-asking query AND we are
# explicitly fanning out (J.5 — deferred). For an unmarked rewrite we
# want the OVERALL universe variable, not a subgroup subcell.
#
# Census S-series labels encode category restrictions with an
# uppercase category name (DISABILITY STATUS, RACE AND HISPANIC OR
# LATINO ORIGIN, SEX, AGE OF HOUSEHOLDER, ...) followed by a specific
# value (With a disability / Black alone / Male / 25 to 44 years).
# Detecting the category marker is a more robust signal than trying
# to enumerate every possible subgroup value.
_SUBGROUP_CATEGORY_MARKERS = (
    "race and hispanic", "hispanic or latino",
    "disability status", "sex", " by age ", "age of householder",
    "nativity", "citizenship", "country of birth",
    "language spoken", "limited english",
    "household type", "family type",
    "educational attainment", "school enrollment",
    "nativity and citizenship",
)
_SUBGROUP_VALUE_TOKENS = (
    "white alone", "black alone", "asian alone", "hispanic",
    "white non-hispanic", "non-hispanic white",
    "black or african american alone",
    "native hawaiian", "american indian",
    "two or more races", "some other race alone",
    "non-hispanic", "latino", "non-latino",
    "with a disability", "without a disability",
    "with any disability",
    "foreign born", "foreign-born", "native born", "native-born",
    "limited english speaking",
)


def _is_subgroup_restricted(label: Optional[str]) -> bool:
    if not label:
        return False
    text = label.lower()
    for marker in _SUBGROUP_CATEGORY_MARKERS:
        if marker in text:
            return True
    for val in _SUBGROUP_VALUE_TOKENS:
        if val in text:
            return True
    return False


def _measure_type_score(
    label: Optional[str], measure_type_sought: Optional[str],
) -> int:
    """Bonus/penalty when the variable's label matches the user's
    desired measure type. Positive = match, negative = mismatch."""
    if not label or not measure_type_sought:
        return 0
    text = label.lower()
    if measure_type_sought == "percentage":
        if any(t in text for t in ("percent", "share", " rate")):
            return 3
        if text.startswith("percent"):
            return 4
        # Count-like labels are a strong mismatch for percentage queries.
        if any(text.startswith(t) for t in (
            "total ", "number ", "count ", "aggregate ",
        )):
            return -3
    elif measure_type_sought == "count":
        if any(t in text for t in (
            "total ", "number ", "count ", "aggregate ",
        )):
            return 2
        if text.startswith("percent"):
            return -3
    elif measure_type_sought == "currency":
        if "median" in text or "mean" in text or "aggregate" in text:
            return 3
    elif measure_type_sought == "median":
        if "median" in text:
            return 4
    return 0


def _representative_year(target) -> Optional[int]:
    """Pick a year for label / denominator-pair lookups against the v1
    stage outputs (which index by year).

    Post-Phase-2.5 schema: ``RoutedTarget.target_year`` is typically
    NULL because question cards are year-agnostic. Fall back to the
    latest year advertised in ``years_available``. Returns None when
    neither is set — caller should skip the lookup rather than key the
    join with NULL.
    """
    if target is None:
        return None
    if getattr(target, "target_year", None) is not None:
        return int(target.target_year)
    years = getattr(target, "years_available", None) or []
    if not years:
        return None
    try:
        return max(int(y) for y in years)
    except (TypeError, ValueError):
        return None


def _lookup_label(
    metadata_db: sqlite3.Connection,
    table_id: str, dataset: str, year: int, variable_id: str,
) -> Optional[str]:
    row = metadata_db.execute(
        "SELECT label_natural, label FROM variables "
        "WHERE table_id = ? AND dataset = ? AND year = ? "
        "  AND variable_id = ?",
        (table_id, dataset, year, variable_id),
    ).fetchone()
    if not row:
        return None
    return row["label_natural"] or row["label"]


class _VarCandidate:
    """Uniform candidate record for composer scoring. Populated from
    router top_variables OR from a direct top-table variable listing."""
    __slots__ = (
        "table_id", "dataset", "year", "variable_id",
        "label", "is_total", "depth", "is_commonly_queried",
        "router_rank", "router_score",
    )

    def __init__(
        self, *, table_id: str, dataset: str, year: int,
        variable_id: str, label: Optional[str],
        is_total: bool, depth: Optional[int],
        is_commonly_queried: bool,
        router_rank: Optional[int] = None,
        router_score: float = 0.0,
    ):
        self.table_id = table_id
        self.dataset = dataset
        self.year = year
        self.variable_id = variable_id
        self.label = label
        self.is_total = is_total
        self.depth = depth
        self.is_commonly_queried = is_commonly_queried
        self.router_rank = router_rank
        self.router_score = router_score


def _list_top_table_variables(
    metadata_db: sqlite3.Connection, top_table_key: tuple[str, str, int],
) -> list[_VarCandidate]:
    """All estimate-kind data variables in the top-ranked table.

    Ensures canonical-universe variables (e.g., S2201_C04_001E) end
    up in the candidate set even when the router's card-based
    retrieval happens to miss them (typically because v1's LLM-
    generated var_question cards for those variables used
    place-specific phrasings that don't match generic queries).

    Excludes ``_INVALID_DENOMINATORS`` (GEO_ID, NAME, STATE, ...) —
    those are dataset-level geo/predicate identifiers that have
    sometimes been miscategorized as ``kind='estimate'`` during v1
    stage7 import. Including them lets the depth-preference bonus
    (these identifiers have depth=1) win over real variables.
    """
    table_id, dataset, year = top_table_key
    rows = metadata_db.execute(
        "SELECT variable_id, label_natural, label, is_total, depth, "
        "       is_commonly_queried "
        "FROM variables "
        "WHERE table_id = ? AND dataset = ? AND year = ? "
        "  AND kind = 'estimate'",
        (table_id, dataset, year),
    ).fetchall()
    out: list[_VarCandidate] = []
    for r in rows:
        if r["variable_id"] in _INVALID_DENOMINATORS:
            continue
        out.append(_VarCandidate(
            table_id=table_id, dataset=dataset, year=year,
            variable_id=r["variable_id"],
            label=r["label_natural"] or r["label"],
            is_total=bool(r["is_total"]),
            depth=r["depth"],
            is_commonly_queried=bool(r["is_commonly_queried"]),
            router_rank=None,
            router_score=0.0,
        ))
    return out


# Concepts that name a subgroup of a larger universe — for these, the
# planner should prefer the subgroup's count/rate variable over the
# universe _001E. Without this guard the composer's +5.0 top-table bonus
# plus the depth=0 / is_total=1 advantages combine to make the universe
# variable beat the correct subgroup variable on tables that contain
# both (e.g. B21001 → _001E "civilian pop 18+" wins over _002E
# "veterans" for a "veterans in Cobb" query).
_SUBGROUP_CONCEPT_KEYWORDS = (
    "veteran", "disab", "below poverty", "in poverty", "poverty rate",
    "child poverty", "snap", "food stamp", "public assistance",
    "renter", "owner-occupied", "cost burden", "rent burden",
    "uninsured", "foreign-born", "foreign born", "native-born",
    "native born", "hispanic", "latino", "black", "asian",
    "american indian", "native hawaiian", "two or more races",
    "elderly", "senior", "over 65", "65 and over", "65+",
    "under 18", "under 5", "youth", "children",
    "single mother", "single father", "single parent",
    "grandparent", "limited english", "lep ",
    "unemploy", "labor force partic", "commute",
    "transit", "no vehicle", "zero vehicle",
    "no internet", "broadband", "computer ownership",
    # Educational-attainment subgroups
    "no bachelor", "no hs diploma", "no high school",
    "without a bachelor", "without a college",
    "without a degree", "without hs", "without high school",
)


def _concept_is_subgroup(concept_text: Optional[str]) -> bool:
    """True when the concept asks for a subgroup of a broader universe.

    Subgroup concepts (veterans, renters, children in poverty, SNAP
    households, uninsured people, Hispanic residents, etc.) need to
    resolve to the *subgroup* variable, not the *universe* _001E. This
    detector is deliberately keyword-based and conservative: false
    negatives are fine (they fall back to the existing behavior); false
    positives would wrongly demote the universe variable on genuine
    total-population queries.
    """
    if not concept_text:
        return False
    t = concept_text.lower()
    return any(kw in t for kw in _SUBGROUP_CONCEPT_KEYWORDS)


def _score_candidate(
    c: _VarCandidate,
    *,
    top_table_key: Optional[tuple[str, str, int]],
    measure_type_sought: Optional[str],
    is_total_likely: Optional[bool],
    concept_is_subgroup: bool = False,
) -> float:
    score = 0.0
    if top_table_key and (
        c.table_id, c.dataset, c.year,
    ) == top_table_key:
        score += 5.0
    # Router-rank bonus only when the router actually retrieved this
    # variable; table-listing candidates get a neutral baseline.
    if c.router_rank is not None:
        score += max(0, 10 - c.router_rank) * 0.3
        score += min(c.router_score, 5.0) * 0.2
    if _is_subgroup_restricted(c.label):
        score -= 8.0
    score += _measure_type_score(c.label, measure_type_sought)
    if c.depth is not None:
        score -= max(0, c.depth - 1) * 3.0
    if c.is_commonly_queried:
        score += 0.5
    if is_total_likely is not None:
        if is_total_likely and c.is_total:
            score += 2.0
        elif is_total_likely and not c.is_total:
            score -= 1.0
        elif not is_total_likely and c.is_total:
            score -= 0.5
    # Subgroup-concept correction: when the query is about a subgroup
    # (veterans, renters, children in poverty, uninsured, etc.), the
    # universe's _001E is_total variable is the WRONG answer — it gives
    # the population-for-whom-status-was-determined count, not the
    # subgroup count. Apply a hard penalty so the subgroup variable
    # wins, even when _001E is at depth 0 in the top table.
    if concept_is_subgroup and c.is_total:
        score -= 6.0
    return score


def _compose_from_router(
    routed: object,                              # RoutedResult
    metadata_db: sqlite3.Connection,
    data_level: str,
    *,
    measure_type_sought: Optional[str] = None,
    is_total_likely: Optional[bool] = None,
    concept_text: Optional[str] = None,
) -> Optional[tuple[str, str, ConceptVariables]]:
    """Pick the best (table, dataset, ConceptVariables) for one geo from
    router output.

    Candidate set:
      (a) Variables from ``routed.top_variables`` — what the router
          actually retrieved based on card matching.
      (b) **All** estimate variables in the top-ranked table, even
          those the router didn't surface. This catches universe /
          overall variables (e.g., S2201_C04_001E) whose LLM-generated
          cards happened to use place-specific phrasings that
          mismatch generic queries.

    Each candidate is scored across:
      * Table match with the top table
      * Router rank and aggregate score (only when (a))
      * Subgroup penalty (race/disability/sex/etc. restrictions)
      * Measure-type label match (percentage vs count vs currency)
      * Depth preference (shallower = more general)
      * is_commonly_queried bonus
      * is_total alignment with the rewrite

    After picking the winning variable, stage3's ``denominator_pairs``
    decides whether it becomes ``value`` (is_total=1) or a
    ``numerator+denominator`` ratio.
    """
    top_table = routed.top_tables[0] if routed.top_tables else None
    top_year = _representative_year(top_table) if top_table else None
    top_key = (
        (top_table.target_table_id, top_table.target_dataset, top_year)
        if top_table and top_year is not None else None
    )

    # (a) Router-hit candidates
    candidates_by_key: dict[tuple[str, str, int, str], _VarCandidate] = {}
    for rank, v in enumerate(routed.top_variables):
        if not v.target_variable_id:
            continue
        # Post-Phase-2.5: RoutedTarget.target_year is typically None for
        # question-derived targets. Pick a representative year from the
        # target's years_available (latest) for label-lookup and
        # denominator-pair joins — which are keyed by year in the v1
        # stage outputs but stable across vintages. When no year is
        # available at all, skip the candidate rather than inserting a
        # NULL-keyed row.
        rep_year = _representative_year(v)
        if rep_year is None:
            continue
        key = (v.target_table_id, v.target_dataset, rep_year,
               v.target_variable_id)
        if key in candidates_by_key:
            # Same variable appeared at multiple ranks (usually one
            # per year bucket); keep the best rank/score.
            existing = candidates_by_key[key]
            if existing.router_rank is None or rank < existing.router_rank:
                existing.router_rank = rank
                existing.router_score = max(
                    existing.router_score, float(v.aggregate_score),
                )
            continue
        label = _lookup_label(
            metadata_db, v.target_table_id, v.target_dataset,
            rep_year, v.target_variable_id,
        )
        candidates_by_key[key] = _VarCandidate(
            table_id=v.target_table_id, dataset=v.target_dataset,
            year=rep_year, variable_id=v.target_variable_id,
            label=label,
            is_total=bool(getattr(v.best_hit, "is_total", 0)),
            depth=getattr(v.best_hit, "depth", None),
            is_commonly_queried=bool(
                getattr(v.best_hit, "is_commonly_queried", 0)
            ),
            router_rank=rank,
            router_score=float(v.aggregate_score),
        )

    # (b) Augment with top-table variables the router may have missed
    if top_key is not None:
        for c in _list_top_table_variables(metadata_db, top_key):
            key = (c.table_id, c.dataset, c.year, c.variable_id)
            if key not in candidates_by_key:
                candidates_by_key[key] = c

    if not candidates_by_key:
        if top_table is not None:
            return (
                top_table.target_table_id, top_table.target_dataset,
                ConceptVariables(value=f"{top_table.target_table_id}_001E"),
            )
        return None

    concept_is_subgroup = _concept_is_subgroup(concept_text)
    scored = [
        (_score_candidate(
            c, top_table_key=top_key,
            measure_type_sought=measure_type_sought,
            is_total_likely=is_total_likely,
            concept_is_subgroup=concept_is_subgroup,
        ), i, c)
        for i, c in enumerate(candidates_by_key.values())
    ]
    scored.sort(key=lambda s: (-s[0], s[1]))
    _, _, winner = scored[0]

    if winner.is_total:
        return winner.table_id, winner.dataset, ConceptVariables(
            value=winner.variable_id,
        )

    pair = metadata_db.execute(
        "SELECT denominator_var FROM denominator_pairs "
        "WHERE table_id = ? AND dataset = ? AND year = ? "
        "  AND numerator_var = ?",
        (winner.table_id, winner.dataset, winner.year,
         winner.variable_id),
    ).fetchone()
    if pair and pair[0] and pair[0] not in _INVALID_DENOMINATORS:
        return winner.table_id, winner.dataset, ConceptVariables(
            numerator=winner.variable_id, denominator=pair[0],
        )
    return winner.table_id, winner.dataset, ConceptVariables(
        value=winner.variable_id,
    )


def _calls_for_semantic_resolution(
    geo: ResolvedGeography,
    geo_idx: int,
    concept_idx: int,
    resolution: ConceptResolution,
    intent: ExtractedIntent,
    metadata_db: sqlite3.Connection,
    role: str = "primary",
    universe_picker: Optional[object] = None,
) -> list[PlannedCall]:
    """K.5 analog of _calls_for_resolved_concept for the semantic tier.

    When ``universe_picker`` is provided, it runs first and its result
    (if any) is used directly. The legacy ``_compose_from_router``
    remains the fallback path — this keeps the universe picker
    feature-flag-controllable and lets us A/B against the old logic
    without deleting it.
    """
    import time as _time
    _t_fn_start = _time.time()
    if resolution.tier != "tier_semantic":
        return []
    if resolution.routed_result is None:
        return []
    data_level = data_level_for(geo)
    rewrite = resolution.rewrite
    concept_text = (
        resolution.concept.text if resolution.concept else None
    )

    composed = None
    _t_pick = _time.time()
    if universe_picker is not None and concept_text:
        try:
            composed = universe_picker.pick(
                query_text=concept_text,
                routed=resolution.routed_result,
                data_level=data_level,
            )
        except Exception as e:                         # pragma: no cover
            logger.warning(
                "universe picker failed on concept=%r: %s — falling back",
                concept_text, e,
            )
            composed = None
    _t_pick_elapsed = _time.time() - _t_pick

    if composed is None:
        composed = _compose_from_router(
            resolution.routed_result, metadata_db, data_level,
            measure_type_sought=(
                getattr(rewrite, "measure_type_sought", None)
                if rewrite else None
            ),
            is_total_likely=(
                getattr(rewrite, "is_total_likely", None)
                if rewrite else None
            ),
            concept_text=concept_text,
        )
    if composed is None:
        return []
    table_id, dataset, variables_obj = composed

    _t_years = _time.time()
    supported = find_supported_years(
        metadata_db, table_id, dataset, [data_level],
    )
    _t_years_elapsed = _time.time() - _t_years
    years = _pick_years(intent, supported)
    if not years:
        return []

    _t_build = _time.time()
    var_ids = variables_obj.all_variable_ids()
    out: list[PlannedCall] = []
    for year in years:
        if data_level == "tract":
            for call, tract_filter in _apicalls_for_tract_geo(
                geo=geo, year=year, dataset=dataset,
                table_id=table_id, variables=var_ids,
            ):
                out.append(PlannedCall(
                    api_call=call, geo_idx=geo_idx, concept_idx=concept_idx,
                    year=year, role=role,
                    variables=variables_obj, tract_filter=tract_filter,
                ))
        else:
            built = _apicall_for_admin_geo(
                geo=geo, data_level=data_level, year=year,
                dataset=dataset, table_id=table_id, variables=var_ids,
            )
            if built is None:
                continue
            call, _ = built
            out.append(PlannedCall(
                api_call=call, geo_idx=geo_idx, concept_idx=concept_idx,
                year=year, role=role,
                variables=variables_obj, tract_filter=[],
            ))
    _t_build_elapsed = _time.time() - _t_build
    _t_fn_elapsed = _time.time() - _t_fn_start
    if _t_fn_elapsed > 0.5:
        logger.warning(
            "[SLOW] _calls_for_semantic_resolution  geo=%s role=%s "
            "concept=%r  total=%.2fs  pick=%.2fs  years=%.2fs  "
            "build=%.2fs",
            geo.display_name, role, concept_text,
            _t_fn_elapsed, _t_pick_elapsed, _t_years_elapsed,
            _t_build_elapsed,
        )
    return out


def resolve_concept(
    concept: ExtractedConcept,
    cmap: ConceptMap,
    metadata_db: sqlite3.Connection,
    data_levels: list[str],
    *,
    decomp_cache: Optional[DecompositionCache] = None,
) -> ConceptResolution:
    """Run the full tier cascade for one concept.

    For composites: Tier 0.5 cache wins when present (richer
    decomposition). On cache miss we DON'T immediately demand LLM
    decomposition — first try Tier 1, since the LLM's composite hint
    sometimes fires on phrases that map cleanly to a single curated
    entry (e.g., "median income inequality" → Tier 1 'income
    inequality' = Gini index). Only when both the cache AND Tier 1
    miss do we surface needs_llm_decomposition.
    """
    notes: list[str] = []

    # Tier 0.5 — only relevant for composites; richer than Tier 1 when present.
    if concept.is_composite and decomp_cache is not None:
        cached = decomp_cache.get(concept.canonical_hint or concept.text)
        if cached is not None:
            sub_resolutions = [
                resolve_concept(
                    ExtractedConcept(text=s, canonical_hint=s),
                    cmap, metadata_db, data_levels,
                    decomp_cache=decomp_cache,
                )
                for s in cached.sub_concepts
            ]
            return ConceptResolution(
                concept=concept,
                tier="tier_0_5_decomposition",
                decomposition=cached,
                sub_resolutions=sub_resolutions,
                notes=notes,
            )

    # Tier 1 — curated concept_map. Tried for composites and singletons
    # alike; for composites, a Tier 1 hit is a graceful degradation
    # from the absent decomposition.
    entry, key = _tier1_lookup(cmap, concept)
    if entry is not None:
        notes.append(f"matched concept_map via {key!r}")
        if concept.is_composite:
            notes.append(
                "composite flagged but no decomposition available — "
                "using Tier 1 single-concept match as best-effort fallback"
            )
        return ConceptResolution(
            concept=concept,
            tier="tier_1_concept_map",
            concept_entry=entry,
            notes=notes,
        )

    # No Tier 1 hit. Composites with no cache and no Tier 1 match need
    # the LLM decomposer; non-composites continue to Tier 2.
    if concept.is_composite:
        if decomp_cache is None:
            notes.append("composite flagged but no decomp cache provided")
        else:
            notes.append("decomposition cache miss; LLM decomposition needed")
        return ConceptResolution(
            concept=concept, tier="needs_llm_decomposition", notes=notes,
        )

    # Tier 2 — FTS metadata search.
    candidate = _tier2_lookup(metadata_db, concept, data_levels)
    if candidate is not None:
        notes.append(
            f"FTS match: {candidate.dataset}/{candidate.year}/{candidate.table_id} "
            f"(bm25={candidate.bm25_score:.2f})"
        )
        return ConceptResolution(
            concept=concept,
            tier="tier_2_fts",
            table_candidate=candidate,
            notes=notes,
        )

    # Tier 3 — defer to orchestrator.
    notes.append("no Tier 1/2 match; LLM fallback required")
    return ConceptResolution(
        concept=concept, tier="tier_3_llm_fallback", notes=notes,
    )


# ---------------------------------------------------------------------------
# Year selection
# ---------------------------------------------------------------------------

def _pick_years(
    intent: ExtractedIntent,
    supported_years: list[int],
) -> list[int]:
    """Pick the year set the planner will request for one (concept, geo).

    ``supported_years`` is the descending list returned by
    ``find_supported_years`` for the effective (table, dataset) at the
    geo's data_level. Empty means we have nothing to plan against.
    """
    if not supported_years:
        return []
    latest = supported_years[0]

    if intent.temporal_intent == "latest":
        return [latest]

    if intent.temporal_intent == "change":
        if len(intent.years) >= 2:
            wanted = sorted({intent.years[0], intent.years[-1]})
        else:
            # Default: latest vs. ~3 years prior, clamped to availability.
            target = latest - 3
            prior = next((y for y in supported_years if y <= target), None)
            wanted = [latest] if prior is None else sorted({prior, latest})
        return [y for y in wanted if y in supported_years] or [latest]

    if intent.temporal_intent == "trend":
        start = intent.years[0] if intent.years else (latest - 4)
        seq = sorted(y for y in supported_years if start <= y <= latest)
        if not seq:
            return [latest]
        # Cap to the most-recent _MAX_TREND_VINTAGES years.
        return seq[-_MAX_TREND_VINTAGES:]

    return [latest]


# ---------------------------------------------------------------------------
# API URL + APIPlanCall construction
# ---------------------------------------------------------------------------

def _build_url(
    *, year: int, dataset: str, variables: list[str],
    api_for_clause: str, api_in_clause: str,
) -> str:
    """Construct the full Census API URL (no key)."""
    get_param = ",".join(["NAME", *variables])
    parts = [
        f"https://api.census.gov/data/{year}/{dataset}",
        f"?get={quote(get_param, safe=',_')}",
        f"&for={quote(api_for_clause, safe=':*')}",
    ]
    if api_in_clause:
        parts.append(f"&in={quote(api_in_clause, safe=':*+')}")
    return "".join(parts)


def _county_groups_from_tracts(tract_geoids: list[str]) -> dict[tuple[str, str], list[str]]:
    """Bucket tract GEOIDs by their (state_fips, county_fips) prefix.

    Census tract GEOID is SSCCCTTTTTT — first 5 chars uniquely identify
    the containing county. Used to issue one tract-level API call per
    county a multi-county neighborhood/region spans.
    """
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in tract_geoids:
        if len(t) >= 5:
            out[(t[:2], t[2:5])].append(t)
    return out


def _apicalls_for_tract_geo(
    *, geo: ResolvedGeography, year: int,
    dataset: str, table_id: str, variables: list[str],
) -> list[tuple[APIPlanCall, list[str]]]:
    """For tract-aggregating geos, emit one APIPlanCall per (state, county)
    prefix and return [(call, tracts_to_filter), ...]."""
    if not geo.tract_geoids:
        return []
    out: list[tuple[APIPlanCall, list[str]]] = []
    for (state, county), tracts in _county_groups_from_tracts(geo.tract_geoids).items():
        api_for = "tract:*"
        api_in = f"state:{state}+county:{county}"
        url = _build_url(
            year=year, dataset=dataset, variables=variables,
            api_for_clause=api_for, api_in_clause=api_in,
        )
        call = APIPlanCall(
            url=url,
            table_id=table_id,
            variables=variables,
            geo_level="tract",
            geo_filter_ids=tracts,
            year=year,
            dataset=dataset,
            ttl_seconds=_ttl_for_dataset(dataset),
        )
        out.append((call, tracts))
    return out


def _apicall_for_admin_geo(
    *, geo: ResolvedGeography, data_level: str, year: int,
    dataset: str, table_id: str, variables: list[str],
) -> Optional[tuple[APIPlanCall, list[str]]]:
    """For pure-admin geos, emit one APIPlanCall using the resolver-
    supplied for/in clauses. Returns (call, []) — no tract filtering."""
    if not geo.api_for_clause:
        return None
    url = _build_url(
        year=year, dataset=dataset, variables=variables,
        api_for_clause=geo.api_for_clause, api_in_clause=geo.api_in_clause,
    )
    call = APIPlanCall(
        url=url, table_id=table_id, variables=variables,
        geo_level=data_level,
        geo_filter_ids=[geo.geo_id],
        year=year, dataset=dataset,
        ttl_seconds=_ttl_for_dataset(dataset),
    )
    return call, []


# ---------------------------------------------------------------------------
# Per-(geo, concept) call construction
# ---------------------------------------------------------------------------

def _calls_for_resolved_concept(
    geo: ResolvedGeography,
    geo_idx: int,
    concept_idx: int,
    resolution: ConceptResolution,
    intent: ExtractedIntent,
    metadata_db: sqlite3.Connection,
    role: str = "primary",
    universe_picker: Optional[object] = None,
) -> list[PlannedCall]:
    """Translate one (geo, ConceptResolution) into PlannedCalls.

    For Tier 0.5 (decomposition), recurses into each sub-resolution and
    emits its calls with role='sub_<i>'. Returns an empty list for
    tiers that haven't found a concrete (table, dataset) yet
    (tier_3_llm_fallback, needs_llm_decomposition, unresolved).
    """
    data_level = data_level_for(geo)

    # K.5 — semantic-router resolutions take a different composer path.
    if resolution.tier == "tier_semantic":
        return _calls_for_semantic_resolution(
            geo, geo_idx, concept_idx, resolution, intent, metadata_db,
            role=role, universe_picker=universe_picker,
        )

    # Tier 0.5 → recurse on sub-resolutions, prepending sub_<i> roles.
    if resolution.tier == "tier_0_5_decomposition":
        out: list[PlannedCall] = []
        for i, sub in enumerate(resolution.sub_resolutions):
            sub_role = f"{role}.sub_{i}"
            out.extend(_calls_for_resolved_concept(
                geo, geo_idx, concept_idx, sub, intent, metadata_db,
                role=sub_role, universe_picker=universe_picker,
            ))
        return out

    # Determine effective (table, dataset, variables).
    if resolution.tier == "tier_1_concept_map":
        eff: ResolvedConcept = resolution.concept_entry.resolve_for(data_level)
        table_id = eff.table_id
        dataset = eff.dataset
        variables_obj = eff.variables
    elif resolution.tier == "tier_2_fts":
        cand = resolution.table_candidate
        table_id = cand.table_id
        dataset = cand.dataset
        # Default to the canonical _001E for value extraction; aggregator
        # treats this as a single-value pull. Phase I can refine.
        variables_obj = ConceptVariables(value=f"{table_id}_001E")
    else:
        # Nothing to call — this resolution needs an LLM step upstream.
        return []

    # Year selection — find supported years for this (table, dataset, level).
    supported = find_supported_years(
        metadata_db, table_id, dataset, [data_level],
    )
    years = _pick_years(intent, supported)
    if not years:
        return []

    var_ids = variables_obj.all_variable_ids()
    out: list[PlannedCall] = []
    for year in years:
        if data_level == "tract":
            for call, tract_filter in _apicalls_for_tract_geo(
                geo=geo, year=year, dataset=dataset,
                table_id=table_id, variables=var_ids,
            ):
                out.append(PlannedCall(
                    api_call=call, geo_idx=geo_idx, concept_idx=concept_idx,
                    year=year, role=role,
                    variables=variables_obj, tract_filter=tract_filter,
                ))
        else:
            built = _apicall_for_admin_geo(
                geo=geo, data_level=data_level, year=year,
                dataset=dataset, table_id=table_id, variables=var_ids,
            )
            if built is None:
                continue
            call, _ = built
            out.append(PlannedCall(
                api_call=call, geo_idx=geo_idx, concept_idx=concept_idx,
                year=year, role=role,
                variables=variables_obj, tract_filter=[],
            ))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_query(
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    cmap: Optional[ConceptMap],
    metadata_db: sqlite3.Connection,
    *,
    decomp_cache: Optional[DecompositionCache] = None,
    semantic_router: Optional[object] = None,    # SemanticRouter
    concept_resolutions: Optional[list[ConceptResolution]] = None,
    extra_geos: Optional[
        list[tuple[ResolvedGeography, str]]
    ] = None,
    universe_picker: Optional[object] = None,     # UniversePicker (new)
) -> PlanResult:
    """Build a PlanResult for the given extracted intent + resolved geos.

    Three input modes (first-match wins):

      - ``concept_resolutions`` provided (Phase 1+): the agent-routing
        layer (rewriter + critic) has already produced resolutions;
        skip internal resolution and just build the API calls.
        ``cmap`` / ``semantic_router`` / ``decomp_cache`` are unused.
      - ``semantic_router`` provided (K.5 legacy): one vector retrieval
        pass per concept with no critic or rewrite step.
      - Neither (original legacy): runs the Tier 1/2/3 cascade
        (``cmap`` is required).

    Does not make any network calls. The orchestrator feeds
    ``PlanResult.calls`` into CensusCaller.fetch_all and then into the
    aggregator with the same metadata.
    """
    notes: list[str] = []
    if not resolved_geos:
        notes.append("no resolved geographies — nothing to plan")
        return PlanResult(
            intent=intent, resolved_geos=[],
            concept_resolutions=[], calls=[], notes=notes,
        )

    # The data levels we need across all geos — drives both the FTS
    # coverage filter (legacy) and the router's coverage filter (K.5+).
    data_levels = sorted({data_level_for(g) for g in resolved_geos})

    if concept_resolutions is not None:
        # Caller pre-resolved concepts (e.g., via the agent-routing
        # pipeline). Validate length matches intent.concepts and use
        # verbatim.
        if len(concept_resolutions) != len(intent.concepts):
            raise ValueError(
                f"plan_query: concept_resolutions has "
                f"{len(concept_resolutions)} entries but intent has "
                f"{len(intent.concepts)} concepts"
            )
    else:
        # Resolve each concept once (independent of geo — the same
        # concept picks the same table everywhere; per-geo overrides
        # happen during call construction via resolve_for /
        # _compose_from_router).
        concept_resolutions = []
        for concept in intent.concepts:
            if semantic_router is not None:
                cr = route_concept_semantic(
                    concept, semantic_router, data_levels,
                )
            else:
                if cmap is None:
                    raise ValueError(
                        "plan_query requires one of: concept_resolutions, "
                        "semantic_router, or cmap"
                    )
                cr = resolve_concept(
                    concept, cmap, metadata_db, data_levels,
                    decomp_cache=decomp_cache,
                )
            concept_resolutions.append(cr)

    # Build the full geo list (primaries + comparator/disparity extras).
    # Extras keep their distinct geo_idx so PlannedCall ↔ resolved_geos
    # round-trips cleanly through the aggregator and synthesizer.
    extras = list(extra_geos or [])
    all_geos = list(resolved_geos) + [g for g, _ in extras]

    # Construct calls. Primary geos always get role="primary"; extras
    # carry their supplied role (e.g., "comparator.msa", "disparity.race").
    calls: list[PlannedCall] = []
    for gi, geo in enumerate(resolved_geos):
        for ci, resolution in enumerate(concept_resolutions):
            calls.extend(_calls_for_resolved_concept(
                geo, gi, ci, resolution, intent, metadata_db,
                role="primary",
                universe_picker=universe_picker,
            ))
    for j, (geo, role) in enumerate(extras):
        gi = len(resolved_geos) + j
        for ci, resolution in enumerate(concept_resolutions):
            calls.extend(_calls_for_resolved_concept(
                geo, gi, ci, resolution, intent, metadata_db,
                role=role,
                universe_picker=universe_picker,
            ))

    if not intent.concepts:
        notes.append("intent has no concepts — produced 0 calls (geo-only)")

    return PlanResult(
        intent=intent,
        resolved_geos=all_geos,
        concept_resolutions=concept_resolutions,
        calls=calls,
        notes=notes,
    )
