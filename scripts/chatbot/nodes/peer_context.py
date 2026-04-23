"""PeerContext — attach grant-narrative peer references to a query.

Runs after the aggregator. Looks up the primary geo in
``peer_features.sqlite`` via :class:`PeerRetriever`, picks 1-3
narrative-relevant axes based on the query's concepts, and returns a
ranked list of peer names per axis.

The node is defensive: if the PeerRetriever can't find the anchor
(unsupported geo level, missing coverage), it returns an empty list
rather than raising. Peer context is non-critical enrichment.

Public API:
    PeerContext             — one axis's peer list for the synthesizer
    get_peer_contexts(...)  — entry point
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.models import ExtractedIntent, ResolvedGeography
from scripts.chatbot.nodes.peer_retriever import (
    PeerRef, PeerRetriever, PeerRetrievalError,
)
from scripts.chatbot.peer_features_catalog import AXES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class PeerContext(BaseModel):
    """One axis's peer set for the synthesizer to reference."""

    model_config = ConfigDict(extra="ignore")

    anchor_geo_name: str
    anchor_geo_level: str
    axis: str
    axis_description: str
    # Peers in similarity order, strongest first.
    peers: list[PeerRef] = Field(default_factory=list)
    # Brief label for the peer pool — "within Georgia" when we
    # restricted to the same state, else "nationwide size-matched".
    pool_scope: str = "nationwide size-matched"
    # Anchor's own feature values for the features that feed this
    # axis. The synthesizer uses these to say e.g. "DeKalb's poverty
    # rate is 13.5%, comparable to Cobb's 8.7%." All peers' values
    # live on each PeerRef.feature_values.
    anchor_feature_values: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concept → axis keyword map
# ---------------------------------------------------------------------------

# Ordered so more specific matches can win when multiple apply.
_AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "housing": (
        "rent", "housing", "home value", "homeowner", "owner-occupied",
        "renter", "cost burden", "crowd", "vacan", "mortgage",
    ),
    "food_benefits": (
        "snap", "food stamp", "food insecur", "food access",
        "food pantry", "hunger", "wic",
    ),
    "digital_access": (
        "internet", "broadband", "digital", "computer", "online",
        "device",
    ),
    "veterans": ("veteran",),
    "disability": ("disability", "disabled", "ambulatory", "cognitive"),
    "family_structure": (
        "single mother", "single father", "single parent",
        "grandparent", "family structure",
    ),
    "commute_transit": (
        "commute", "transit", "transport", "travel time",
        "vehicle", "car", "walked",
    ),
    "health_insurance": (
        "insurance", "uninsured", "medicaid", "medicare",
        "health coverage",
    ),
    "language": (
        "language", "spanish", "english proficien", "limited english",
        "linguistic",
    ),
    "nativity": (
        "immigra", "foreign-born", "foreign born", "native-born",
        "native born", "naturalized", "citizen",
    ),
    "race_ethnicity": (
        "race", "ethnicity", "hispanic", "latino", "black", "african",
        "asian", "white", "nhpi", "aian", "indigenous",
    ),
    "education": (
        "bachelor", "degree", "education", "high school", "college",
        "diploma", "graduate", "attainment", "literacy",
    ),
    "age_structure": (
        "senior", "elderly", "older adult",
        "child", "under 18", "youth", "age distribution",
        "dependency", " 65 ", "aging",
    ),
    "employment": (
        "occupation", "industry", "job", "manufacturing",
        "service sector", "professional occupation",
    ),
    "residential_stability": (
        "displace", "mobility", "moved", "stayed", "migration",
    ),
    "income_distribution": (
        "inequality", "income distribution", "top ", "bottom ",
        "gini",
    ),
    # Economic is checked LAST so the more specific housing /
    # food_benefits keys can win first. Its keywords are broad because
    # "economic" is the sensible default for most grant queries.
    "economic": (
        "poverty", "income", "unemploy", "labor", "wage", "earn",
        "wealth", "economic", "economy", "financial",
    ),
    "size": ("population size", "population count"),
}


def _axes_for_text(text: str) -> list[str]:
    """Return axes whose keywords appear in the query/concept text,
    preserving the iteration order above (housing > food_benefits >
    ... > economic)."""
    t = text.lower()
    hits: list[str] = []
    for axis, kws in _AXIS_KEYWORDS.items():
        if any(kw in t for kw in kws):
            hits.append(axis)
    return hits


def _axes_for_intent(intent: ExtractedIntent, query: str) -> list[str]:
    """Pick up to `max_axes` axes most relevant to the query."""
    collected: list[str] = []
    for axis in _axes_for_text(query):
        if axis not in collected:
            collected.append(axis)
    for concept in intent.concepts:
        text = (concept.canonical_hint or concept.text or "")
        for axis in _axes_for_text(text):
            if axis not in collected:
                collected.append(axis)
    if not collected:
        collected = ["economic"]     # sensible default
    return collected


# ---------------------------------------------------------------------------
# Geo-level / geo_id translation
# ---------------------------------------------------------------------------

# Geo levels that map directly to entries in peer_features.sqlite.
# Neighborhoods are supported in two modes:
#   1. Exact-match: the gazetteer geo_id starts with ``ATL_NBH_`` →
#      direct lookup against pre-computed neighborhood rows.
#   2. Composite: the user query resolves to a multi-tract neighborhood
#      (e.g. "Buckhead" = 33 tracts spanning several sub-neighborhoods).
#      We aggregate the constituent tract-level neighborhoods' feature
#      vectors on the fly via ``_composite_anchor_from_tracts``.
_SUPPORTED_ANCHOR_LEVELS: dict[str, str] = {
    "county": "county",
    "msa": "msa",
    "place": "place",
    "neighborhood": "neighborhood",
}


def _resolve_anchor_key(
    geo: ResolvedGeography,
) -> Optional[tuple[str, str]]:
    """Return (peer_level, peer_geo_id) or None when the geo isn't
    directly representable in peer_features. Composite neighborhoods
    (multi-tract aggregates with non-`ATL_NBH_*` geo_ids) are handled
    separately via the composite-anchor path — see
    ``_composite_anchor_from_tracts``."""
    peer_level = _SUPPORTED_ANCHOR_LEVELS.get(geo.geo_level)
    if peer_level is None:
        return None
    gid = geo.geo_id
    if not gid:
        return None
    # Neighborhoods come in two flavors. Only direct `ATL_NBH_*` ids
    # exist in peer_features; composite geo_ids (e.g. synthetic IDs
    # built by the resolver for multi-tract neighborhoods) need the
    # composite-anchor path instead.
    if peer_level == "neighborhood" and not gid.startswith("ATL_NBH_"):
        return None
    return peer_level, gid


def _anchor_by_tract_overlap(
    primary: ResolvedGeography,
    peer_retriever: PeerRetriever,
) -> Optional[object]:
    """Look up the pre-computed neighborhood whose tract list overlaps
    the query's tract_geoids most, and return that neighborhood as the
    anchor. None when no overlap is found.

    This is the composite-neighborhood path — handles queries like
    "North Buckhead" or "Buckhead" where the gazetteer aggregates
    tract lists that don't map 1:1 onto atl_opendata neighborhoods.
    """
    query_tracts = set(primary.tract_geoids or [])
    if not query_tracts:
        return None

    # We need tract membership per stored ATL_NBH_* neighborhood. Rather
    # than round-trip to the gazetteer (which the peer_retriever doesn't
    # have access to), rely on the fact that ATL neighborhood features
    # were derived from specific tract sets at build time. We re-derive
    # the match via a lightweight gazetteer query inside the retriever's
    # helper. Keep this path best-effort: if gazetteer lookup fails,
    # fall through to None and the caller skips peer context.
    try:
        import sqlite3
        from pathlib import Path
        gaz_path = (
            Path(peer_retriever._db_path).parent.parent
            / "geo" / "gazetteer.db"
        )
        if not gaz_path.exists():
            return None
        gaz = sqlite3.connect(gaz_path)
        gaz.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in query_tracts)
        rows = gaz.execute(
            f"""
            SELECT np.place_id, np.name,
                   COUNT(ptm.tract_geoid) AS overlap
              FROM named_places np
              JOIN place_tract_map ptm ON ptm.place_id = np.place_id
             WHERE np.source = 'atl_opendata'
               AND np.place_type = 'neighborhood'
               AND ptm.tract_geoid IN ({placeholders})
             GROUP BY np.place_id
             ORDER BY overlap DESC
             LIMIT 5
            """,
            list(query_tracts),
        ).fetchall()
        gaz.close()
    except Exception as e:                         # pragma: no cover
        logger.debug("tract-overlap fallback failed: %s", e)
        return None

    # Walk the top matches; the first one with features in
    # peer_features wins. Some neighborhoods may not have been
    # processed by the pre-compute tool (min_tracts filter in an
    # older run), so it's worth trying a few.
    for r in rows:
        try:
            anchor = peer_retriever.lookup_anchor(
                geo_level="neighborhood", geo_id=r["place_id"],
            )
        except PeerRetrievalError:
            continue
        if anchor is not None:
            return anchor
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_peer_contexts(
    *,
    resolved_geos: list[ResolvedGeography],
    intent: ExtractedIntent,
    query: str,
    peer_retriever: Optional[PeerRetriever],
    max_axes: int = 3,
    top_k: int = 5,
    restrict_to_state: bool = True,
) -> list[PeerContext]:
    """Return grant-narrative peer context for the primary geo.

    Strategy:
      1. Pick 1-3 axes relevant to the concepts in the query
         (housing/food_benefits/economic/etc.).
      2. Look up the primary geo in peer_features.sqlite.
      3. For each axis, return top-K peers. When the primary is a
         county and ``restrict_to_state`` is True, try the within-state
         pool first and fall back to nationwide if empty.
      4. Silent no-op for unsupported geo levels (neighborhoods,
         tracts, regions, states, US) — returns [].
    """
    if peer_retriever is None or not resolved_geos:
        return []
    primary = resolved_geos[0]

    anchor = None
    # Direct-match path: geo_id is one of the pre-computed entries.
    key = _resolve_anchor_key(primary)
    if key is not None:
        peer_level, geo_id = key
        try:
            anchor = peer_retriever.lookup_anchor(
                geo_level=peer_level, geo_id=geo_id,
            )
        except PeerRetrievalError as e:
            logger.warning("peer_context: lookup failed: %s", e)
            return []

    # Tract-overlap fallback: when the gazetteer resolves to a
    # neighborhood that doesn't share an ID with peer_features
    # (e.g. "North Buckhead" → synthetic geo with 9 tracts), find the
    # stored neighborhood whose tract list overlaps the query's most
    # and use its features as the anchor. Better than nothing for
    # grant-narrative peer context.
    if anchor is None and primary.geo_level == "neighborhood" \
            and primary.tract_geoids:
        anchor = _anchor_by_tract_overlap(primary, peer_retriever)
        if anchor is not None:
            peer_level = "neighborhood"
            logger.info(
                "peer_context: neighborhood %r matched via tract "
                "overlap to stored %r",
                primary.display_name, anchor.geo_name,
            )

    if anchor is None:
        logger.debug(
            "peer_context: no anchor for %s / %s (%s)",
            primary.geo_level, primary.geo_id, primary.display_name,
        )
        return []

    axes = _axes_for_intent(intent, query)[:max_axes]
    # State restriction only meaningful for county-level peers.
    restrict_state = (
        anchor.state_fips
        if (restrict_to_state and peer_level == "county")
        else None
    )

    contexts: list[PeerContext] = []
    for axis in axes:
        if axis not in AXES:
            continue
        description = AXES[axis].get("description", "")
        axis_features = list(AXES[axis].get("features", []))
        try:
            peers = peer_retriever.peers(
                anchor, axis=axis, geo_level=peer_level,
                top_k=top_k, restrict_state=restrict_state,
            )
            scope = (
                f"within state {anchor.state_fips}" if restrict_state
                else "nationwide size-matched"
            )
            # Fall back to nationwide if within-state produced nothing
            # meaningful.
            if restrict_state and len(peers) < 2:
                peers = peer_retriever.peers(
                    anchor, axis=axis, geo_level=peer_level,
                    top_k=top_k, restrict_state=None,
                )
                scope = "nationwide size-matched"
        except PeerRetrievalError as e:
            logger.warning(
                "peer_context: ranking failed on axis=%s: %s", axis, e,
            )
            continue
        if not peers:
            continue
        # Anchor's values on this axis — lets the synthesizer write
        # "DeKalb 13.5% vs Cobb 8.7%" style comparisons.
        anchor_vals = {
            fname: anchor.features[fname]
            for fname in axis_features
            if fname in anchor.features
        }
        contexts.append(PeerContext(
            anchor_geo_name=anchor.geo_name,
            anchor_geo_level=anchor.geo_level,
            axis=axis,
            axis_description=description,
            peers=peers,
            pool_scope=scope,
            anchor_feature_values=anchor_vals,
        ))
    return contexts
