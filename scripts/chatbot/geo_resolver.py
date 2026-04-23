"""STEP 2: deterministic resolution of ExtractedIntent → ResolvedGeography.

No LLM here. Pure DB + rules. Unit-testable against the real gazetteer.

Public API:
    resolve_intent(intent, db)         — orchestrator entry point
    resolve_ref(ref, db, max_alts=3)   — single-ref resolution (also used by
                                         the clarifier to fetch alternatives)

Per-strategy dispatch is private and keyed off ``ref.ref_type``:
    administrative → admin_geographies (GA boost, nationwide candidates)
    neighborhood   → named_places (neighborhood-family) → place_tract_map
    corridor       → roads → road_tract_map
    region         → named_places (region-family) → place_tract_map
    landmark_area  → named_places (landmark-family) → near_tract_map
    national       → hardcoded United States resolution
    relative       → dispatched by proximity_tier (very_local/local/regional)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from scripts.chatbot.candidates import (
    ResolutionCandidate,
    fts_search,
)
from scripts.chatbot.models import (
    ExtractedGeoRef,
    ExtractedIntent,
    ResolvedGeography,
)
from scripts.chatbot.scoring import compute_confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# place_type groupings per strategy (POC)
# ---------------------------------------------------------------------------

# Neighborhoods-as-anchors: things a user would call "a neighborhood".
_NEIGHBORHOOD_PLACE_TYPES = frozenset({
    "neighborhood", "npu", "superdistrict", "beltline_sub", "nsa",
})

# Informal regions — multi-tract logical areas.
_REGION_PLACE_TYPES = frozenset({
    "hssa", "aaa", "rc", "arc_region", "area", "superdistrict", "npu",
    "beltline_sub",
})

# Landmarks — points of interest.
_LANDMARK_PLACE_TYPES = frozenset({
    "park", "airport", "university", "college", "hospital", "stadium",
    "museum", "arts", "government", "military", "correctional",
})

# admin_geographies.geo_type → our GeoLevel literal. geo_types not in
# this map are skipped (not returned as resolutions) — they're admin
# rows that aren't usefully queryable via Stage 8 data (urban clusters,
# unclassified school districts, etc.).
_ADMIN_GEO_LEVEL_MAP = {
    "state": "state",
    "county": "county",
    "tract": "tract",
    "place": "place",
    "zcta": "zcta",
    "msa": "msa",
    "micro": "msa",
    "school_district_unified": "school_district_unified",
    "school_district_elementary": "school_district_unified",
    "school_district_secondary": "school_district_unified",
}

# admin geo_types we skip as resolutions — not user-facing Census geographies.
_ADMIN_GEO_TYPE_SKIP = frozenset({"urban_cluster", "urbanized_area"})

# Corridor buffer tiers (miles) available in road_tract_map.
_ROAD_BUFFERS_AVAILABLE = [1.0, 2.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_level_for_admin(geo_type: str, state_fips: str) -> str:
    """Finest Stage 8 geography level for which we actually stored data.

    Returns one of:
      - a specific level ("state", "county", "place", "msa", "us",
        "school_district_unified") — data exists directly at this level
      - "none" — the resolver could identify the place but we have no
        Census data for it at ANY level. Triggers a data_gap clarification.

    Rules (2026-04-20 scope):
      - GA (state_fips='13'): full data at state/county/place/tract/zcta/
        school_district_unified
      - Non-GA: state-level rollup available. Counties/places/tracts
        outside GA don't have their own data, but the parent state does.
        We surface this as ``data_level_available='state'`` plus an
        assumption_note — users get the state rollup with a clear signal.
      - Non-GA ZCTAs: no data (we only have GA ZCTAs).
      - MSAs: populated if the MSA is in the top-100 filter (currently
        always at query time, since we don't know the filter here; we
        return 'msa' and let the orchestrator re-check if needed).
    """
    if state_fips == "13":
        # GA places have all levels.
        return _ADMIN_GEO_LEVEL_MAP.get(geo_type, "state")
    # Non-GA cases
    if geo_type == "state":
        return "state"
    if geo_type in ("msa", "micro"):
        # Stage 8 captured top-100 CBSAs + all GA CBSAs. We can't cheaply
        # check membership here — assume 'msa' available; the
        # orchestrator or downstream consumer can re-verify.
        return "msa"
    if geo_type == "zcta":
        return "none"      # ZCTA ingestion was GA-only
    # county, place, school_district_unified, tract outside GA:
    # we only have the parent state's rollup.
    return "state"


def _select_road_buffer(ref: ExtractedGeoRef, default: float) -> float:
    """Pick a buffer tier from the precomputed set [1, 2, 3, 5] miles.

    "along X" uses the road's default (2.0 for most Georgia corridors).
    "near X" defaults to 2.0 as well — 1.0 mi is too tight for most
    "in city Y" intersections (e.g., Downtown tracts don't overlap
    I-85's 1 mi buffer but do at 2 mi).
    "near X" at the `regional` tier goes wider (3.0 mi).
    """
    qualifier = ref.qualifier
    tier = ref.proximity_tier
    if qualifier == "along":
        target = default or 2.0
    elif qualifier == "near" and tier == "regional":
        target = 3.0
    elif qualifier == "near":
        target = 2.0
    else:
        target = default or 2.0
    return min(_ROAD_BUFFERS_AVAILABLE, key=lambda x: abs(x - target))


def _fetch_place_tracts(
    db: sqlite3.Connection, place_id: str,
) -> tuple[list[str], Optional[str]]:
    """Return (tract_geoids, fallback_county_geoid) for a named place."""
    rows = db.execute(
        "SELECT tract_geoid, county_geoid FROM place_tract_map "
        "WHERE place_id = ? ORDER BY overlap_pct DESC",
        (place_id,),
    ).fetchall()
    tract_ids = [r["tract_geoid"] for r in rows]
    # Pick the most-common containing county as the fallback
    county: Optional[str] = None
    if rows:
        from collections import Counter
        counties = Counter(r["county_geoid"] for r in rows)
        county = counties.most_common(1)[0][0]
    return tract_ids, county


def _fetch_near_place_tracts(
    db: sqlite3.Connection, place_id: str, buffer_miles: float = 0.5,
) -> tuple[list[str], Optional[str]]:
    """Return tracts + fallback county for a near-place anchor."""
    rows = db.execute(
        "SELECT tract_geoid FROM near_tract_map "
        "WHERE anchor_type = 'named_place' AND anchor_id = ? "
        "  AND buffer_miles = ? "
        "ORDER BY dist_meters",
        (place_id, buffer_miles),
    ).fetchall()
    tract_ids = [r["tract_geoid"] for r in rows]
    county: Optional[str] = None
    if tract_ids:
        # Derive county from the tract GEOID prefix (GA = state 13, county = next 3 digits)
        county = tract_ids[0][:5] if len(tract_ids[0]) >= 5 else None
    return tract_ids, county


def _fetch_near_admin_tracts(
    db: sqlite3.Connection, admin_geoid: str, buffer_miles: float = 2.0,
) -> tuple[list[str], Optional[str]]:
    """Return tracts within buffer_miles of a GA admin place."""
    rows = db.execute(
        "SELECT tract_geoid FROM near_tract_map "
        "WHERE anchor_type = 'admin_place' AND anchor_id = ? "
        "  AND buffer_miles = ? "
        "ORDER BY dist_meters",
        (admin_geoid, buffer_miles),
    ).fetchall()
    tract_ids = [r["tract_geoid"] for r in rows]
    county: Optional[str] = None
    if tract_ids:
        county = tract_ids[0][:5] if len(tract_ids[0]) >= 5 else None
    return tract_ids, county


def _fetch_road_tracts(
    db: sqlite3.Connection, road_id: str, buffer_miles: float,
) -> list[str]:
    rows = db.execute(
        "SELECT tract_geoid FROM road_tract_map "
        "WHERE road_id = ? AND buffer_miles = ? "
        "ORDER BY dist_meters",
        (road_id, buffer_miles),
    ).fetchall()
    return [r["tract_geoid"] for r in rows]


def _fetch_adjacent_counties(
    db: sqlite3.Connection, county_geoid: str,
) -> list[str]:
    rows = db.execute(
        "SELECT neighbor_geoid FROM county_adjacency "
        "WHERE county_geoid = ?",
        (county_geoid,),
    ).fetchall()
    return [r["neighbor_geoid"] for r in rows]


# ---------------------------------------------------------------------------
# Per-strategy resolvers
# ---------------------------------------------------------------------------

def _resolve_admin_candidate(
    cand: ResolutionCandidate,
    ref: ExtractedGeoRef,
    all_candidates: list[ResolutionCandidate],
    db: sqlite3.Connection,
) -> Optional[ResolvedGeography]:
    row = db.execute(
        "SELECT geoid, name, geo_type, state_fips, api_for_clause, api_in_clause "
        "FROM admin_geographies WHERE geoid = ?",
        (cand.target_id,),
    ).fetchone()
    if not row:
        return None
    if row["geo_type"] in _ADMIN_GEO_TYPE_SKIP:
        return None
    geo_level = _ADMIN_GEO_LEVEL_MAP.get(row["geo_type"])
    if geo_level is None:
        return None
    data_level = _data_level_for_admin(row["geo_type"], row["state_fips"] or "")
    notes: list[str] = []
    if row["state_fips"] != "13":
        notes.append(
            "no tract-level Census data outside Georgia — "
            "only state/MSA aggregates are available"
        )
    return ResolvedGeography(
        geo_id=row["geoid"],
        geo_level=geo_level,
        geo_type=row["geo_type"],
        display_name=row["name"],
        tract_geoids=[],
        county_geoid=(row["geoid"] if row["geo_type"] == "county" else None),
        api_for_clause=row["api_for_clause"] or "",
        api_in_clause=row["api_in_clause"] or "",
        confidence=compute_confidence(cand, ref, all_candidates),
        assumption_notes=notes,
        data_level_available=data_level,
        source_ref=ref,
    )


def _resolve_named_place_candidate(
    cand: ResolutionCandidate,
    ref: ExtractedGeoRef,
    all_candidates: list[ResolutionCandidate],
    db: sqlite3.Connection,
    *,
    use_near_buffer: Optional[float] = None,
    geo_level: str = "neighborhood",
) -> Optional[ResolvedGeography]:
    row = db.execute(
        "SELECT place_id, name, place_type FROM named_places "
        "WHERE place_id = ?",
        (cand.target_id,),
    ).fetchone()
    if not row:
        return None

    if use_near_buffer is not None:
        tracts, county = _fetch_near_place_tracts(
            db, row["place_id"], buffer_miles=use_near_buffer,
        )
    else:
        tracts, county = _fetch_place_tracts(db, row["place_id"])

    return ResolvedGeography(
        geo_id=row["place_id"],
        geo_level=geo_level,
        geo_type=row["place_type"] or "",
        display_name=row["name"],
        tract_geoids=tracts,
        county_geoid=county,
        api_for_clause="",
        api_in_clause="",
        confidence=compute_confidence(cand, ref, all_candidates),
        assumption_notes=[],
        data_level_available="tract" if tracts else "none",
        source_ref=ref,
    )


def _resolve_road_candidate(
    cand: ResolutionCandidate,
    ref: ExtractedGeoRef,
    all_candidates: list[ResolutionCandidate],
    db: sqlite3.Connection,
) -> Optional[ResolvedGeography]:
    row = db.execute(
        "SELECT road_id, name, full_name, road_type, default_buffer_miles "
        "FROM roads WHERE road_id = ?",
        (cand.target_id,),
    ).fetchone()
    if not row:
        return None
    buffer_mi = _select_road_buffer(ref, row["default_buffer_miles"] or 2.0)
    tracts = _fetch_road_tracts(db, row["road_id"], buffer_mi)
    notes = [f"using {buffer_mi:.1f}-mile buffer along {row['name']}"]
    return ResolvedGeography(
        geo_id=row["road_id"],
        geo_level="corridor",
        geo_type=row["road_type"] or "",
        display_name=row["full_name"] or row["name"],
        tract_geoids=tracts,
        county_geoid=None,
        api_for_clause="",
        api_in_clause="",
        confidence=compute_confidence(cand, ref, all_candidates),
        assumption_notes=notes,
        data_level_available="tract" if tracts else "none",
        source_ref=ref,
    )


def _sort_by_confidence(
    resolutions: list[ResolvedGeography],
) -> list[ResolvedGeography]:
    return sorted(resolutions, key=lambda r: r.confidence, reverse=True)


def _resolve_administrative(
    ref: ExtractedGeoRef,
    db: sqlite3.Connection,
    max_alternatives: int,
) -> list[ResolvedGeography]:
    cands = fts_search(db, ref.text, target_type_filter="admin", limit=10)
    out: list[ResolvedGeography] = []
    seen_ids: set[str] = set()
    for cand in cands:
        if cand.target_id in seen_ids:
            continue
        seen_ids.add(cand.target_id)
        resolved = _resolve_admin_candidate(cand, ref, cands, db)
        if resolved:
            out.append(resolved)
    out = _sort_by_confidence(out)
    return out[: 1 + max_alternatives]


def _resolve_named_place_strategy(
    ref: ExtractedGeoRef,
    db: sqlite3.Connection,
    *,
    place_type_filter: frozenset[str],
    geo_level: str,
    use_near_buffer: Optional[float],
    max_alternatives: int,
) -> list[ResolvedGeography]:
    # FTS search scoped to named_place; then filter by place_type.
    # If nothing matches the filter, fall back to ANY named_place — the
    # type-mismatch penalty in scoring lowers confidence so the clarifier
    # will surface the ambiguity. Covers gazetteer gaps (e.g., a user
    # asking for "Piedmont Park" when only a BeltLine sub-area containing
    # it exists).
    raw = fts_search(db, ref.text, target_type_filter="named_place", limit=20)
    if not raw:
        return []
    filtered = [c for c in raw if c.subtype in place_type_filter]
    fallback_used = False
    if not filtered:
        filtered = raw
        fallback_used = True

    out: list[ResolvedGeography] = []
    seen_ids: set[str] = set()
    for cand in filtered:
        if cand.target_id in seen_ids:
            continue
        seen_ids.add(cand.target_id)
        resolved = _resolve_named_place_candidate(
            cand, ref, filtered, db,
            use_near_buffer=use_near_buffer,
            geo_level=geo_level,
        )
        if resolved:
            if fallback_used:
                resolved = resolved.model_copy(update={
                    "assumption_notes": resolved.assumption_notes + [
                        f"no exact match for {ref.ref_type}; "
                        f"using closest-match named place instead"
                    ],
                })
            out.append(resolved)
    out = _sort_by_confidence(out)
    return out[: 1 + max_alternatives]


def _resolve_neighborhood(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    return _resolve_named_place_strategy(
        ref, db,
        place_type_filter=_NEIGHBORHOOD_PLACE_TYPES,
        geo_level="neighborhood",
        use_near_buffer=None,
        max_alternatives=max_alternatives,
    )


def _resolve_region(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    return _resolve_named_place_strategy(
        ref, db,
        place_type_filter=_REGION_PLACE_TYPES,
        geo_level="region",
        use_near_buffer=None,
        max_alternatives=max_alternatives,
    )


def _resolve_landmark_area(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    return _resolve_named_place_strategy(
        ref, db,
        place_type_filter=_LANDMARK_PLACE_TYPES,
        geo_level="neighborhood",
        use_near_buffer=0.5,
        max_alternatives=max_alternatives,
    )


def _resolve_corridor(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    cands = fts_search(db, ref.text, target_type_filter="road", limit=10)
    out: list[ResolvedGeography] = []
    seen_ids: set[str] = set()
    for cand in cands:
        if cand.target_id in seen_ids:
            continue
        seen_ids.add(cand.target_id)
        resolved = _resolve_road_candidate(cand, ref, cands, db)
        if resolved:
            out.append(resolved)
    out = _sort_by_confidence(out)
    return out[: 1 + max_alternatives]


def _resolve_national(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    return [
        ResolvedGeography(
            geo_id="1",
            geo_level="us",
            geo_type="us",
            display_name="United States",
            tract_geoids=[],
            county_geoid=None,
            api_for_clause="us:1",
            api_in_clause="",
            confidence=1.0,
            assumption_notes=[],
            data_level_available="us",
            source_ref=ref,
        )
    ]


def _resolve_relative(
    ref: ExtractedGeoRef, db: sqlite3.Connection, max_alternatives: int,
) -> list[ResolvedGeography]:
    """Resolve a proximity reference via its tier.

    very_local → landmark_area logic (0.5 mi around a named_place anchor).
    local      → admin place + 2.0 mi via near_tract_map.
    regional   → admin county + county_adjacency.
    """
    tier = ref.proximity_tier or "very_local"

    if tier == "very_local":
        return _resolve_landmark_area(ref, db, max_alternatives)

    if tier == "local":
        # Resolve as admin, then expand via admin-near precomputed buffer
        admin_hits = _resolve_administrative(ref, db, max_alternatives=0)
        expanded: list[ResolvedGeography] = []
        for hit in admin_hits:
            tracts, county = _fetch_near_admin_tracts(db, hit.geo_id, 2.0)
            notes = list(hit.assumption_notes)
            notes.append("expanded to 2-mile neighborhood around the place")
            expanded.append(
                hit.model_copy(update={
                    "tract_geoids": tracts,
                    "county_geoid": county or hit.county_geoid,
                    "assumption_notes": notes,
                    "data_level_available": "tract" if tracts else hit.data_level_available,
                })
            )
        return expanded

    if tier == "regional":
        # Resolve as admin (expecting a county), then expand via adjacency
        admin_hits = _resolve_administrative(ref, db, max_alternatives=0)
        expanded = []
        for hit in admin_hits:
            if hit.geo_level != "county":
                expanded.append(hit)
                continue
            neighbors = _fetch_adjacent_counties(db, hit.geo_id)
            notes = list(hit.assumption_notes)
            notes.append(
                f"expanded to {len(neighbors)} adjacent counties"
            )
            # We don't pre-expand tracts at this layer — downstream can
            # iterate the neighbor list to fetch county-level data.
            expanded.append(
                hit.model_copy(update={
                    "assumption_notes": notes,
                    # county_geoid stays as the primary; neighbors are
                    # advisory. A richer model could add neighbor_geoids.
                })
            )
        return expanded

    # Unknown tier — fall back to very_local
    return _resolve_landmark_area(ref, db, max_alternatives)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_STRATEGY_DISPATCH = {
    "administrative": _resolve_administrative,
    "neighborhood":   _resolve_neighborhood,
    "corridor":       _resolve_corridor,
    "region":         _resolve_region,
    "landmark_area":  _resolve_landmark_area,
    "national":       _resolve_national,
    "relative":       _resolve_relative,
}


def resolve_ref(
    ref: ExtractedGeoRef,
    db: sqlite3.Connection,
    *,
    max_alternatives: int = 3,
) -> list[ResolvedGeography]:
    """Resolve a single reference to [primary, alt1, alt2, ...].

    Empty list means no candidate cleared the FTS search; the clarifier
    should surface this as an open-ended clarification.
    """
    strategy = _STRATEGY_DISPATCH.get(ref.ref_type)
    if strategy is None:
        logger.warning("No resolver strategy for ref_type=%r", ref.ref_type)
        return []
    return strategy(ref, db, max_alternatives)


def resolve_intent(
    intent: ExtractedIntent,
    db: sqlite3.Connection,
) -> list[ResolvedGeography]:
    """Resolve an entire intent into one ResolvedGeography per operand.

    Each entry in ``intent.compounds`` consumes its primary index (and
    filter index, when op != 'directional') and yields a single combined
    resolution. Indices not consumed by any compound are resolved
    standalone. Output order follows geo_ref order: each operand surfaces
    at the position of its primary ref.

    Alternatives are not returned here — Phase 4's clarifier calls
    ``resolve_ref`` directly to pull them when confidence is low.
    """
    if not intent.geo_refs:
        return []

    primaries: list[Optional[ResolvedGeography]] = []
    for ref in intent.geo_refs:
        hits = resolve_ref(ref, db, max_alternatives=0)
        primaries.append(hits[0] if hits else None)

    if not intent.compounds:
        return [p for p in primaries if p is not None]

    # Lazy import to avoid circular: composer depends on models only.
    from scripts.chatbot.composer import apply_compound

    # Map primary_ref_idx → compound, and track which indices a compound
    # has consumed (so they aren't also emitted as standalone).
    compound_by_primary: dict[int, "object"] = {
        c.primary_ref_idx: c for c in intent.compounds
    }
    consumed: set[int] = set()
    for c in intent.compounds:
        consumed.add(c.primary_ref_idx)
        if c.op != "directional":
            consumed.add(c.filter_ref_idx)

    out: list[ResolvedGeography] = []
    for i, primary in enumerate(primaries):
        if i in compound_by_primary:
            combined = apply_compound(primaries, compound_by_primary[i], db)
            if combined is not None:
                out.append(combined)
            continue
        if i in consumed:
            # This ref is the filter side of some compound — already folded
            # into the compound's combined result.
            continue
        if primary is not None:
            out.append(primary)
    return out
