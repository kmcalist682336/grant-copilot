"""Compound-geography composer.

Three operations supported:

    intersect   — "near I-85 in downtown Atlanta"
                  set-intersect tract_geoids between primary and filter.

    directional — "north side of Piedmont Park"
                  half-plane filter on the primary's tracts using
                  compass-direction lat/lon comparisons vs. the primary's
                  geometric centroid.

    difference  — "DeKalb County excluding Decatur"
                  set-subtract: primary tracts minus filter tracts.

All operations produce a single combined ``ResolvedGeography``. Admin
primaries/filters (place, county) don't carry pre-populated tract lists
on their own — ``_ensure_tracts`` derives them on demand for the
operations that need a tract set. Empty results are surfaced explicitly
with ``data_level_available='none'`` plus an assumption_note explaining
why — the clarifier (Phase 4) is expected to turn that into a user-facing
"no tracts match" message.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from scripts.chatbot.models import (
    CompoundSpec,
    ResolvedGeography,
)

logger = logging.getLogger(__name__)

_COMPOUND_CONFIDENCE_PENALTY = 0.95


# ---------------------------------------------------------------------------
# Anchor-centroid lookup (for directional filter)
# ---------------------------------------------------------------------------

def _anchor_centroid(
    db: sqlite3.Connection,
    resolved: ResolvedGeography,
) -> Optional[tuple[float, float]]:
    """Return (lat, lon) of the primary anchor's geometric centroid.

    Looks up geometry by the resolved geography's geo_level:
      - neighborhood / region  → named_places.geom
      - corridor               → roads.geom (linestring centroid)
      - county / state / place → admin_geographies.geom
    """
    if resolved.geo_level in ("neighborhood", "region"):
        row = db.execute(
            "SELECT Y(Centroid(geom)) AS lat, X(Centroid(geom)) AS lon "
            "FROM named_places WHERE place_id = ?",
            (resolved.geo_id,),
        ).fetchone()
    elif resolved.geo_level == "corridor":
        row = db.execute(
            "SELECT Y(Centroid(geom)) AS lat, X(Centroid(geom)) AS lon "
            "FROM roads WHERE road_id = ?",
            (resolved.geo_id,),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT Y(Centroid(geom)) AS lat, X(Centroid(geom)) AS lon "
            "FROM admin_geographies WHERE geoid = ?",
            (resolved.geo_id,),
        ).fetchone()
    if not row or row["lat"] is None or row["lon"] is None:
        return None
    return float(row["lat"]), float(row["lon"])


# ---------------------------------------------------------------------------
# Directional half-plane filter
# ---------------------------------------------------------------------------

def _filter_tracts_directional(
    db: sqlite3.Connection,
    tract_geoids: list[str],
    direction: str,
    anchor_lat: float,
    anchor_lon: float,
) -> list[str]:
    """Return subset of tract_geoids whose centroids satisfy a half-plane
    test against the anchor centroid for the given compass direction.

    Direction codes:
      n  = tract_lat > anchor_lat
      s  = tract_lat < anchor_lat
      e  = tract_lon > anchor_lon
      w  = tract_lon < anchor_lon
      ne = n AND e
      nw = n AND w
      se = s AND e
      sw = s AND w
    """
    if not tract_geoids:
        return []

    # Pull centroid coords for all candidate tracts in one shot.
    placeholders = ",".join("?" * len(tract_geoids))
    rows = db.execute(
        f"SELECT geoid, "
        f"       Y(Centroid(geom)) AS lat, "
        f"       X(Centroid(geom)) AS lon "
        f"FROM admin_geographies "
        f"WHERE geoid IN ({placeholders}) AND geo_type = 'tract'",
        tract_geoids,
    ).fetchall()

    def matches(lat: float, lon: float) -> bool:
        if "n" in direction and lat <= anchor_lat:
            return False
        if "s" in direction and lat >= anchor_lat:
            return False
        if "e" in direction and lon <= anchor_lon:
            return False
        if "w" in direction and lon >= anchor_lon:
            return False
        return True

    return [
        r["geoid"] for r in rows
        if r["lat"] is not None and r["lon"] is not None
        and matches(float(r["lat"]), float(r["lon"]))
    ]


# ---------------------------------------------------------------------------
# Compound operations
# ---------------------------------------------------------------------------

def _combine_county_geoid(
    primary: ResolvedGeography,
    filter_: Optional[ResolvedGeography],
    tracts: list[str],
) -> Optional[str]:
    """Prefer a single county when the compound lies within one; otherwise
    drop the single-county hint."""
    if not tracts:
        return primary.county_geoid
    counties = {t[:5] for t in tracts if len(t) >= 5}
    if len(counties) == 1:
        return counties.pop()
    return primary.county_geoid


def _ensure_tracts(
    resolved: ResolvedGeography,
    db: sqlite3.Connection,
) -> tuple[list[str], bool]:
    """Return (tract_geoids, derived) for a resolved geography.

    If the resolution already carries a tract list (neighborhoods,
    corridors, near-X buffers), return it as-is with derived=False.
    Otherwise — for admin places/counties — derive the full tract set
    from the gazetteer and return derived=True so callers can drop a
    note about the expansion. Returns ([], False) when the admin level
    isn't derivable (state, msa).
    """
    if resolved.tract_geoids:
        return resolved.tract_geoids, False
    if resolved.geo_level == "place":
        rows = db.execute(
            "SELECT tract_geoid FROM admin_place_tract_map "
            "WHERE admin_geoid = ? ORDER BY overlap_pct DESC",
            (resolved.geo_id,),
        ).fetchall()
        return [r["tract_geoid"] for r in rows], True
    if resolved.geo_level == "county":
        # Tract GEOID structure: SSCCCNNNNNN where SS=state, CCC=county.
        # County GEOID is the 5-char prefix.
        if len(resolved.geo_id) < 5:
            return [], False
        state_fips = resolved.geo_id[:2]
        county_fips = resolved.geo_id[2:5]
        rows = db.execute(
            "SELECT geoid FROM admin_geographies "
            "WHERE geo_type = 'tract' AND state_fips = ? "
            "  AND county_fips = ?",
            (state_fips, county_fips),
        ).fetchall()
        return [r["geoid"] for r in rows], True
    return [], False


def _intersect(
    primary: ResolvedGeography,
    filter_: ResolvedGeography,
    spec: CompoundSpec,
    db: sqlite3.Connection,
) -> ResolvedGeography:
    p_tracts, p_derived = _ensure_tracts(primary, db)
    f_tracts, f_derived = _ensure_tracts(filter_, db)
    f_set = set(f_tracts)
    common = [t for t in p_tracts if t in f_set]  # preserve primary order
    conf = min(primary.confidence, filter_.confidence) * _COMPOUND_CONFIDENCE_PENALTY
    notes = list(primary.assumption_notes)
    if p_derived:
        notes.append(
            f"derived {len(p_tracts)} tracts from {primary.display_name} "
            f"before intersect"
        )
    if f_derived:
        notes.append(
            f"derived {len(f_tracts)} tracts from {filter_.display_name} "
            f"before intersect"
        )
    notes.append(f"filtered to tracts within {filter_.display_name}")
    if not common:
        notes.append(
            "no tracts satisfy both conditions — widen the buffer or "
            "pick a broader filter"
        )
    return ResolvedGeography(
        geo_id=primary.geo_id,
        geo_level=primary.geo_level,
        geo_type=primary.geo_type,
        display_name=f"{primary.display_name} in {filter_.display_name}",
        tract_geoids=common,
        county_geoid=_combine_county_geoid(primary, filter_, common),
        api_for_clause="",
        api_in_clause="",
        confidence=conf,
        assumption_notes=notes,
        data_level_available="tract" if common else "none",
        source_ref=primary.source_ref,
    )


def _difference(
    primary: ResolvedGeography,
    filter_: ResolvedGeography,
    spec: CompoundSpec,
    db: sqlite3.Connection,
) -> ResolvedGeography:
    p_tracts, p_derived = _ensure_tracts(primary, db)
    f_tracts, f_derived = _ensure_tracts(filter_, db)
    f_set = set(f_tracts)
    remaining = [t for t in p_tracts if t not in f_set]  # preserve primary order
    conf = min(primary.confidence, filter_.confidence) * _COMPOUND_CONFIDENCE_PENALTY
    notes = list(primary.assumption_notes)
    if p_derived:
        notes.append(
            f"derived {len(p_tracts)} tracts from {primary.display_name} "
            f"before subtracting"
        )
    if f_derived:
        notes.append(
            f"derived {len(f_tracts)} tracts from {filter_.display_name} "
            f"to subtract"
        )
    removed = len(p_tracts) - len(remaining)
    notes.append(
        f"excluded {removed} tract(s) overlapping {filter_.display_name}"
    )
    if not remaining:
        notes.append(
            "filter consumed every tract — no area remains after exclusion"
        )
    elif removed == 0:
        notes.append(
            f"{filter_.display_name} did not overlap any tract in "
            f"{primary.display_name} — exclusion had no effect"
        )
    return ResolvedGeography(
        geo_id=primary.geo_id,
        geo_level=primary.geo_level,
        geo_type=primary.geo_type,
        display_name=f"{primary.display_name} excluding {filter_.display_name}",
        tract_geoids=remaining,
        county_geoid=_combine_county_geoid(primary, filter_, remaining),
        api_for_clause="",
        api_in_clause="",
        confidence=conf,
        assumption_notes=notes,
        data_level_available="tract" if remaining else "none",
        source_ref=primary.source_ref,
    )


def _directional(
    primary: ResolvedGeography,
    spec: CompoundSpec,
    db: sqlite3.Connection,
) -> ResolvedGeography:
    assert spec.direction is not None
    direction = spec.direction
    conf = primary.confidence * _COMPOUND_CONFIDENCE_PENALTY
    notes = list(primary.assumption_notes)
    notes.append(
        f"filtered to tracts on the {direction} side of "
        f"{primary.display_name}"
    )

    # If primary is an admin (place / county) without a pre-populated
    # tract list, derive it now. This is the hook for "southwest Atlanta"
    # / "north Fulton" compound directionals.
    tract_geoids, derived = _ensure_tracts(primary, db)
    if derived:
        notes.append(
            f"derived {len(tract_geoids)} tracts from "
            f"{primary.display_name} before directional filter"
        )

    centroid = _anchor_centroid(db, primary)
    if centroid is None:
        notes.append("could not determine anchor centroid — filter skipped")
        return primary.model_copy(update={
            "confidence": conf,
            "assumption_notes": notes,
        })
    anchor_lat, anchor_lon = centroid

    filtered = _filter_tracts_directional(
        db, tract_geoids, direction, anchor_lat, anchor_lon,
    )
    if not filtered:
        notes.append("no tracts found on that side of the anchor")
    return ResolvedGeography(
        geo_id=primary.geo_id,
        geo_level=primary.geo_level,
        geo_type=primary.geo_type,
        display_name=f"{direction} side of {primary.display_name}",
        tract_geoids=filtered,
        county_geoid=_combine_county_geoid(primary, None, filtered),
        api_for_clause="",
        api_in_clause="",
        confidence=conf,
        assumption_notes=notes,
        data_level_available="tract" if filtered else "none",
        source_ref=primary.source_ref,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_compound(
    resolved_per_ref: list[Optional[ResolvedGeography]],
    spec: CompoundSpec,
    db: sqlite3.Connection,
) -> Optional[ResolvedGeography]:
    """Combine per-ref resolutions according to the compound spec.

    Returns None if the required refs couldn't be resolved individually.
    """
    if spec.primary_ref_idx >= len(resolved_per_ref):
        logger.warning("compound.primary_ref_idx out of range")
        return None
    primary = resolved_per_ref[spec.primary_ref_idx]
    if primary is None:
        return None

    if spec.op == "directional":
        return _directional(primary, spec, db)

    if spec.op in ("intersect", "difference"):
        if spec.filter_ref_idx >= len(resolved_per_ref):
            logger.warning("compound.filter_ref_idx out of range")
            return None
        filter_ = resolved_per_ref[spec.filter_ref_idx]
        if filter_ is None:
            return None
        if spec.op == "intersect":
            return _intersect(primary, filter_, spec, db)
        return _difference(primary, filter_, spec, db)

    logger.warning("Unknown compound op: %r", spec.op)
    return None
