"""J.1 — comparator-geo synthesis.

For any sub-state geography, derive the comparator chain
(county → MSA → state → US) and return synthetic ResolvedGeography
rows the planner can fan out additional calls against.

The chain uses tables already present in the gazetteer:
  - ``admin_geographies`` — county / state / msa rows with api_*_clause
  - ``cbsa_counties`` — county_geoid → cbsa_geoid xref

The US row isn't in admin_geographies, so we synthesize it inline
(``geo_id="1"``, ``api_for_clause="us:1"``) — same convention the
geo_resolver uses for explicit national references.

Public API:
    comparator_geos(geo, db, *, max_comparators=4)
                              -> list[(ResolvedGeography, role)]
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from scripts.chatbot.models import ExtractedGeoRef, ResolvedGeography

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default comparator chain order, by primary geo level
# ---------------------------------------------------------------------------

_CHAIN_BY_LEVEL = {
    "tract":         ["county", "msa", "state", "us"],
    "neighborhood":  ["county", "msa", "state", "us"],
    "region":        ["county", "msa", "state", "us"],
    "corridor":      ["county", "msa", "state", "us"],
    "landmark_area": ["county", "msa", "state", "us"],
    "place":         ["county", "msa", "state", "us"],
    "zcta":          ["county", "msa", "state", "us"],
    "school_district_unified": ["county", "state", "us"],
    "county":        ["msa", "state", "us"],
    "msa":           ["state", "us"],
    "state":         ["us"],
    "us":            [],
}


# Synthetic geo_ref attached to comparator rows so any code that
# inspects source_ref still has a valid object.
_SYNTHETIC_REF = ExtractedGeoRef(
    text="(comparator)", ref_type="administrative",
)


def _state_fips_for_geo(
    geo: ResolvedGeography, db: sqlite3.Connection,
) -> Optional[str]:
    """Best-effort state FIPS for a geo. Tract GEOIDs encode state in
    the first 2 chars; admin geos carry it in admin_geographies."""
    if geo.county_geoid and len(geo.county_geoid) >= 2:
        return geo.county_geoid[:2]
    if geo.tract_geoids and len(geo.tract_geoids[0]) >= 2:
        return geo.tract_geoids[0][:2]
    if geo.geo_level in ("county", "place", "tract", "state"):
        # admin_geographies row has state_fips
        row = db.execute(
            "SELECT state_fips FROM admin_geographies WHERE geoid = ?",
            (geo.geo_id,),
        ).fetchone()
        if row and row["state_fips"]:
            return row["state_fips"]
    if geo.geo_level == "msa":
        # MSAs span states; use the first county's state as a stand-in.
        row = db.execute(
            "SELECT state_fips FROM cbsa_counties WHERE cbsa_geoid = ? "
            "LIMIT 1",
            (geo.geo_id,),
        ).fetchone()
        if row and row["state_fips"]:
            return row["state_fips"]
    return None


def _county_geoid_for_geo(
    geo: ResolvedGeography, db: sqlite3.Connection,
) -> Optional[str]:
    """The most representative containing county for a geo."""
    if geo.county_geoid:
        return geo.county_geoid
    if geo.geo_level == "county":
        return geo.geo_id
    # Modal county across the geo's tracts.
    if geo.tract_geoids:
        from collections import Counter
        counties = Counter(t[:5] for t in geo.tract_geoids if len(t) >= 5)
        if counties:
            return counties.most_common(1)[0][0]
    return None


def _county_to_cbsa(
    db: sqlite3.Connection, county_geoid: str,
) -> Optional[tuple[str, str]]:
    """Look up CBSA (geoid, name) for a county."""
    row = db.execute(
        "SELECT cbsa_geoid, cbsa_name FROM cbsa_counties "
        "WHERE county_geoid = ? LIMIT 1",
        (county_geoid,),
    ).fetchone()
    if row:
        return row["cbsa_geoid"], row["cbsa_name"]
    return None


def _admin_to_resolved(
    db: sqlite3.Connection, admin_geoid: str, expected_geo_type: str,
) -> Optional[ResolvedGeography]:
    """Look up an admin_geographies row by geoid + geo_type and build
    a synthetic ResolvedGeography from it."""
    row = db.execute(
        "SELECT geoid, name, geo_type, state_fips, county_fips, "
        "       api_for_clause, api_in_clause "
        "FROM admin_geographies "
        "WHERE geoid = ? AND geo_type = ?",
        (admin_geoid, expected_geo_type),
    ).fetchone()
    if not row:
        return None
    geo_level = {
        "county": "county",
        "state": "state",
        "msa": "msa",
        "place": "place",
    }.get(row["geo_type"], row["geo_type"])
    data_level_available = geo_level if geo_level != "msa" else "msa"
    return ResolvedGeography(
        geo_id=row["geoid"],
        geo_level=geo_level,                    # type: ignore[arg-type]
        geo_type=row["geo_type"],
        display_name=row["name"],
        tract_geoids=[],
        county_geoid=(row["geoid"] if geo_level == "county" else None),
        api_for_clause=row["api_for_clause"] or "",
        api_in_clause=row["api_in_clause"] or "",
        confidence=1.0,
        assumption_notes=["synthesized as comparator"],
        data_level_available=data_level_available,    # type: ignore[arg-type]
        source_ref=_SYNTHETIC_REF,
    )


def _us_geography() -> ResolvedGeography:
    """Synthesize the US row — admin_geographies doesn't carry it."""
    return ResolvedGeography(
        geo_id="1",
        geo_level="us",
        geo_type="us",
        display_name="United States",
        tract_geoids=[],
        county_geoid=None,
        api_for_clause="us:1",
        api_in_clause="",
        confidence=1.0,
        assumption_notes=["synthesized as comparator"],
        data_level_available="us",
        source_ref=_SYNTHETIC_REF,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def comparator_geos(
    geo: ResolvedGeography,
    db: sqlite3.Connection,
    *,
    max_comparators: int = 4,
) -> list[tuple[ResolvedGeography, str]]:
    """Return (geo, role) pairs for the comparator chain of ``geo``.

    Each pair's role is ``"comparator.<level>"`` (e.g.,
    ``"comparator.county"``, ``"comparator.msa"``, ``"comparator.us"``).

    Returns an empty list when:
      - ``max_comparators`` is 0
      - ``geo.geo_level`` has no chain (e.g., 'us')
      - the gazetteer can't resolve the chain (no county / MSA found)

    The chain is capped at ``max_comparators`` from the start of the
    sequence — useful for cost/latency control via REPL flags.
    """
    if max_comparators <= 0:
        return []
    chain = _CHAIN_BY_LEVEL.get(geo.geo_level, [])
    if not chain:
        return []

    # Pre-resolve identifiers we need.
    state_fips = _state_fips_for_geo(geo, db)
    county_geoid = _county_geoid_for_geo(geo, db)

    pairs: list[tuple[ResolvedGeography, str]] = []
    seen_keys: set[tuple[str, str]] = set()       # de-dup against primary

    primary_key = (geo.geo_level, geo.geo_id)
    seen_keys.add(primary_key)

    for level in chain:
        comparator: Optional[ResolvedGeography] = None
        if level == "county":
            if county_geoid:
                comparator = _admin_to_resolved(db, county_geoid, "county")
        elif level == "msa":
            if county_geoid:
                cbsa_hit = _county_to_cbsa(db, county_geoid)
                if cbsa_hit:
                    cbsa_geoid, _ = cbsa_hit
                    comparator = _admin_to_resolved(db, cbsa_geoid, "msa")
        elif level == "state":
            if state_fips:
                comparator = _admin_to_resolved(db, state_fips, "state")
        elif level == "us":
            comparator = _us_geography()

        if comparator is None:
            logger.debug(
                "comparator chain: could not resolve %s for geo %s",
                level, geo.display_name,
            )
            continue
        key = (comparator.geo_level, comparator.geo_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        pairs.append((comparator, f"comparator.{level}"))
        if len(pairs) >= max_comparators:
            break

    return pairs
