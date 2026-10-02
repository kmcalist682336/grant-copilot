"""Shared tract-prefix geography mapping for record stores."""
import logging
import sqlite3
from typing import Optional
from scripts.chatbot.models import ResolvedGeography

logger = logging.getLogger(__name__)

def _geo_prefixes(geo: ResolvedGeography) -> list[str]:
    """Return FIPS prefixes usable against a record census_tract column."""
    if geo.tract_geoids:
        return sorted(set(str(value) for value in geo.tract_geoids))
    if geo.geo_level == "state":
        return [str(geo.geo_id)[:2]]
    if geo.geo_level == "county":
        return [str(geo.geo_id)[:5]]
    # A place/MSA may not have a tract list in the gazetteer.  Do not guess
    # a prefix; the caller will query the available record set explicitly.
    return []

def _admin_place_tract_prefixes(
    geo: ResolvedGeography,
    geo_db: Optional[sqlite3.Connection],
) -> list[str]:
    """Expand an admin place to tracts for record-level filtering only.

    Prefer the gazetteer's curated ``admin_place_tract_map``.  That is the
    same lookup used by validation/debug queries and avoids subtle differences
    from ad-hoc geometry intersections at query time.  The spatial fallback is
    retained for older gazetteers that do not have the mapping table.
    """
    if (
        geo_db is None
        or geo.geo_level != "place"
    ):
        return []
    try:
        rows = geo_db.execute(
            """
            SELECT tract_geoid
            FROM admin_place_tract_map
            WHERE admin_geoid = ?
            ORDER BY tract_geoid
            """,
            (geo.geo_id,),
        ).fetchall()
        mapped = [str(row["tract_geoid"]) for row in rows]
        if mapped:
            return mapped
    except sqlite3.Error:
        logger.debug(
            "admin_place_tract_map unavailable; falling back to spatial "
            "place/tract intersection",
            exc_info=True,
        )
    rows = geo_db.execute(
        """
        SELECT t.geoid AS tract_geoid
        FROM admin_geographies AS t
        JOIN admin_geographies AS p
          ON p.geoid = ?
        WHERE t.geo_type = 'tract'
          AND t.state_fips = p.state_fips
          AND MbrIntersects(t.geom, p.geom)
          AND ST_Intersects(t.geom, p.geom)
        ORDER BY t.geoid
        """,
        (geo.geo_id,),
    ).fetchall()
    return [str(row["tract_geoid"]) for row in rows]

def _record_geo_prefixes(
    geo: ResolvedGeography,
    geo_db: Optional[sqlite3.Connection],
) -> list[str]:
    """Return geography filters for record-level data without changing Census."""
    prefixes = _geo_prefixes(geo)
    if prefixes:
        return prefixes
    return _admin_place_tract_prefixes(geo, geo_db)
