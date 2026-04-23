"""Pre-compute spatial join mappings for fast query-time resolution.

Runs once during gazetteer build. Results stored in:
  place_tract_map  — which tracts overlap each named place
  road_tract_map   — which tracts are within N miles of each road

At query time, these tables replace expensive spatial joins with
simple indexed lookups.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# Small geometry threshold — GNIS/landmark boxes are ~0.01 degrees wide
# Real neighborhood polygons are much larger
# ~0.01 * 0.02 degrees ≈ 500m x 1km bounding box
SMALL_GEOM_THRESHOLD_SQ_DEG = 0.0002

# Buffer distances to pre-compute for road corridors
# Users can say "along I-85" (1 mile), "near I-85" (2 miles),
# "I-85 corridor" (3 miles), "I-85 area" (5 miles)
ROAD_BUFFER_MILES = [1.0, 2.0, 3.0, 5.0]

# Meters per mile
METERS_PER_MILE = 1609.34

# EPSG:26916 — UTM Zone 16N (meters) — used for distance calculations in Georgia
UTM_SRID = 26916


def compute_place_tract_mappings(
    conn: sqlite3.Connection,
    state_fips: str = "13",
    min_overlap_pct: float = 0.05,
) -> int:
    """Compute which tracts overlap each named place polygon.

    Uses different strategies based on geometry size:
    - Real polygons (neighborhoods, districts): area overlap percentage
    - Point-like geometries (GNIS, landmarks stored as tiny boxes):
      point-in-polygon using the centroid — every point must fall in exactly
      one tract since tracts have 100% coverage with no gaps
    """
    logger.info("Computing place → tract mappings for state %s", state_fips)

    places = conn.execute(
        "SELECT place_id, name FROM named_places "
        "WHERE state_fips = ? AND geom IS NOT NULL",
        (state_fips,),
    ).fetchall()

    if not places:
        logger.warning("No named places found for state %s", state_fips)
        return 0

    # Pre-validate all geometries to avoid repeated MakeValid() calls
    # This fixes invalid geometries (common in coastal areas) once upfront
    logger.info("  Validating tract geometries...")
    invalid_tracts = conn.execute("""
        SELECT geoid FROM admin_geographies
        WHERE geo_type = 'tract' AND state_fips = ?
          AND NOT IsValid(geom)
    """, (state_fips,)).fetchall()

    if invalid_tracts:
        logger.info("  Repairing %d invalid tract geometries", len(invalid_tracts))
        with conn:
            for row in invalid_tracts:
                conn.execute("""
                    UPDATE admin_geographies
                    SET geom = CastToMultiPolygon(MakeValid(geom))
                    WHERE geoid = ?
                """, (row["geoid"],))

    invalid_places = conn.execute("""
        SELECT place_id FROM named_places
        WHERE state_fips = ? AND geom IS NOT NULL
          AND NOT IsValid(geom)
    """, (state_fips,)).fetchall()

    if invalid_places:
        logger.info("  Repairing %d invalid named place geometries", len(invalid_places))
        with conn:
            for row in invalid_places:
                conn.execute("""
                    UPDATE named_places
                    SET geom = CastToMultiPolygon(MakeValid(geom))
                    WHERE place_id = ?
                """, (row["place_id"],))

    total_inserted = 0

    for place in places:
        place_id = place["place_id"]
        place_name = place["name"]
        inserted_for_place = 0

        # Check if this is a small/point geometry or a real polygon
        area = conn.execute(
            "SELECT Area(geom) FROM named_places WHERE place_id = ?",
            (place_id,),
        ).fetchone()
        geom_area = float(area[0] or 0) if area else 0

        if geom_area < SMALL_GEOM_THRESHOLD_SQ_DEG:
            # Point-like geometry — use centroid point-in-polygon
            # Every point falls in exactly one tract (100% coverage, no gaps)
            # Skip spatial index hint for point queries — the index needs a
            # bounding box, not a point. For 2,796 tracts this is fast enough.
            result = conn.execute("""
                SELECT
                    t.geoid as tract_geoid,
                    substr(t.geoid, 1, 5) as county_geoid
                FROM admin_geographies t, named_places p
                WHERE p.place_id = ?
                  AND t.geo_type = 'tract'
                  AND t.state_fips = ?
                  AND Contains(t.geom, Centroid(p.geom))
                LIMIT 1
            """, (place_id, state_fips)).fetchall()

            for tract_row in result:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO place_tract_map "
                        "(place_id, tract_geoid, county_geoid, "
                        " overlap_pct, overlap_type) "
                        "VALUES (?, ?, ?, 1.0, 'contains')",
                        (place_id,
                         tract_row["tract_geoid"],
                         tract_row["county_geoid"]),
                    )
                    inserted_for_place += 1
                except sqlite3.IntegrityError:
                    pass

        else:
            # Real polygon — use area overlap
            # Geometries were pre-validated above so no MakeValid() needed here
            intersecting = conn.execute("""
                SELECT
                    t.geoid as tract_geoid,
                    substr(t.geoid, 1, 5) as county_geoid,
                    CASE
                        WHEN Area(Transform(t.geom, ?)) = 0 THEN 0
                        ELSE Area(Intersection(
                            Transform(t.geom, ?),
                            Transform(p.geom, ?)
                        )) / Area(Transform(t.geom, ?))
                    END as tract_overlap_pct,
                    CASE
                        WHEN Area(Transform(p.geom, ?)) = 0 THEN 0
                        ELSE Area(Intersection(
                            Transform(t.geom, ?),
                            Transform(p.geom, ?)
                        )) / Area(Transform(p.geom, ?))
                    END as place_overlap_pct
                FROM admin_geographies t
                JOIN named_places p ON p.place_id = ?
                WHERE t.geo_type = 'tract'
                  AND t.state_fips = ?
                  AND Intersects(t.geom, p.geom)
                  AND t.rowid IN (
                      SELECT rowid FROM SpatialIndex
                      WHERE f_table_name = 'admin_geographies'
                        AND search_frame = p.geom
                  )
            """, (
                UTM_SRID, UTM_SRID, UTM_SRID, UTM_SRID,  # tract_overlap
                UTM_SRID, UTM_SRID, UTM_SRID, UTM_SRID,  # place_overlap
                place_id, state_fips,
            )).fetchall()

            for tract_row in intersecting:
                tract_pct = float(tract_row["tract_overlap_pct"] or 0)
                place_pct = float(tract_row["place_overlap_pct"] or 0)
                # Use the larger of the two — catches both small and large places
                overlap_pct = max(tract_pct, place_pct)

                if overlap_pct < min_overlap_pct:
                    continue

                if place_pct >= 0.95:
                    overlap_type = "contains"   # tract contains most of place
                elif tract_pct >= 0.5:
                    overlap_type = "majority"   # place covers most of tract
                else:
                    overlap_type = "intersects"

                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO place_tract_map "
                        "(place_id, tract_geoid, county_geoid, "
                        " overlap_pct, overlap_type) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            place_id,
                            tract_row["tract_geoid"],
                            tract_row["county_geoid"],
                            round(overlap_pct, 4),
                            overlap_type,
                        ),
                    )
                    inserted_for_place += 1
                except sqlite3.IntegrityError:
                    pass

        if inserted_for_place == 0:
            logger.warning(
                "  %s → no tract mapping (centroid may be outside Georgia "
                "or geometry is invalid)", place_name,
            )
        else:
            logger.debug(
                "  %s → %d tracts", place_name, inserted_for_place
            )

        total_inserted += inserted_for_place

    conn.commit()
    logger.info(
        "Place→tract mappings complete: %d total mappings for %d places",
        total_inserted, len(places),
    )
    return total_inserted


def compute_road_tract_mappings(
    conn: sqlite3.Connection,
    state_fips: str = "13",
    buffer_miles: Optional[list[float]] = None,
) -> int:
    """Compute which tracts fall within N miles of each road.

    Stores results in road_tract_map for each buffer distance.

    Args:
        conn: SpatiaLite connection
        state_fips: Only map tracts in this state
        buffer_miles: Buffer distances to pre-compute

    Returns:
        Number of mappings inserted
    """
    if buffer_miles is None:
        buffer_miles = ROAD_BUFFER_MILES

    logger.info(
        "Computing road → tract mappings for buffers: %s miles",
        buffer_miles,
    )

    roads = conn.execute(
        "SELECT road_id, name FROM roads WHERE state_fips = ?",
        (state_fips,),
    ).fetchall()

    if not roads:
        logger.warning("No roads found for state %s", state_fips)
        return 0

    total_inserted = 0

    for road in roads:
        road_id = road["road_id"]
        road_name = road["name"]

        for miles in buffer_miles:
            buffer_meters = miles * METERS_PER_MILE

            # Find all tracts within N meters of this road using
            # SpatiaLite-compatible buffer approach
            # (ST_DWithin is PostGIS only — SpatiaLite uses Buffer+Intersects)
            nearby_tracts = conn.execute("""
                SELECT
                    t.geoid as tract_geoid,
                    substr(t.geoid, 1, 5) as county_geoid,
                    Distance(
                        Transform(Centroid(t.geom), ?),
                        Transform(r.geom, ?)
                    ) as dist_meters
                FROM admin_geographies t
                JOIN roads r ON r.road_id = ?
                WHERE t.geo_type = 'tract'
                  AND t.state_fips = ?
                  AND Intersects(
                      t.geom,
                      Transform(
                          Buffer(Transform(r.geom, ?), ?),
                          4326
                      )
                  )
                  AND t.rowid IN (
                      SELECT rowid FROM SpatialIndex
                      WHERE f_table_name = 'admin_geographies'
                        AND search_frame = Transform(
                            Buffer(Transform(r.geom, ?), ?),
                            4326
                        )
                  )
            """, (
                UTM_SRID, UTM_SRID,
                road_id, state_fips,
                UTM_SRID, buffer_meters,
                UTM_SRID, buffer_meters,
            )).fetchall()

            for tract_row in nearby_tracts:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO road_tract_map "
                        "(road_id, buffer_miles, tract_geoid, "
                        " county_geoid, dist_meters) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            road_id, miles,
                            tract_row["tract_geoid"],
                            tract_row["county_geoid"],
                            round(float(tract_row["dist_meters"] or 0), 1),
                        ),
                    )
                    total_inserted += 1
                except sqlite3.IntegrityError:
                    pass

            logger.debug(
                "  %s @ %.1f miles → %d tracts",
                road_name, miles, len(nearby_tracts),
            )

    conn.commit()
    logger.info(
        "Road→tract mappings complete: %d total mappings", total_inserted
    )
    return total_inserted


def get_tracts_for_place(
    conn: sqlite3.Connection,
    place_id: str,
    min_overlap: str = "intersects",
) -> list[dict]:
    """Look up pre-computed tract mapping for a named place.

    Args:
        place_id: Named place ID
        min_overlap: Minimum overlap type ('intersects'|'majority'|'contains')

    Returns:
        List of {tract_geoid, county_geoid, overlap_pct, overlap_type}
    """
    overlap_filter = {
        "contains": ("contains",),
        "majority": ("contains", "majority"),
        "intersects": ("contains", "majority", "intersects"),
    }.get(min_overlap, ("contains", "majority", "intersects"))

    placeholders = ",".join("?" * len(overlap_filter))
    rows = conn.execute(
        f"SELECT tract_geoid, county_geoid, overlap_pct, overlap_type "
        f"FROM place_tract_map "
        f"WHERE place_id = ? AND overlap_type IN ({placeholders}) "
        f"ORDER BY overlap_pct DESC",
        (place_id, *overlap_filter),
    ).fetchall()

    return [dict(r) for r in rows]


def get_tracts_for_road(
    conn: sqlite3.Connection,
    road_id: str,
    buffer_miles: float = 2.0,
) -> list[dict]:
    """Look up pre-computed tract mapping for a road corridor.

    Selects the closest pre-computed buffer distance to the requested one.

    Returns:
        List of {tract_geoid, county_geoid, dist_meters}
    """
    # Find the closest pre-computed buffer
    available = conn.execute(
        "SELECT DISTINCT buffer_miles FROM road_tract_map "
        "WHERE road_id = ? ORDER BY ABS(buffer_miles - ?) LIMIT 1",
        (road_id, buffer_miles),
    ).fetchone()

    if not available:
        return []

    actual_buffer = available["buffer_miles"]
    rows = conn.execute(
        "SELECT tract_geoid, county_geoid, dist_meters "
        "FROM road_tract_map "
        "WHERE road_id = ? AND buffer_miles = ? "
        "ORDER BY dist_meters",
        (road_id, actual_buffer),
    ).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Near-tract and county-adjacency precomputes (chatbot resolver)
# ---------------------------------------------------------------------------

# Regional/metro-scale named_place types that are filters, not near-anchors.
# "Near ITP" or "near RC1" is nonsensical for tract-level proximity.
NEAR_ANCHOR_EXCLUDED_PLACE_TYPES = frozenset({
    "aaa", "rc", "arc_region", "hssa", "area",
})

# Default "near" buffer for tract-level anchors (very_local tier).
NEAR_BUFFER_MILES = 0.5

# "near" buffer for city/place-scale anchors (local tier — "near Decatur").
NEAR_ADMIN_BUFFER_MILES = 2.0


def compute_near_tract_mappings(
    conn: sqlite3.Connection,
    state_fips: str = "13",
    buffer_miles: float = NEAR_BUFFER_MILES,
) -> int:
    """Pre-compute 'near X' tract lists for every anchor-eligible named place.

    For each named_place (excluding regional/metro-scale types), buffer the
    geometry outward by ``buffer_miles`` in UTM 26916 and record every GA
    tract that intersects the buffer. Inclusive: the anchor's own tracts
    are kept (users saying "near Buckhead" typically include Buckhead).
    """
    logger.info(
        "Computing near-tract mappings at %.2f mi (state %s)",
        buffer_miles, state_fips,
    )

    excluded = sorted(NEAR_ANCHOR_EXCLUDED_PLACE_TYPES)
    excluded_csv = ",".join("?" * len(excluded))
    places = conn.execute(
        f"SELECT place_id, name, place_type FROM named_places "
        f"WHERE state_fips = ? AND geom IS NOT NULL "
        f"  AND place_type NOT IN ({excluded_csv})",
        (state_fips, *excluded),
    ).fetchall()

    if not places:
        logger.warning(
            "No near-anchor places found for state %s", state_fips
        )
        return 0

    buffer_meters = buffer_miles * METERS_PER_MILE
    logger.info("  Processing %d anchor places", len(places))

    total_inserted = 0
    commit_every = 500

    for i, place in enumerate(places):
        place_id = place["place_id"]

        rows = conn.execute("""
            SELECT
                t.geoid AS tract_geoid,
                Distance(
                    Transform(Centroid(t.geom), ?),
                    Transform(np.geom, ?)
                ) AS dist_meters
            FROM admin_geographies t
            JOIN named_places np ON np.place_id = ?
            WHERE t.geo_type = 'tract'
              AND t.state_fips = ?
              AND Intersects(
                  t.geom,
                  Transform(
                      Buffer(Transform(np.geom, ?), ?),
                      4326
                  )
              )
              AND t.rowid IN (
                  SELECT rowid FROM SpatialIndex
                  WHERE f_table_name = 'admin_geographies'
                    AND search_frame = Transform(
                        Buffer(Transform(np.geom, ?), ?),
                        4326
                    )
              )
        """, (
            UTM_SRID, UTM_SRID,
            place_id, state_fips,
            UTM_SRID, buffer_meters,
            UTM_SRID, buffer_meters,
        )).fetchall()

        for r in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO near_tract_map "
                    "(anchor_type, anchor_id, buffer_miles, tract_geoid, "
                    " dist_meters) "
                    "VALUES ('named_place', ?, ?, ?, ?)",
                    (
                        place_id,
                        buffer_miles,
                        r["tract_geoid"],
                        round(float(r["dist_meters"] or 0), 1),
                    ),
                )
                total_inserted += 1
            except sqlite3.IntegrityError:
                pass

        if (i + 1) % commit_every == 0:
            conn.commit()
            logger.info(
                "  [%d/%d] anchors processed, %d mappings so far",
                i + 1, len(places), total_inserted,
            )

    conn.commit()
    logger.info(
        "Near-tract mappings complete: %d mappings across %d anchors",
        total_inserted, len(places),
    )
    return total_inserted


def compute_admin_near_tract_mappings(
    conn: sqlite3.Connection,
    state_fips: str = "13",
    buffer_miles: float = NEAR_ADMIN_BUFFER_MILES,
) -> int:
    """Pre-compute 'near X' tract lists where X is a GA admin place.

    Supports the ``local`` proximity tier: "near Decatur" at ~2 miles.
    Iterates admin_geographies rows with geo_type='place' and the given
    state_fips, buffers outward by ``buffer_miles``, records every GA
    tract that intersects the buffer into ``near_tract_map`` with
    ``anchor_type='admin_place'``.
    """
    logger.info(
        "Computing admin-near-tract mappings at %.2f mi (state %s)",
        buffer_miles, state_fips,
    )

    places = conn.execute(
        "SELECT geoid, name FROM admin_geographies "
        "WHERE state_fips = ? AND geo_type = 'place' AND geom IS NOT NULL",
        (state_fips,),
    ).fetchall()

    if not places:
        logger.warning(
            "No admin places found for state %s", state_fips
        )
        return 0

    buffer_meters = buffer_miles * METERS_PER_MILE
    logger.info("  Processing %d admin places", len(places))

    total_inserted = 0
    commit_every = 200

    for i, place in enumerate(places):
        geoid = place["geoid"]

        rows = conn.execute("""
            SELECT
                t.geoid AS tract_geoid,
                Distance(
                    Transform(Centroid(t.geom), ?),
                    Transform(ap.geom, ?)
                ) AS dist_meters
            FROM admin_geographies t
            JOIN admin_geographies ap ON ap.geoid = ?
            WHERE t.geo_type = 'tract'
              AND t.state_fips = ?
              AND Intersects(
                  t.geom,
                  Transform(
                      Buffer(Transform(ap.geom, ?), ?),
                      4326
                  )
              )
              AND t.rowid IN (
                  SELECT rowid FROM SpatialIndex
                  WHERE f_table_name = 'admin_geographies'
                    AND search_frame = Transform(
                        Buffer(Transform(ap.geom, ?), ?),
                        4326
                    )
              )
        """, (
            UTM_SRID, UTM_SRID,
            geoid, state_fips,
            UTM_SRID, buffer_meters,
            UTM_SRID, buffer_meters,
        )).fetchall()

        for r in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO near_tract_map "
                    "(anchor_type, anchor_id, buffer_miles, tract_geoid, "
                    " dist_meters) "
                    "VALUES ('admin_place', ?, ?, ?, ?)",
                    (
                        geoid,
                        buffer_miles,
                        r["tract_geoid"],
                        round(float(r["dist_meters"] or 0), 1),
                    ),
                )
                total_inserted += 1
            except sqlite3.IntegrityError:
                pass

        if (i + 1) % commit_every == 0:
            conn.commit()
            logger.info(
                "  [%d/%d] admin places processed, %d mappings so far",
                i + 1, len(places), total_inserted,
            )

    conn.commit()
    logger.info(
        "Admin-near-tract mappings complete: %d mappings across %d places",
        total_inserted, len(places),
    )
    return total_inserted


def compute_county_adjacency(
    conn: sqlite3.Connection,
    state_fips: str = "13",
) -> int:
    """Build a symmetric county-adjacency table for the given state.

    Both directions (a→b, b→a) are stored so lookups don't care about
    ordering. Self-loops (a→a) are included so "near X county" uniformly
    includes X itself without special-casing at the resolver layer.
    """
    logger.info("Computing county adjacency for state %s", state_fips)

    counties = conn.execute(
        "SELECT geoid FROM admin_geographies "
        "WHERE geo_type = 'county' AND state_fips = ?",
        (state_fips,),
    ).fetchall()

    if not counties:
        logger.warning("No counties found for state %s", state_fips)
        return 0

    pairs = conn.execute("""
        SELECT a.geoid AS a_geoid, b.geoid AS b_geoid
        FROM admin_geographies a, admin_geographies b
        WHERE a.geo_type = 'county' AND a.state_fips = ?
          AND b.geo_type = 'county' AND b.state_fips = ?
          AND a.geoid != b.geoid
          AND ST_Touches(a.geom, b.geom)
          AND a.rowid IN (
              SELECT rowid FROM SpatialIndex
              WHERE f_table_name = 'admin_geographies'
                AND search_frame = b.geom
          )
    """, (state_fips, state_fips)).fetchall()

    total = 0
    with conn:
        for c in counties:
            conn.execute(
                "INSERT OR IGNORE INTO county_adjacency "
                "(county_geoid, neighbor_geoid) VALUES (?, ?)",
                (c["geoid"], c["geoid"]),
            )
            total += 1
        for pair in pairs:
            conn.execute(
                "INSERT OR IGNORE INTO county_adjacency "
                "(county_geoid, neighbor_geoid) VALUES (?, ?)",
                (pair["a_geoid"], pair["b_geoid"]),
            )
            total += 1

    logger.info(
        "County adjacency complete: %d rows (incl. self-loops)", total,
    )
    return total


def get_tracts_near_place(
    conn: sqlite3.Connection,
    place_id: str,
    buffer_miles: float = NEAR_BUFFER_MILES,
) -> list[dict]:
    """Look up pre-computed 'near' tract list for a named place anchor.

    Returns list of {tract_geoid, dist_meters} sorted ascending by distance.
    """
    rows = conn.execute(
        "SELECT tract_geoid, dist_meters "
        "FROM near_tract_map "
        "WHERE anchor_type = 'named_place' AND anchor_id = ? "
        "  AND buffer_miles = ? "
        "ORDER BY dist_meters",
        (place_id, buffer_miles),
    ).fetchall()
    return [dict(r) for r in rows]


def compute_admin_place_tract_mappings(
    conn: sqlite3.Connection,
    state_fips: str = "13",
) -> int:
    """Pre-compute the tract list for every GA admin place.

    Mirrors ``compute_place_tract_mappings`` but sourced from
    ``admin_geographies`` (geo_type='place') instead of ``named_places``.
    Output feeds compound directional operations on admin refs — e.g.,
    "southwest Atlanta" becomes (Atlanta city tracts) filtered by the
    SW half-plane relative to Atlanta's centroid.

    Strategy: intersect each place geometry with GA tracts via
    SpatiaLite's spatial index + ST_Intersects; record overlap_pct as a
    fraction of TRACT area within the place (useful for later filtering).
    """
    logger.info(
        "Computing admin place → tract mappings for state %s", state_fips,
    )

    places = conn.execute(
        "SELECT geoid, name FROM admin_geographies "
        "WHERE geo_type = 'place' AND state_fips = ? AND geom IS NOT NULL",
        (state_fips,),
    ).fetchall()

    if not places:
        logger.warning("No admin places found for state %s", state_fips)
        return 0

    logger.info("  Processing %d admin places", len(places))

    total_inserted = 0
    commit_every = 100

    for i, place in enumerate(places):
        geoid = place["geoid"]

        rows = conn.execute("""
            SELECT
                t.geoid AS tract_geoid,
                substr(t.geoid, 1, 5) AS county_geoid,
                ST_Area(Intersection(t.geom, ap.geom)) / ST_Area(t.geom)
                    AS tract_overlap_pct
            FROM admin_geographies t
            JOIN admin_geographies ap ON ap.geoid = ?
            WHERE t.geo_type = 'tract'
              AND t.state_fips = ?
              AND Intersects(t.geom, ap.geom)
              AND t.rowid IN (
                  SELECT rowid FROM SpatialIndex
                  WHERE f_table_name = 'admin_geographies'
                    AND search_frame = ap.geom
              )
        """, (geoid, state_fips)).fetchall()

        for r in rows:
            overlap = float(r["tract_overlap_pct"] or 0.0)
            # Keep only non-trivial overlaps — a tract touching the
            # place's boundary corner with < 1% area overlap isn't
            # meaningfully "inside" the place.
            if overlap < 0.01:
                continue
            if overlap >= 0.95:
                overlap_type = "contains"   # tract essentially inside place
            elif overlap >= 0.50:
                overlap_type = "majority"
            else:
                overlap_type = "intersects"
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO admin_place_tract_map "
                    "(admin_geoid, tract_geoid, county_geoid, "
                    " overlap_pct, overlap_type) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        geoid, r["tract_geoid"], r["county_geoid"],
                        round(overlap, 4), overlap_type,
                    ),
                )
                total_inserted += 1
            except sqlite3.IntegrityError:
                pass

        if (i + 1) % commit_every == 0:
            conn.commit()
            logger.info(
                "  [%d/%d] admin places, %d mappings",
                i + 1, len(places), total_inserted,
            )

    conn.commit()
    logger.info(
        "Admin place → tract mappings: %d rows across %d places",
        total_inserted, len(places),
    )
    return total_inserted


def get_tracts_for_admin_place(
    conn: sqlite3.Connection,
    admin_geoid: str,
) -> list[dict]:
    """Look up pre-computed tract list for an admin place (city)."""
    rows = conn.execute(
        "SELECT tract_geoid, county_geoid, overlap_pct, overlap_type "
        "FROM admin_place_tract_map "
        "WHERE admin_geoid = ? "
        "ORDER BY overlap_pct DESC",
        (admin_geoid,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_adjacent_counties(
    conn: sqlite3.Connection,
    county_geoid: str,
) -> list[str]:
    """Return county GEOIDs adjacent to (and including) the given county."""
    rows = conn.execute(
        "SELECT neighbor_geoid FROM county_adjacency "
        "WHERE county_geoid = ? ORDER BY neighbor_geoid",
        (county_geoid,),
    ).fetchall()
    return [r["neighbor_geoid"] for r in rows]
