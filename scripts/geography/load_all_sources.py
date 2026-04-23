"""Master gazetteer loader.

Loads all geographic data sources into the SpatiaLite gazetteer database
in dependency order. Each source is a standalone function so individual
sources can be reloaded without rebuilding everything.

Run order:
  1. TIGER national (states, counties, CBSAs, metro divisions, urban areas)
  2. TIGER Georgia (tracts, places, school districts)
  3. ARC statewide GeoJSON (named geographies)
  4. City of Atlanta neighborhoods
  5. OSM roads
  6. OSM landmarks
  7. GNIS populated places
  8. CBSA county crosswalk
  9. ZCTA-place relationship
  10. National gazetteer (major cities)
  11. Aliases
  12. Spatial pre-computation

Usage:
    python -m scripts.geography.load_all_sources
    python -m scripts.geography.load_all_sources --only tiger_national
    python -m scripts.geography.load_all_sources --skip precompute
    python -m scripts.geography.load_all_sources --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import sys
import time
import zipfile
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _geo_dir() -> Path:
    return _repo_root() / "data" / "geo"


def _raw_dir() -> Path:
    return _geo_dir() / "raw"


def _db_path() -> Path:
    return _geo_dir() / "gazetteer.db"


def _open_db() -> sqlite3.Connection:
    from scripts.geography.schema import open_spatialite
    return open_spatialite(_db_path())


def _count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return conn.execute(sql).fetchone()[0]


def _wkt_to_multipolygon(wkt: str) -> str:
    """Ensure geometry is MULTIPOLYGON for consistent storage."""
    if wkt.startswith("POLYGON"):
        return f"MULTIPOLYGON(({wkt[8:]}))"
    return wkt


def _read_shapefile(zip_path: Path):
    """Read a shapefile from a zip archive using geopandas."""
    import geopandas as gpd
    return gpd.read_file(f"zip://{zip_path}").to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# 1. TIGER National
# ---------------------------------------------------------------------------

def load_tiger_national(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load US states, counties, CBSAs, metro divisions, urban areas."""
    counts = {}
    tiger = raw / "tiger"

    # States
    if (tiger / "tl_2020_us_state.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_us_state.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                sf = str(row["STATEFP"]).zfill(2)
                geoid = sf
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, aland, "
                    " api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, 'state', ?, ?, ?, '', "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, row["NAME"], sf,
                     float(row.get("ALAND", 0) or 0),
                     f"state:{sf}", wkt),
                )
                n += 1
        counts["states"] = n
        logger.info("  States: %d", n)

    # Counties
    if (tiger / "tl_2020_us_county.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_us_county.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                sf = str(row["STATEFP"]).zfill(2)
                cf = str(row["COUNTYFP"]).zfill(3)
                geoid = f"{sf}{cf}"
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, county_fips, "
                    " lsad, aland, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, 'county', ?, ?, ?, ?, ?, ?, "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, row["NAMELSAD"], sf, cf,
                     str(row.get("LSAD", "")),
                     float(row.get("ALAND", 0) or 0),
                     f"county:{cf}", f"state:{sf}", wkt),
                )
                n += 1
        counts["counties"] = n
        logger.info("  Counties: %d", n)

    # CBSAs (metro and micropolitan statistical areas)
    if (tiger / "tl_2020_us_cbsa.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_us_cbsa.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                geoid = str(row["CBSAFP"]).zfill(5)
                name = row["NAMELSAD"]
                lsad = str(row.get("LSAD", ""))
                # M1 = Metropolitan, M2 = Micropolitan
                geo_type = "msa" if lsad == "M1" else "micro"
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, lsad, "
                    " aland, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, ?, '', ?, ?, ?, '', "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, name, geo_type, lsad,
                     float(row.get("ALAND", 0) or 0),
                     f"cbsa:{geoid}", wkt),
                )
                n += 1
        counts["cbsas"] = n
        logger.info("  CBSAs: %d", n)

    # Urban Areas
    if (tiger / "tl_2020_us_uac10.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_us_uac10.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                geoid = f"UA{str(row['UACE10']).zfill(5)}"
                name = row["NAME10"]
                uatype = str(row.get("UATYP10", ""))
                geo_type = "urbanized_area" if uatype == "U" else "urban_cluster"
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, "
                    " aland, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, ?, '', ?, '', '', "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, name, geo_type,
                     float(row.get("ALAND10", 0) or 0), wkt),
                )
                n += 1
        counts["urban_areas"] = n
        logger.info("  Urban areas: %d", n)

    return counts


# ---------------------------------------------------------------------------
# 2. TIGER Georgia
# ---------------------------------------------------------------------------

def load_tiger_georgia(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load Georgia tracts, places, and school districts."""
    counts = {}
    tiger = raw / "tiger"

    # Tracts
    if (tiger / "tl_2020_13_tract.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_13_tract.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                sf = str(row["STATEFP"]).zfill(2)
                cf = str(row["COUNTYFP"]).zfill(3)
                tc = str(row["TRACTCE"]).zfill(6)
                geoid = f"{sf}{cf}{tc}"
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, county_fips, "
                    " aland, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, 'tract', ?, ?, ?, ?, ?, "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, f"Census Tract {row['NAME']}",
                     sf, cf,
                     float(row.get("ALAND", 0) or 0),
                     f"tract:{tc}",
                     f"state:{sf}+county:{cf}", wkt),
                )
                n += 1
        counts["tracts"] = n
        logger.info("  Georgia tracts: %d", n)

    # Places
    if (tiger / "tl_2020_13_place.zip").exists():
        gdf = _read_shapefile(tiger / "tl_2020_13_place.zip")
        n = 0
        with conn:
            for _, row in gdf.iterrows():
                sf = str(row["STATEFP"]).zfill(2)
                pf = str(row["PLACEFP"]).zfill(5)
                geoid = f"{sf}{pf}"
                wkt = _wkt_to_multipolygon(row.geometry.wkt)
                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, place_fips, "
                    " lsad, aland, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, 'place', ?, ?, ?, ?, ?, ?, "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, row["NAME"], sf, pf,
                     str(row.get("LSAD", "")),
                     float(row.get("ALAND", 0) or 0),
                     f"place:{pf}", f"state:{sf}", wkt),
                )
                n += 1
        counts["places"] = n
        logger.info("  Georgia places: %d", n)

    # School districts
    school_files = [
        ("tl_2020_13_unsd.zip", "school_district_unified"),
        ("tl_2020_13_elsd.zip", "school_district_elementary"),
        ("tl_2020_13_scsd.zip", "school_district_secondary"),
    ]
    for fname, geo_type in school_files:
        fpath = tiger / fname
        if not fpath.exists():
            continue
        try:
            gdf = _read_shapefile(fpath)
            n = 0
            with conn:
                for _, row in gdf.iterrows():
                    sf = str(row["STATEFP"]).zfill(2)
                    geoid = str(row["GEOID"])
                    name = str(row.get("NAME", row.get("NAMELSAD", geoid)))
                    wkt = _wkt_to_multipolygon(row.geometry.wkt)
                    conn.execute(
                        "INSERT OR REPLACE INTO admin_geographies "
                        "(geoid, name, geo_type, state_fips, "
                        " aland, api_for_clause, api_in_clause, geom) "
                        "VALUES (?, ?, ?, ?, ?, '', '', "
                        "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                        (geoid, name, geo_type, sf,
                         float(row.get("ALAND", 0) or 0), wkt),
                    )
                    n += 1
            counts[geo_type] = n
            logger.info("  %s: %d", geo_type, n)
        except Exception as e:
            logger.warning("  %s failed: %s", fname, e)

    return counts


# ---------------------------------------------------------------------------
# 3. ARC Statewide GeoJSON
# ---------------------------------------------------------------------------

# SumLev → (geo_type, load_as)
# load_as: 'admin' = admin_geographies, 'named' = named_places, 'skip'
ARC_LEVELS = {
    "State":                ("state",            "admin"),
    "County":               ("county",           "admin"),
    "Tract":                ("tract",            "admin"),
    "City":                 ("place",            "admin"),
    "ZCTA":                 ("zcta",             "admin"),
    "SuperDistrict":        ("superdistrict",    "named"),
    "NSA":                  ("nsa",              "named"),
    "NPU":                  ("npu",              "named"),
    "BeltLineStatisticalSub": ("beltline_sub",   "named"),
    "HSSA":                 ("hssa",             "named"),
    "AAA":                  ("aaa",              "named"),
    "RC":                   ("rc",               "named"),
    # Single-polygon service areas — load as named for alias use
    "ARC21":                ("arc_region",       "named"),
    "ARWDB7":               ("arc_region",       "named"),
    "MetroWater15":         ("arc_region",       "named"),
    "UWGA13":               ("arc_region",       "named"),
    "BeltLineStatistical":  ("arc_region",       "named"),
    # Skip legislative/redistricting
    "CCDIST":               ("skip",             "skip"),
    "CCSUPDIST":            ("skip",             "skip"),
    "GAHouse":              ("skip",             "skip"),
    "GASenate":             ("skip",             "skip"),
    "Congress":             ("skip",             "skip"),
}

# Named place types and their metro context
ARC_METRO = {
    "SuperDistrict": "atlanta",
    "NSA": "atlanta",
    "NPU": "atlanta",
    "BeltLineStatisticalSub": "atlanta",
    "BeltLineStatistical": "atlanta",
    "HSSA": "atlanta",
}


def load_arc_statewide(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load ARC statewide GeoJSON into admin_geographies and named_places."""
    geojson_files = list(raw.glob("GeographicBoundaries2024statewide_*.geojson"))
    if not geojson_files:
        logger.warning("ARC statewide GeoJSON not found in %s", raw)
        return {}

    geojson_path = geojson_files[0]
    logger.info("  Loading ARC file: %s", geojson_path.name)

    with geojson_path.open() as f:
        data = json.load(f)

    features = data["features"]
    logger.info("  Total features: %d", len(features))

    admin_counts = defaultdict(int)
    named_counts = defaultdict(int)
    skipped = 0

    with conn:
        for feat in features:
            props = feat["properties"]
            geom = feat["geometry"]
            if not geom:
                continue

            sumlev = props.get("SumLev", "")
            level_info = ARC_LEVELS.get(sumlev)
            if not level_info:
                skipped += 1
                continue

            geo_type, load_as = level_info
            if load_as == "skip":
                skipped += 1
                continue

            geoid = str(props.get("GEOID", ""))
            name = str(props.get("Name", ""))
            wkt = _geojson_to_wkt(geom)
            if not wkt:
                continue

            if load_as == "admin":
                # Determine state_fips and county_fips from GEOID
                sf = _state_fips_from_geoid(geoid, geo_type)
                cf = _county_fips_from_geoid(geoid, geo_type)
                pf = _place_fips_from_geoid(geoid, geo_type)

                conn.execute(
                    "INSERT OR REPLACE INTO admin_geographies "
                    "(geoid, name, geo_type, state_fips, county_fips, "
                    " place_fips, api_for_clause, api_in_clause, geom) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, name, geo_type, sf, cf, pf,
                     _api_for(geoid, geo_type),
                     _api_in(geoid, geo_type),
                     wkt),
                )
                admin_counts[geo_type] += 1

            else:  # named_place
                metro = ARC_METRO.get(sumlev, "")
                conn.execute(
                    "INSERT OR REPLACE INTO named_places "
                    "(place_id, name, place_type, metro, state_fips, "
                    " source, geom) "
                    "VALUES (?, ?, ?, ?, '13', 'arc', "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (geoid, name, geo_type, metro, wkt),
                )
                named_counts[geo_type] += 1

    logger.info("  Admin geographies loaded: %s", dict(admin_counts))
    logger.info("  Named places loaded: %s", dict(named_counts))
    logger.info("  Skipped: %d", skipped)

    return {"admin": dict(admin_counts), "named": dict(named_counts)}


# ---------------------------------------------------------------------------
# 4. City of Atlanta Neighborhoods
# ---------------------------------------------------------------------------

def load_atl_neighborhoods(
    conn: sqlite3.Connection,
    raw: Path,
    neighborhood_csv: Optional[Path] = None,
) -> int:
    """Load City of Atlanta official neighborhood polygons."""
    geojson_path = raw / "Official_Neighborhoods_-_Open_Data.geojson"
    if not geojson_path.exists():
        logger.warning("ATL neighborhoods GeoJSON not found")
        return 0

    with geojson_path.open() as f:
        data = json.load(f)

    # Load optional CSV for extra attributes (OLDNAME etc.)
    csv_attrs = {}
    if neighborhood_csv and neighborhood_csv.exists():
        with neighborhood_csv.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                obj_id = int(row["OBJECTID"])
                csv_attrs[obj_id] = row

    n = 0
    with conn:
        for feat in data["features"]:
            props = feat["properties"]
            geom = feat["geometry"]
            if not geom:
                continue

            obj_id = props.get("OBJECTID")
            name = props.get("NAME", "")
            npu = props.get("NPU", "")
            old_name = props.get("OLDNAME", "")

            # Supplement from CSV if available
            if obj_id and obj_id in csv_attrs:
                csv_row = csv_attrs[obj_id]
                old_name = csv_row.get("OLDNAME", old_name) or old_name

            # Create a stable place_id
            place_id = f"ATL_NBH_{obj_id}"

            wkt = _geojson_to_wkt(geom)
            if not wkt:
                continue

            notes = f"NPU {npu}"
            if old_name and old_name != name:
                notes += f"; formerly '{old_name}'"

            conn.execute(
                "INSERT OR REPLACE INTO named_places "
                "(place_id, name, place_type, metro, state_fips, "
                " source, notes, geom) "
                "VALUES (?, ?, 'neighborhood', 'atlanta', '13', "
                "  'atl_opendata', ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (place_id, name, notes, wkt),
            )

            # Also store NPU linkage as extra attribute
            conn.execute(
                "INSERT OR IGNORE INTO geo_aliases "
                "(alias, target_id, target_type, target_name, "
                " alias_type, state_fips, confidence) "
                "VALUES (?, ?, 'named_place', ?, 'npu_member', '13', 0.8)",
                (f"NPU {npu} neighborhood", place_id, name),
            )

            # Old name as alias
            if old_name and old_name != name:
                conn.execute(
                    "INSERT OR IGNORE INTO geo_aliases "
                    "(alias, target_id, target_type, target_name, "
                    " alias_type, state_fips, confidence) "
                    "VALUES (?, ?, 'named_place', ?, 'historical', '13', 0.85)",
                    (old_name, place_id, name),
                )

            n += 1

    logger.info("  ATL neighborhoods: %d", n)
    return n


# ---------------------------------------------------------------------------
# 5. OSM Roads
# ---------------------------------------------------------------------------

# Ref patterns to match for each road we care about
ROAD_REF_MAP = {
    "I-20":   ["I 20", "I-20", "I20"],
    "I-75":   ["I 75", "I-75", "I75"],
    "I-85":   ["I 85", "I-85", "I85"],
    "I-285":  ["I 285", "I-285", "I285"],
    "I-575":  ["I 575", "I-575", "I575"],
    "I-675":  ["I 675", "I-675", "I675"],
    "GA-400": ["SR 400", "GA 400", "400"],
    "US-23":  ["US 23", "US-23"],
    "US-41":  ["US 41", "US-41"],
    "US-78":  ["US 78", "US-78"],
    "SR-316": ["SR 316", "GA 316"],
}

ROAD_METADATA = {
    "I-20":   ("Interstate 20",   "interstate"),
    "I-75":   ("Interstate 75",   "interstate"),
    "I-85":   ("Interstate 85",   "interstate"),
    "I-285":  ("Interstate 285",  "interstate"),
    "I-575":  ("Interstate 575",  "interstate"),
    "I-675":  ("Interstate 675",  "interstate"),
    "GA-400": ("Georgia 400",     "state_route"),
    "US-23":  ("Buford Highway",  "us_highway"),
    "US-41":  ("Cobb Parkway",    "us_highway"),
    "US-78":  ("Stone Mountain Freeway", "us_highway"),
    "SR-316": ("Georgia 316",     "state_route"),
}

# Georgia bounding box for filtering out-of-state segments
GA_BOUNDS = {"min_lat": 30.35, "max_lat": 35.0,
             "min_lon": -85.6, "max_lon": -80.75}


def _in_georgia(coords: list) -> bool:
    """Check if any coordinate in a linestring is within Georgia bounds."""
    for lon, lat in coords:
        if (GA_BOUNDS["min_lat"] <= lat <= GA_BOUNDS["max_lat"] and
                GA_BOUNDS["min_lon"] <= lon <= GA_BOUNDS["max_lon"]):
            return True
    return False


def load_osm_roads(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load Georgia road corridors from OSM GeoJSON, dissolved by road ID."""
    roads_path = raw / "osm" / "georgia_roads.geojson"
    if not roads_path.exists():
        logger.warning("OSM roads GeoJSON not found at %s", roads_path)
        return {}

    # Build reverse lookup: ref_pattern → road_id
    ref_to_road: dict[str, str] = {}
    for road_id, patterns in ROAD_REF_MAP.items():
        for pat in patterns:
            ref_to_road[pat.upper()] = road_id

    # Collect linestring segments per road
    road_segments: dict[str, list] = defaultdict(list)

    with roads_path.open() as f:
        data = json.load(f)

    for feat in data["features"]:
        props = feat["properties"]
        geom = feat["geometry"]
        if not geom or geom["type"] != "LineString":
            continue

        coords = geom["coordinates"]
        if not _in_georgia(coords):
            continue

        ref = str(props.get("ref", "")).upper().strip()
        road_id = ref_to_road.get(ref)
        if not road_id:
            # Try partial match
            for pattern, rid in ref_to_road.items():
                if pattern in ref:
                    road_id = rid
                    break
        if road_id:
            road_segments[road_id].append(coords)

    # Insert each road as a MULTILINESTRING
    n = 0
    with conn:
        for road_id, segments in road_segments.items():
            full_name, road_type = ROAD_METADATA.get(
                road_id, (road_id, "unknown")
            )
            # Build WKT MULTILINESTRING
            parts = []
            for seg in segments:
                coord_str = ", ".join(f"{c[0]} {c[1]}" for c in seg)
                parts.append(f"({coord_str})")
            wkt = f"MULTILINESTRING({', '.join(parts)})"

            conn.execute(
                "INSERT OR REPLACE INTO roads "
                "(road_id, name, full_name, road_type, state_fips, "
                " default_buffer_miles, geom) "
                "VALUES (?, ?, ?, ?, '13', 2.0, "
                "  CastToMultiLinestring(GeomFromText(?, 4326)))",
                (road_id, road_id, full_name, road_type, wkt),
            )
            n += 1
            logger.debug("  Road %s: %d segments", road_id, len(segments))

    logger.info("  OSM roads loaded: %d", n)

    # Simplify road geometries to collapse collinear segments. OSM imports
    # each road as many 2-point linestrings; without merging first, ST_Simplify
    # has nothing to remove (a 2-point line has no intermediate vertices).
    # ST_LineMerge fuses connected segments into longer linestrings, then
    # ST_Simplify drops redundant vertices. Downstream road-tract precompute
    # goes from hours to minutes once part/vertex counts drop.
    before = {
        rid: (parts, verts)
        for rid, parts, verts in conn.execute(
            "SELECT road_id, NumGeometries(geom), ST_NPoints(geom) FROM roads"
        )
    }
    with conn:
        conn.execute(
            "UPDATE roads SET geom = "
            "CastToMultiLinestring(ST_Simplify(ST_LineMerge(geom), 0.001))"
        )
    after = {
        rid: (parts, verts)
        for rid, parts, verts in conn.execute(
            "SELECT road_id, NumGeometries(geom), ST_NPoints(geom) FROM roads"
        )
    }
    changes = []
    for rid in sorted(before):
        b_parts, b_verts = before[rid]
        a_parts, a_verts = after.get(rid, (0, 0))
        if (b_parts, b_verts) != (a_parts, a_verts):
            changes.append(
                f"{rid} {b_parts}\u2192{a_parts} parts "
                f"({b_verts}\u2192{a_verts} verts)"
            )
    if changes:
        logger.info("  Simplified road geometries: %s", ", ".join(changes))
    else:
        logger.info("  Simplified road geometries: no change")

    missing = set(ROAD_REF_MAP.keys()) - set(road_segments.keys())
    if missing:
        logger.warning("  Roads not found in OSM data: %s", sorted(missing))
    return {"roads": n, "missing": sorted(missing)}


# ---------------------------------------------------------------------------
# 6. OSM Landmarks
# ---------------------------------------------------------------------------

# Landmark types to keep and their category labels
LANDMARK_KEEP = {
    "aerodrome":        "airport",
    "terminal":         "airport",
    "university":       "university",
    "college":          "college",
    "school":           "school",
    "hospital":         "hospital",
    "clinic":           "hospital",
    "stadium":          "stadium",
    "arts_centre":      "arts",
    "theatre":          "arts",
    "library":          "library",
    "community_centre": "arts",
    "courthouse":       "government",
    "townhall":         "government",
    "prison":           "correctional",
    "police":           "government",
    "fire_station":     "government",
    "military":         "military",
    "park":             "park",
    "garden":           "park",
    "nature_reserve":   "park",
    "museum":           "museum",
    "attraction":       "attraction",
    "zoo":              "attraction",
    "aquarium":         "attraction",
    "monument":         "monument",
    "memorial":         "monument",
    "place_of_worship": "place_of_worship",
    "mall":             "mall",
    "neighbourhood":    "neighborhood",
    "suburb":           "neighborhood",
    "quarter":          "neighborhood",
}

# Parks to skip — too small or generic
PARK_SKIP_KEYWORDS = {
    "monument", "memorial", "rest area", "parking",
    "trailhead", "wildlife refuge"  # keep big ones
}


def _landmark_type_from_props(props: dict) -> Optional[str]:
    """Pick the first matching landmark category from OSM tag keys."""
    # Priority order — specific beats generic.
    for tag_key in (
        "aeroway", "leisure", "amenity", "tourism", "historic",
        "place", "shop", "building", "landuse",
    ):
        value = props.get(tag_key)
        if value and value in LANDMARK_KEEP:
            return LANDMARK_KEEP[value]
    return None


def _coords_to_wkt_polygon(geom: dict) -> Optional[str]:
    """Convert a GeoJSON Point / Polygon / MultiPolygon to MULTIPOLYGON WKT.

    Points are expanded to ~100m boxes so they can sit in a MULTIPOLYGON
    column. Polygons/MultiPolygons pass through (rings preserved).
    Returns None for unsupported geometry types (Line, MultiLine, etc.).
    """
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point":
        lon, lat = coords
        d = 0.001  # ~100m
        return (
            f"MULTIPOLYGON((("
            f"{lon-d} {lat-d}, {lon+d} {lat-d}, "
            f"{lon+d} {lat+d}, {lon-d} {lat+d}, "
            f"{lon-d} {lat-d}"
            f")))"
        )
    if gtype == "Polygon":
        rings = []
        for ring in coords:
            ring_str = ", ".join(f"{c[0]} {c[1]}" for c in ring)
            rings.append(f"({ring_str})")
        return f"MULTIPOLYGON(({', '.join(rings)}))"
    if gtype == "MultiPolygon":
        polys = []
        for poly in coords:
            rings = []
            for ring in poly:
                ring_str = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({ring_str})")
            polys.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polys)})"
    return None


def load_osm_landmarks(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load named Georgia landmarks into named_places.

    Prefers ``georgia_landmarks_v2.geojson`` (produced by
    ``refresh_osm_landmarks``) which includes polygons for landmarks
    like Piedmont Park, Mercedes-Benz Stadium, and Hartsfield-Jackson.
    Falls back to the points-only ``georgia_landmarks.geojson`` if v2
    isn't present.

    Returns ``{"inserted": N, "skipped_geom": N, "by_type": {...}}``.
    """
    v2_path = raw / "osm" / "georgia_landmarks_v2.geojson"
    legacy_path = raw / "osm" / "georgia_landmarks.geojson"
    landmarks_path = v2_path if v2_path.exists() else legacy_path
    if not landmarks_path.exists():
        logger.warning("OSM landmarks GeoJSON not found")
        return {"inserted": 0, "skipped_geom": 0, "by_type": {}}
    logger.info("  OSM landmarks source: %s", landmarks_path.name)

    with landmarks_path.open() as f:
        data = json.load(f)

    from collections import Counter
    import hashlib
    by_type: Counter = Counter()
    skipped_geom = 0
    inserted = 0
    # Track duplicates by (name_lower, type) — keep the best-geometry
    # representation when OSM emits the same landmark as both a
    # LineString outline and a MultiPolygon face.
    _GEOM_PRIORITY = {"MultiPolygon": 0, "Polygon": 1, "Point": 2, "LineString": 3}
    seen: dict[tuple[str, str], tuple[str, int]] = {}

    def _stable_place_id(name: str, geom: dict) -> str:
        """Content-based ID so repeated loads of the same data stay stable."""
        coords = json.dumps(geom.get("coordinates"), sort_keys=True)[:200]
        h = hashlib.sha1(
            f"{name.lower()}|{geom.get('type')}|{coords}".encode("utf-8")
        ).hexdigest()[:12]
        return f"LANDMARK_{h}"

    # Clean up prior OSM landmark rows to avoid stale/orphaned entries
    # when the source file changes. We only delete OSM-sourced rows.
    conn.execute(
        "DELETE FROM geo_aliases WHERE target_type = 'named_place' "
        "AND target_id IN ("
        "  SELECT place_id FROM named_places "
        "  WHERE source = 'osm' AND place_id LIKE 'LANDMARK_%'"
        ")"
    )
    conn.execute(
        "DELETE FROM named_places "
        "WHERE source = 'osm' AND place_id LIKE 'LANDMARK_%'"
    )
    conn.commit()

    # First pass: dedupe, pick best-geometry representation per (name, type)
    for feat in data["features"]:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        name = (props.get("name") or "").strip()
        if not name:
            continue
        lm_type = _landmark_type_from_props(props)
        if not lm_type:
            continue
        if lm_type == "park":
            name_lower = name.lower()
            if any(kw in name_lower for kw in PARK_SKIP_KEYWORDS):
                continue
        gtype = geom.get("type")
        priority = _GEOM_PRIORITY.get(gtype, 99)
        if priority == 99:
            continue  # unsupported geometry type
        dedupe_key = (name.lower(), lm_type)
        existing = seen.get(dedupe_key)
        if existing is None or priority < existing[1]:
            seen[dedupe_key] = (json.dumps(feat), priority)

    # Second pass: insert the best representatives
    with conn:
        for (name_lower, lm_type), (feat_json, _) in seen.items():
            feat = json.loads(feat_json)
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            name = (props.get("name") or "").strip()

            wkt = _coords_to_wkt_polygon(geom)
            if wkt is None:
                skipped_geom += 1
                continue

            place_id = _stable_place_id(name, geom)
            notes_bits = []
            for k in ("wikipedia", "wikidata", "website"):
                v = props.get(k)
                if v:
                    notes_bits.append(f"{k}={v}")

            conn.execute(
                "INSERT OR REPLACE INTO named_places "
                "(place_id, name, place_type, metro, state_fips, "
                " source, notes, geom) "
                "VALUES (?, ?, ?, '', '13', 'osm', ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (place_id, name, lm_type, "; ".join(notes_bits), wkt),
            )
            conn.execute(
                "INSERT OR IGNORE INTO geo_aliases "
                "(alias, target_id, target_type, target_name, "
                " alias_type, state_fips, confidence) "
                "VALUES (?, ?, 'named_place', ?, 'exact', '13', 1.0)",
                (name, place_id, name),
            )
            inserted += 1
            by_type[lm_type] += 1

    logger.info("  OSM landmarks loaded: %d (%d skipped unsupported geom)",
                inserted, skipped_geom)
    return {
        "inserted": inserted,
        "skipped_geom": skipped_geom,
        "by_type": dict(by_type),
    }


# ---------------------------------------------------------------------------
# 6b. Atlanta BeltLine — ABI Planning Area subareas + trails
# ---------------------------------------------------------------------------

def _linestring_to_multi_wkt(geom: dict) -> Optional[str]:
    """Normalize a LineString / MultiLineString GeoJSON to WKT MultiLineString."""
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    if gtype == "LineString":
        line_str = ", ".join(f"{c[0]} {c[1]}" for c in coords)
        return f"MULTILINESTRING(({line_str}))"
    if gtype == "MultiLineString":
        parts = []
        for line in coords:
            line_str = ", ".join(f"{c[0]} {c[1]}" for c in line)
            parts.append(f"({line_str})")
        return f"MULTILINESTRING({', '.join(parts)})"
    return None


def _split_slash_names(name: str) -> list[str]:
    """Split 'A/B/C' or 'A / B / C' into ['A', 'B', 'C'], trimming."""
    out = []
    for piece in name.split("/"):
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out


def load_abi_beltline(conn: sqlite3.Connection, raw: Path) -> dict:
    """Load Atlanta BeltLine subareas (polygons) and trails (linear buffers).

    Uses ABI open data files:
      - ABI_BPA_Subareas_public.geojson — Planning Area subarea polygons
      - ABI_Trails_public.geojson       — named trail segments

    Trails are grouped by name, dissolved into MultiLineStrings, and
    buffered ~75 m into small polygons so they live in the same
    MULTIPOLYGON geom column as other named_places. That lets the
    resolver treat them uniformly via place_tract_map.
    """
    sub_path = raw / "ABI_BPA_Subareas_public.geojson"
    trl_path = raw / "ABI_Trails_public.geojson"

    if not sub_path.exists() and not trl_path.exists():
        logger.warning("ABI BeltLine files not found in %s", raw)
        return {"subareas": 0, "trails": 0}

    n_sub = 0
    n_trail = 0
    alias_count = 0

    # ---- Subareas ----
    if sub_path.exists():
        with sub_path.open() as f:
            subdata = json.load(f)

        with conn:
            for feat in subdata["features"]:
                props = feat.get("properties") or {}
                geom = feat.get("geometry") or {}

                sa_int = props.get("sa_int")
                name_full = (props.get("name_full") or "").strip()
                if not name_full:
                    continue

                place_id = f"BPA_SUB_{int(sa_int):02d}" if sa_int else f"BPA_SUB_{hash(name_full) & 0xFFFF:04x}"
                study_group = props.get("studygroup") or ""
                description = (props.get("description") or "").strip()

                wkt = _coords_to_wkt_polygon(geom)
                if wkt is None:
                    continue

                conn.execute(
                    "INSERT OR REPLACE INTO named_places "
                    "(place_id, name, place_type, metro, state_fips, "
                    " description, source, notes, geom) "
                    "VALUES (?, ?, 'beltline_sub', 'atlanta', '13', "
                    "        ?, 'abi', ?, "
                    "        CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (
                        place_id, name_full,
                        description,
                        f"sa_int={sa_int}; studygroup={study_group}",
                        wkt,
                    ),
                )
                n_sub += 1

                # Generate aliases
                aliases: set[str] = set()
                aliases.add(name_full)
                # Split "X/Y/Z" → each piece individually
                aliases.update(_split_slash_names(name_full))
                # Subarea-number aliases
                if sa_int is not None:
                    aliases.add(f"BeltLine Subarea {int(sa_int)}")
                    aliases.add(f"BPA Subarea {int(sa_int)}")
                    aliases.add(f"Subarea {int(sa_int)}")
                # Study group combo ("Westside BeltLine")
                if study_group:
                    aliases.add(f"{study_group} BeltLine")
                    aliases.add(f"BeltLine {study_group}")
                # Legacy names
                for legacy_key in ("legacy_name", "legacy_name2"):
                    legacy = (props.get(legacy_key) or "").strip()
                    if legacy:
                        aliases.add(legacy)
                        aliases.update(_split_slash_names(legacy))

                for alias in aliases:
                    if not alias or alias == place_id:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO geo_aliases "
                        "(alias, target_id, target_type, target_name, "
                        " alias_type, state_fips, confidence) "
                        "VALUES (?, ?, 'named_place', ?, ?, '13', 0.95)",
                        (
                            alias, place_id, name_full,
                            "exact" if alias == name_full else "common",
                        ),
                    )
                    alias_count += 1

    # ---- Trails (group by name, merge, buffer) ----
    if trl_path.exists():
        with trl_path.open() as f:
            trldata = json.load(f)

        # Group line segments by trail name.
        from collections import defaultdict
        by_name: dict[str, list[list]] = defaultdict(list)
        meta_by_name: dict[str, dict] = {}
        for feat in trldata["features"]:
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            trail_name = (props.get("name") or "").strip()
            if not trail_name:
                continue
            gtype = geom.get("type")
            coords = geom.get("coordinates") or []
            if gtype == "LineString":
                by_name[trail_name].append(coords)
            elif gtype == "MultiLineString":
                by_name[trail_name].extend(coords)
            # Keep first meta block per name
            meta_by_name.setdefault(trail_name, props)

        with conn:
            for trail_name, segments in by_name.items():
                if not segments:
                    continue
                props = meta_by_name.get(trail_name, {})
                wkt_ml = _linestring_to_multi_wkt(
                    {"type": "MultiLineString", "coordinates": segments}
                )
                if wkt_ml is None:
                    continue

                place_id = f"BPA_TRAIL_{trail_name.replace(' ', '_').upper()}"
                display = f"BeltLine {trail_name} Trail"
                description = (props.get("description") or "").strip()
                notes_bits = []
                for k in ("extent", "status2", "miles"):
                    v = props.get(k)
                    if v:
                        notes_bits.append(f"{k}={v}")

                # Buffer the line ~75 m in UTM to make a polygon.
                conn.execute(
                    """
                    INSERT OR REPLACE INTO named_places
                        (place_id, name, place_type, metro, state_fips,
                         description, source, notes, geom)
                    VALUES (
                        ?, ?, 'beltline_trail', 'atlanta', '13',
                        ?, 'abi', ?,
                        CastToMultiPolygon(
                            Transform(
                                Buffer(
                                    Transform(GeomFromText(?, 4326), 26916),
                                    75.0
                                ),
                                4326
                            )
                        )
                    )
                    """,
                    (
                        place_id, display,
                        description,
                        "; ".join(notes_bits),
                        wkt_ml,
                    ),
                )
                n_trail += 1

                # Aliases
                trail_aliases = {
                    display,
                    trail_name,
                    f"BeltLine {trail_name}",
                    f"{trail_name} BeltLine",
                    f"{trail_name} Trail",
                    f"BeltLine {trail_name} Segment",
                    f"Atlanta BeltLine {trail_name}",
                    f"Atlanta BeltLine {trail_name} Trail",
                }
                for alias in trail_aliases:
                    conn.execute(
                        "INSERT OR IGNORE INTO geo_aliases "
                        "(alias, target_id, target_type, target_name, "
                        " alias_type, state_fips, confidence) "
                        "VALUES (?, ?, 'named_place', ?, "
                        "        CASE WHEN ?=? THEN 'exact' ELSE 'common' END, "
                        "        '13', 0.95)",
                        (alias, place_id, display, alias, display),
                    )
                    alias_count += 1

    logger.info(
        "  BeltLine: %d subareas, %d trails, %d aliases",
        n_sub, n_trail, alias_count,
    )
    return {"subareas": n_sub, "trails": n_trail, "aliases": alias_count}


# ---------------------------------------------------------------------------
# 7. GNIS Populated Places
# ---------------------------------------------------------------------------

GNIS_KEEP_CLASSES = {"Populated Place", "Civil", "Military", "Area"}


def load_gnis(conn: sqlite3.Connection, raw: Path) -> int:
    """Load GNIS Georgia named places (populated places and military)."""
    gnis_path = raw / "DomesticNames_GA.txt"
    if not gnis_path.exists():
        logger.warning("GNIS file not found at %s", gnis_path)
        return 0

    n = 0
    with gnis_path.open(encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f, delimiter="|")
        with conn:
            for row in reader:
                if row["state_name"] != "Georgia":
                    continue
                if row["feature_class"] not in GNIS_KEEP_CLASSES:
                    continue

                name = row["feature_name"].strip()
                if not name:
                    continue

                try:
                    lat = float(row["prim_lat_dec"])
                    lon = float(row["prim_long_dec"])
                except (ValueError, KeyError):
                    continue

                if lat == 0.0 and lon == 0.0:
                    continue

                feature_id = row["feature_id"]
                place_id = f"GNIS_{feature_id}"
                feature_class = row["feature_class"]
                county = row.get("county_name", "")

                # Small bounding box around point
                d = 0.005  # ~500m
                wkt = (f"MULTIPOLYGON((({lon-d} {lat-d}, {lon+d} {lat-d}, "
                       f"{lon+d} {lat+d}, {lon-d} {lat+d}, {lon-d} {lat-d})))")

                geo_type = {
                    "Populated Place": "populated_place",
                    "Civil": "civil",
                    "Military": "military",
                    "Area": "area",
                }.get(feature_class, "populated_place")

                notes = f"{county} County"

                conn.execute(
                    "INSERT OR REPLACE INTO named_places "
                    "(place_id, name, place_type, state_fips, "
                    " source, notes, geom) "
                    "VALUES (?, ?, ?, '13', 'gnis', ?, "
                    "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                    (place_id, name, geo_type, notes, wkt),
                )

                conn.execute(
                    "INSERT OR IGNORE INTO geo_aliases "
                    "(alias, target_id, target_type, target_name, "
                    " alias_type, state_fips, confidence) "
                    "VALUES (?, ?, 'named_place', ?, 'exact', '13', 0.9)",
                    (name, place_id, name),
                )
                n += 1

    logger.info("  GNIS places: %d", n)
    return n


# ---------------------------------------------------------------------------
# 8. CBSA County Crosswalk
# ---------------------------------------------------------------------------

def load_cbsa_crosswalk(conn: sqlite3.Connection, raw: Path) -> int:
    """Load CBSA→county relationships from list1_2020.xlsx.

    Stores county membership in a new cbsa_counties table and
    enriches CBSA aliases with proper MSA names.
    """
    # Ensure the table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cbsa_counties (
            cbsa_geoid  TEXT NOT NULL,
            cbsa_name   TEXT NOT NULL,
            cbsa_type   TEXT NOT NULL,
            county_geoid TEXT NOT NULL,
            county_name  TEXT NOT NULL,
            state_fips   TEXT NOT NULL,
            is_central   INTEGER DEFAULT 0,
            PRIMARY KEY (cbsa_geoid, county_geoid)
        )
    """)
    conn.commit()

    xlsx_path = raw / "list1_2020.xlsx"
    if not xlsx_path.exists():
        xlsx_path = raw / "list1_2020.xls"
    if not xlsx_path.exists():
        logger.warning("CBSA crosswalk not found")
        return 0

    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        rows = list(ws.values)
    except Exception as e:
        logger.warning("Could not read CBSA crosswalk: %s", e)
        return 0

    # Find header row — skip title rows and blank rows
    # The actual data header contains numeric CBSA codes
    # Look for row containing "CBSA" OR "cbsa" OR row where first non-empty
    # cell looks like a 5-digit code (data row) — header is just before it
    header_row = None
    for i, row in enumerate(rows):
        if not row:
            continue
        row_str = " ".join(str(c) for c in row if c is not None)
        # Header row contains "CBSA" and "County" and "State"
        if ("CBSA" in row_str or "cbsa" in row_str.lower()) and \
           ("County" in row_str or "county" in row_str.lower()):
            header_row = i
            break
        # Fallback: look for row with "Metropolitan" or "Metropolitan Division"
        if "Metropolitan" in row_str or "Micropolitan" in row_str:
            header_row = i
            break

    if header_row is None:
        # Last resort: print first 5 rows to help debug
        logger.warning("Could not find header row. First 5 rows:")
        for i, row in enumerate(rows[:5]):
            logger.warning("  Row %d: %s", i, row)
        return 0

    headers = [str(c).strip() if c else "" for c in rows[header_row]]

    def get_col(row, name):
        for i, h in enumerate(headers):
            if name.lower() in h.lower() and i < len(row):
                val = row[i]
                return str(val).strip() if val is not None else ""
        return ""

    n = 0
    with conn:
        for row in rows[header_row + 1:]:
            if not row or not any(row):
                continue

            cbsa_code = get_col(row, "CBSA Code")
            cbsa_title = get_col(row, "CBSA Title")
            metro_type = get_col(row, "Metropolitan")
            county_fips = get_col(row, "FIPS County")
            state_fips = get_col(row, "FIPS State")
            county_name = get_col(row, "County/County")
            central = get_col(row, "Central/Outlying")

            if not cbsa_code or not county_fips:
                continue

            # Pad to standard lengths
            cbsa_code = str(cbsa_code).zfill(5)
            state_fips = str(state_fips).zfill(2)
            county_fips_full = f"{state_fips}{str(county_fips).zfill(3)}"

            cbsa_type = "msa" if "Metropolitan" in metro_type else "micro"
            is_central = 1 if "Central" in central else 0

            conn.execute(
                "INSERT OR REPLACE INTO cbsa_counties "
                "(cbsa_geoid, cbsa_name, cbsa_type, county_geoid, "
                " county_name, state_fips, is_central) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cbsa_code, cbsa_title, cbsa_type, county_fips_full,
                 county_name, state_fips, is_central),
            )
            n += 1

    logger.info("  CBSA-county relationships: %d", n)

    # Add MSA name aliases
    msas = conn.execute(
        "SELECT DISTINCT cbsa_geoid, cbsa_name FROM cbsa_counties"
    ).fetchall()
    alias_count = 0
    with conn:
        for cbsa_geoid, cbsa_name in msas:
            # Full name alias
            conn.execute(
                "INSERT OR IGNORE INTO geo_aliases "
                "(alias, target_id, target_type, target_name, "
                " alias_type, confidence) "
                "VALUES (?, ?, 'admin', ?, 'exact', 1.0)",
                (cbsa_name, cbsa_geoid, cbsa_name),
            )
            # Short name — strip state suffix "Atlanta-Sandy Springs-Alpharetta, GA"
            short = cbsa_name.split(",")[0].strip()
            if short != cbsa_name:
                conn.execute(
                    "INSERT OR IGNORE INTO geo_aliases "
                    "(alias, target_id, target_type, target_name, "
                    " alias_type, confidence) "
                    "VALUES (?, ?, 'admin', ?, 'common', 0.9)",
                    (short, cbsa_geoid, cbsa_name),
                )
            alias_count += 1

    logger.info("  CBSA aliases added: %d", alias_count)
    return n


# ---------------------------------------------------------------------------
# 9. ZCTA-Place Relationship
# ---------------------------------------------------------------------------

def load_zcta_place_rel(conn: sqlite3.Connection, raw: Path) -> int:
    """Load ZCTA→place relationship for primary city lookup."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS zcta_place_rel (
            zcta5        TEXT NOT NULL,
            place_geoid  TEXT NOT NULL,
            state_fips   TEXT NOT NULL,
            zpop_pct     REAL,
            zarea_pct    REAL,
            PRIMARY KEY (zcta5, place_geoid)
        )
    """)
    conn.commit()

    rel_path = raw / "zcta_place_rel_2020.txt"
    if not rel_path.exists():
        logger.warning("ZCTA-place relationship file not found")
        return 0

    n = 0
    with rel_path.open() as f:
        reader = csv.DictReader(f)
        with conn:
            for row in reader:
                state = row.get("STATE", "").zfill(2)
                # Load all states — useful for national CBSA queries
                zcta = row.get("ZCTA5", "").strip()
                place_geoid = row.get("GEOID", "").strip()
                if not zcta or not place_geoid:
                    continue
                try:
                    zpop_pct = float(row.get("ZPOPPCT", 0) or 0)
                    zarea_pct = float(row.get("ZAREAPCT", 0) or 0)
                except ValueError:
                    zpop_pct = zarea_pct = 0.0

                conn.execute(
                    "INSERT OR REPLACE INTO zcta_place_rel "
                    "(zcta5, place_geoid, state_fips, zpop_pct, zarea_pct) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (zcta, place_geoid, state, zpop_pct, zarea_pct),
                )
                n += 1

    logger.info("  ZCTA-place relationships: %d", n)
    return n


# ---------------------------------------------------------------------------
# 10. National Gazetteer — Major Cities
# ---------------------------------------------------------------------------

MAJOR_CITY_POP_THRESHOLD = 50_000


def load_major_cities(conn: sqlite3.Connection, raw: Path) -> int:
    """Extract major US cities (pop > 50K) from national gazetteer.

    Handles both zipped and plain TSV formats of the Census gazetteer file.
    """
    gaz_path = raw / "tiger" / "2020_gaz_place_national.zip"
    if not gaz_path.exists():
        logger.warning("National gazetteer not found")
        return 0

    # Try as zip first, fall back to plain text
    def _read_rows(path: Path):
        try:
            with zipfile.ZipFile(path) as z:
                fname = z.namelist()[0]
                with z.open(fname) as f:
                    yield from csv.DictReader(
                        TextIOWrapper(f, encoding="utf-8"), delimiter="\t"
                    )
        except zipfile.BadZipFile:
            # File is actually a plain TSV
            with path.open(encoding="utf-8-sig") as f:
                yield from csv.DictReader(f, delimiter="\t")

    n = 0
    try:
        with conn:
            for row in _read_rows(gaz_path):
                try:
                    pop = int(row.get("POP10", 0) or 0)
                except ValueError:
                    continue
                if pop < MAJOR_CITY_POP_THRESHOLD:
                    continue

                geoid = row.get("GEOID", "").strip()
                name = row.get("NAME", "").strip()
                state = row.get("USPS", "").strip()
                if not geoid or not name:
                    continue

                for alias in [
                    name,
                    f"{name}, {state}",
                    f"{name} {state}",
                ]:
                    conn.execute(
                        "INSERT OR IGNORE INTO geo_aliases "
                        "(alias, target_id, target_type, target_name, "
                        " alias_type, confidence) "
                        "VALUES (?, ?, 'admin', ?, 'exact', 0.95)",
                        (alias, geoid, name),
                    )
                n += 1
    except Exception as e:
        logger.error("Failed to load major cities: %s", e)
        return 0

    logger.info("  Major city aliases: %d cities", n)
    return n


# ---------------------------------------------------------------------------
# 10b. Curated aliases (nicknames / abbreviations / historical names)
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Case-fold and normalize Unicode dashes / spacing for name matching.

    OSM commonly stores names with en-dashes (U+2013) where users (and
    our curated aliases YAML) use ASCII hyphens, e.g., "Hartsfield–Jackson"
    vs "Hartsfield-Jackson". Normalizing both sides lets curated aliases
    match without maintaining two spellings.
    """
    if not name:
        return ""
    # Replace en-dash, em-dash, figure-dash, minus with ASCII hyphen
    for bad in ("\u2013", "\u2014", "\u2012", "\u2212"):
        name = name.replace(bad, "-")
    # Collapse internal whitespace
    return " ".join(name.split()).lower()


def _find_target(
    conn: sqlite3.Connection,
    match_name: str,
    match_type: str,
    match_subtype: Optional[str] = None,
    state_fips: str = "13",
) -> Optional[tuple[str, str]]:
    """Look up a gazetteer entity by name + type. Returns (target_id, target_name).

    Match is case-insensitive and dash-normalized. If match_subtype is
    provided it narrows via place_type / geo_type / road_type. When
    multiple rows match across states, prefers the supplied state_fips.
    """
    needle = _normalize_name(match_name)

    def _filter(rows: list) -> Optional[tuple[str, str]]:
        for r in rows:
            if _normalize_name(r[1]) == needle:
                return r[0], r[1]
        return None

    if match_type == "named_place":
        if match_subtype:
            rows = conn.execute(
                "SELECT place_id, name FROM named_places "
                "WHERE place_type = ? AND state_fips = ? "
                "ORDER BY place_id",
                (match_subtype, state_fips),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT place_id, name FROM named_places "
                "WHERE state_fips = ? ORDER BY place_id",
                (state_fips,),
            ).fetchall()
        return _filter(rows)

    if match_type == "admin":
        if match_subtype:
            rows = conn.execute(
                "SELECT geoid, name FROM admin_geographies "
                "WHERE geo_type = ? AND state_fips = ? "
                "ORDER BY geoid",
                (match_subtype, state_fips),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT geoid, name FROM admin_geographies "
                "WHERE state_fips = ? ORDER BY geoid",
                (state_fips,),
            ).fetchall()
        return _filter(rows)

    if match_type == "road":
        rows = conn.execute(
            "SELECT road_id, name FROM roads "
            "WHERE state_fips = ?",
            (state_fips,),
        ).fetchall()
        return _filter(rows)

    return None


def load_curated_aliases(conn: sqlite3.Connection, raw: Path) -> dict:
    """Attach curated aliases (nicknames, abbreviations, historical names)
    to existing gazetteer entities.

    Reads ``data/geo/raw/curated_aliases.yaml``. Each entry specifies an
    alias plus a (match_name, match_type, match_subtype, state_fips)
    target selector. Missing targets are logged and skipped — curated
    aliases are best-effort.
    """
    import yaml

    path = raw / "curated_aliases.yaml"
    if not path.exists():
        logger.warning("curated_aliases.yaml not found at %s", path)
        return {"inserted": 0, "missing": 0}

    with path.open() as f:
        doc = yaml.safe_load(f)
    entries = doc.get("aliases", [])
    if not entries:
        return {"inserted": 0, "missing": 0}

    inserted = 0
    missing = 0
    missing_samples: list[str] = []

    with conn:
        for entry in entries:
            alias = (entry.get("alias") or "").strip()
            match_name = (entry.get("match_name") or "").strip()
            match_type = entry.get("match_type")
            match_subtype = entry.get("match_subtype")
            state_fips = entry.get("state_fips", "13")
            alias_type = entry.get("alias_type", "common")
            confidence = float(entry.get("confidence", 0.95))

            if not alias or not match_name or match_type not in (
                "named_place", "admin", "road",
            ):
                continue

            target = _find_target(
                conn, match_name, match_type,
                match_subtype=match_subtype, state_fips=state_fips,
            )
            if target is None:
                missing += 1
                if len(missing_samples) < 15:
                    missing_samples.append(
                        f"{alias!r} → {match_name!r} "
                        f"({match_type}/{match_subtype or '*'})"
                    )
                continue

            target_id, target_name = target
            conn.execute(
                "INSERT OR IGNORE INTO geo_aliases "
                "(alias, target_id, target_type, target_name, "
                " alias_type, state_fips, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    alias, target_id, match_type, target_name,
                    alias_type, state_fips, confidence,
                ),
            )
            inserted += 1

    logger.info(
        "  Curated aliases: %d attached, %d missing targets",
        inserted, missing,
    )
    if missing_samples:
        logger.info("    missing examples:")
        for s in missing_samples:
            logger.info("      %s", s)
    return {"inserted": inserted, "missing": missing}


# ---------------------------------------------------------------------------
# 11. Build Aliases
# ---------------------------------------------------------------------------

def build_aliases(conn: sqlite3.Connection) -> dict:
    """Generate algorithmic aliases for all administrative geographies."""
    from scripts.geography.alias_builder import (
        build_admin_aliases,
        load_curated_county_aliases,
        load_curated_place_aliases,
        load_road_aliases,
        load_named_place_aliases,
        rebuild_fts_index,
    )

    counts = {}
    counts["admin"] = build_admin_aliases(conn)
    counts["curated_county"] = load_curated_county_aliases(conn)
    counts["curated_place"] = load_curated_place_aliases(conn)
    counts["road"] = load_road_aliases(conn)
    counts["named_place"] = load_named_place_aliases(conn)

    rebuild_fts_index(conn)
    logger.info("  Aliases: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# 12. Spatial Pre-computation
# ---------------------------------------------------------------------------

def run_precompute(conn: sqlite3.Connection) -> dict:
    """Pre-compute place→tract, road→tract, near-tract, and county-adjacency."""
    from scripts.geography.spatial_precompute import (
        compute_place_tract_mappings,
        compute_road_tract_mappings,
        compute_near_tract_mappings,
        compute_admin_near_tract_mappings,
        compute_admin_place_tract_mappings,
        compute_county_adjacency,
    )
    counts = {}
    counts["place_tract"] = compute_place_tract_mappings(conn)
    counts["road_tract"] = compute_road_tract_mappings(conn)
    counts["near_tract"] = compute_near_tract_mappings(conn)
    counts["admin_near_tract"] = compute_admin_near_tract_mappings(conn)
    counts["admin_place_tract"] = compute_admin_place_tract_mappings(conn)
    counts["county_adjacency"] = compute_county_adjacency(conn)
    logger.info("  Pre-computed: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _geojson_to_wkt(geom: dict) -> Optional[str]:
    """Convert GeoJSON geometry dict to WKT MULTIPOLYGON string."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        rings = []
        for ring in coords:
            ring_str = ", ".join(f"{c[0]} {c[1]}" for c in ring)
            rings.append(f"({ring_str})")
        return f"MULTIPOLYGON(({', '.join(rings)}))"

    elif gtype == "MultiPolygon":
        polys = []
        for poly in coords:
            rings = []
            for ring in poly:
                ring_str = ", ".join(f"{c[0]} {c[1]}" for c in ring)
                rings.append(f"({ring_str})")
            polys.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON({', '.join(polys)})"

    return None


def _state_fips_from_geoid(geoid: str, geo_type: str) -> str:
    if geo_type in ("state",):
        return geoid[:2] if len(geoid) >= 2 else "13"
    if geo_type in ("county", "tract", "place", "zcta",
                    "school_district_unified"):
        return geoid[:2] if len(geoid) >= 2 else "13"
    return "13"  # default to Georgia for ARC data


def _county_fips_from_geoid(geoid: str, geo_type: str) -> Optional[str]:
    if geo_type in ("county",) and len(geoid) == 5:
        return geoid[2:]
    if geo_type in ("tract",) and len(geoid) == 11:
        return geoid[2:5]
    return None


def _place_fips_from_geoid(geoid: str, geo_type: str) -> Optional[str]:
    if geo_type in ("place",) and len(geoid) == 7:
        return geoid[2:]
    return None


def _api_for(geoid: str, geo_type: str) -> str:
    if geo_type == "state":
        return f"state:{geoid}"
    if geo_type == "county" and len(geoid) == 5:
        return f"county:{geoid[2:]}"
    if geo_type == "tract" and len(geoid) == 11:
        return f"tract:{geoid[5:]}"
    if geo_type == "place" and len(geoid) == 7:
        return f"place:{geoid[2:]}"
    if geo_type == "zcta":
        return f"zip+4 tabulation area:{geoid}"
    return ""


def _api_in(geoid: str, geo_type: str) -> str:
    if geo_type == "county" and len(geoid) == 5:
        return f"state:{geoid[:2]}"
    if geo_type == "tract" and len(geoid) == 11:
        return f"state:{geoid[:2]}+county:{geoid[2:5]}"
    if geo_type == "place" and len(geoid) == 7:
        return f"state:{geoid[:2]}"
    return ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

ALL_STEPS = [
    "tiger_national",
    "tiger_georgia",
    "arc_statewide",
    "atl_neighborhoods",
    "osm_roads",
    "osm_landmarks",
    "abi_beltline",
    "gnis",
    "cbsa_crosswalk",
    "cbsa_population",
    "zcta_place_rel",
    "major_cities",
    "curated_aliases",
    "aliases",
    "precompute",
]


def run_all(
    conn: sqlite3.Connection,
    raw: Path,
    steps: Optional[list[str]] = None,
    skip: Optional[list[str]] = None,
) -> dict:
    """Run all loading steps in order."""
    if steps is None:
        steps = ALL_STEPS
    if skip:
        steps = [s for s in steps if s not in skip]

    results = {}
    neighborhood_csv = _repo_root() / "data" / "geo" / "raw" / "Neighborhood.csv"

    for step in steps:
        t0 = time.time()
        logger.info("[%s]", step.upper())
        try:
            if step == "tiger_national":
                results[step] = load_tiger_national(conn, raw)
            elif step == "tiger_georgia":
                results[step] = load_tiger_georgia(conn, raw)
            elif step == "arc_statewide":
                results[step] = load_arc_statewide(conn, raw)
            elif step == "atl_neighborhoods":
                results[step] = load_atl_neighborhoods(
                    conn, raw, neighborhood_csv
                )
            elif step == "osm_roads":
                results[step] = load_osm_roads(conn, raw)
            elif step == "osm_landmarks":
                results[step] = load_osm_landmarks(conn, raw)
            elif step == "abi_beltline":
                results[step] = load_abi_beltline(conn, raw)
            elif step == "gnis":
                results[step] = load_gnis(conn, raw)
            elif step == "cbsa_crosswalk":
                results[step] = load_cbsa_crosswalk(conn, raw)
            elif step == "cbsa_population":
                from scripts.geography.populate_cbsa_population import (
                    populate_cbsa_population,
                )
                results[step] = populate_cbsa_population(conn)
            elif step == "zcta_place_rel":
                results[step] = load_zcta_place_rel(conn, raw)
            elif step == "major_cities":
                results[step] = load_major_cities(conn, raw)
            elif step == "curated_aliases":
                results[step] = load_curated_aliases(conn, raw)
            elif step == "aliases":
                results[step] = build_aliases(conn)
            elif step == "precompute":
                results[step] = run_precompute(conn)
        except Exception as e:
            logger.error("Step %s failed: %s", step, e, exc_info=True)
            results[step] = {"error": str(e)}

        elapsed = time.time() - t0
        logger.info("  → %.1fs", elapsed)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load all gazetteer sources into SpatiaLite database"
    )
    parser.add_argument(
        "--only", nargs="+", choices=ALL_STEPS,
        help="Run only these steps",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=ALL_STEPS,
        help="Skip these steps",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reinitialize the database before loading",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to database (default: data/geo/gazetteer.db)",
    )
    parser.add_argument(
        "--raw", default=None,
        help="Path to raw data directory (default: data/geo/raw)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db_path = Path(args.db) if args.db else _db_path()
    raw = Path(args.raw) if args.raw else _raw_dir()

    # Initialize or open database. init_database is idempotent — safe to
    # call against an existing DB, and ensures any newly-added tables
    # (e.g. near_tract_map, county_adjacency) get created before we
    # try to populate them.
    from scripts.geography.schema import init_database
    if args.overwrite:
        logger.info("Reinitializing database at %s", db_path)
    elif db_path.exists():
        logger.info("Opening existing database at %s", db_path)
    else:
        logger.info("Initializing new database at %s", db_path)
    conn = init_database(db_path, overwrite=args.overwrite)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("GAZETTEER LOAD")
    logger.info("  Database: %s", db_path)
    logger.info("  Raw data: %s", raw)
    logger.info("=" * 60)

    results = run_all(
        conn, raw,
        steps=args.only,
        skip=args.skip,
    )

    elapsed = time.time() - t0

    # Final counts
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPLETE (%.1fs)", elapsed)
    for table in ["admin_geographies", "named_places", "roads",
                  "geo_aliases", "place_tract_map", "road_tract_map",
                  "near_tract_map", "county_adjacency",
                  "cbsa_counties", "zcta_place_rel"]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info("  %-30s %d rows", table, n)
        except Exception:
            pass
    logger.info("=" * 60)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
