"""Load Georgia road network from OpenStreetMap via Overpass API.

Fetches major road geometries (interstates, US highways, state routes)
for Georgia. These are used for corridor queries like "along I-85"
and "near I-20".

Uses the Overpass API directly — no large PBF download needed for
just the major road network.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass queries for Georgia roads
# Each returns a relation or way with its geometry
ROAD_QUERIES = {
    "I-20": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="20"]["state"="GA"];
        out geom;
    """,
    "I-75": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="75"]["state"="GA"];
        out geom;
    """,
    "I-85": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="85"]["state"="GA"];
        out geom;
    """,
    "I-285": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="285"];
        out geom;
    """,
    "I-575": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="575"];
        out geom;
    """,
    "I-675": """
        [out:json][timeout:60];
        relation["network"="US:I"]["ref"="675"];
        out geom;
    """,
    "GA-400": """
        [out:json][timeout:60];
        relation["network"="US:GA"]["ref"="400"];
        out geom;
    """,
    "US-78": """
        [out:json][timeout:60];
        relation["network"="US:US"]["ref"="78"]["state"="GA"];
        out geom;
    """,
    "US-23": """
        [out:json][timeout:60];
        (
          way["network"="US:US"]["ref"="23"]
             (33.5,-84.6,34.0,-84.2);
        );
        out geom;
    """,
    "US-41": """
        [out:json][timeout:60];
        (
          way["network"="US:US"]["ref"="41"]
             (33.5,-84.8,34.1,-84.3);
        );
        out geom;
    """,
    "SR-316": """
        [out:json][timeout:60];
        relation["network"="US:GA"]["ref"="316"];
        out geom;
    """,
    "US-29": """
        [out:json][timeout:60];
        (
          way["network"="US:US"]["ref"="29"]
             (33.0,-85.0,34.0,-83.5);
        );
        out geom;
    """,
    "US-19": """
        [out:json][timeout:60];
        (
          way["network"="US:US"]["ref"="19"]
             (31.0,-84.7,34.5,-84.0);
        );
        out geom;
    """,
}

ROAD_METADATA = {
    "I-20":   {"full_name": "Interstate 20",        "road_type": "interstate"},
    "I-75":   {"full_name": "Interstate 75",        "road_type": "interstate"},
    "I-85":   {"full_name": "Interstate 85",        "road_type": "interstate"},
    "I-285":  {"full_name": "Interstate 285",       "road_type": "interstate"},
    "I-575":  {"full_name": "Interstate 575",       "road_type": "interstate"},
    "I-675":  {"full_name": "Interstate 675",       "road_type": "interstate"},
    "GA-400": {"full_name": "Georgia State Route 400", "road_type": "state_route"},
    "US-78":  {"full_name": "US Highway 78",        "road_type": "us_highway"},
    "US-23":  {"full_name": "US Highway 23 (Buford Highway)", "road_type": "us_highway"},
    "US-41":  {"full_name": "US Highway 41 (Cobb Parkway)",   "road_type": "us_highway"},
    "SR-316": {"full_name": "Georgia State Route 316",         "road_type": "state_route"},
    "US-29":  {"full_name": "US Highway 29",        "road_type": "us_highway"},
    "US-19":  {"full_name": "US Highway 19",        "road_type": "us_highway"},
}


def fetch_road_from_overpass(
    road_id: str,
    query: str,
    retries: int = 3,
) -> Optional[str]:
    """Fetch a road's geometry from Overpass API and return as WKT MULTILINESTRING.

    Returns None if the fetch fails or returns no geometry.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()

            wkt = _overpass_to_wkt(data, road_id)
            if wkt:
                return wkt

            logger.warning(
                "No geometry returned for %s (attempt %d/%d)",
                road_id, attempt, retries,
            )

        except requests.RequestException as e:
            logger.warning(
                "Overpass request failed for %s (attempt %d/%d): %s",
                road_id, attempt, retries, e,
            )

        if attempt < retries:
            time.sleep(5 * attempt)  # Back off between retries

    return None


def _overpass_to_wkt(data: dict, road_id: str) -> Optional[str]:
    """Convert Overpass JSON response to WKT MULTILINESTRING."""
    linestrings = []

    elements = data.get("elements", [])
    for el in elements:
        el_type = el.get("type")

        if el_type == "relation":
            # Collect geometry from member ways
            for member in el.get("members", []):
                if member.get("type") == "way" and "geometry" in member:
                    coords = member["geometry"]
                    if len(coords) >= 2:
                        coord_str = ", ".join(
                            f"{c['lon']} {c['lat']}" for c in coords
                        )
                        linestrings.append(f"({coord_str})")

        elif el_type == "way" and "geometry" in el:
            coords = el["geometry"]
            if len(coords) >= 2:
                coord_str = ", ".join(
                    f"{c['lon']} {c['lat']}" for c in coords
                )
                linestrings.append(f"({coord_str})")

    if not linestrings:
        return None

    return f"MULTILINESTRING({', '.join(linestrings)})"


def load_roads_from_overpass(
    conn: sqlite3.Connection,
    road_ids: Optional[list[str]] = None,
    delay_seconds: float = 2.0,
) -> dict[str, bool]:
    """Fetch and load all roads into the roads table.

    Args:
        conn: SpatiaLite connection
        road_ids: List of road IDs to fetch. If None, fetches all.
        delay_seconds: Pause between Overpass requests (be polite)

    Returns:
        Dict of {road_id: success}
    """
    if road_ids is None:
        road_ids = list(ROAD_QUERIES.keys())

    results = {}
    for road_id in road_ids:
        query = ROAD_QUERIES.get(road_id)
        if not query:
            logger.warning("No query defined for road %s", road_id)
            results[road_id] = False
            continue

        meta = ROAD_METADATA.get(road_id, {})
        logger.info("Fetching geometry for %s (%s)", road_id, meta.get("full_name", ""))

        wkt = fetch_road_from_overpass(road_id, query)

        if wkt:
            try:
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO roads "
                        "(road_id, name, full_name, road_type, state_fips, "
                        " default_buffer_miles, geom) "
                        "VALUES (?, ?, ?, ?, '13', 2.0, "
                        "  CastToMultiLinestring(GeomFromText(?, 4326)))",
                        (
                            road_id, road_id,
                            meta.get("full_name", road_id),
                            meta.get("road_type", "unknown"),
                            wkt,
                        ),
                    )
                logger.info("Loaded %s", road_id)
                results[road_id] = True
            except Exception as e:
                logger.error("Failed to insert %s: %s", road_id, e)
                results[road_id] = False
        else:
            logger.warning("No geometry obtained for %s — skipping", road_id)
            results[road_id] = False

        # Be polite to the Overpass API
        time.sleep(delay_seconds)

    return results


def load_roads_from_geojson(
    conn: sqlite3.Connection,
    geojson_path: str,
) -> int:
    """Load roads from a local GeoJSON file as an alternative to Overpass.

    The GeoJSON should have features with properties:
      road_id, name, full_name, road_type

    Use this if Overpass is unavailable or you want reproducible builds.
    """
    import json
    from pathlib import Path

    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")

    with path.open() as f:
        data = json.load(f)

    loaded = 0
    with conn:
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})

            if not props.get("road_id"):
                continue

            # Convert GeoJSON geometry to WKT
            wkt = _geojson_geom_to_wkt(geom)
            if not wkt:
                continue

            conn.execute(
                "INSERT OR REPLACE INTO roads "
                "(road_id, name, full_name, road_type, state_fips, "
                " default_buffer_miles, geom) "
                "VALUES (?, ?, ?, ?, ?, ?, "
                "  CastToMultiLinestring(GeomFromText(?, 4326)))",
                (
                    props["road_id"],
                    props.get("name", props["road_id"]),
                    props.get("full_name", ""),
                    props.get("road_type", "unknown"),
                    props.get("state_fips", "13"),
                    props.get("buffer_miles", 2.0),
                    wkt,
                ),
            )
            loaded += 1

    logger.info("Loaded %d roads from GeoJSON", loaded)
    return loaded


def _geojson_geom_to_wkt(geom: dict) -> Optional[str]:
    """Convert a GeoJSON geometry object to WKT."""
    geom_type = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if geom_type == "LineString":
        coord_str = ", ".join(f"{c[0]} {c[1]}" for c in coords)
        return f"MULTILINESTRING(({coord_str}))"

    elif geom_type == "MultiLineString":
        parts = []
        for line in coords:
            coord_str = ", ".join(f"{c[0]} {c[1]}" for c in line)
            parts.append(f"({coord_str})")
        return f"MULTILINESTRING({', '.join(parts)})"

    return None
