"""Download and load Census TIGER/Line shapefiles into the gazetteer.

Downloads:
  - National: states, counties
  - Georgia (FIPS 13): tracts, places, county subdivisions
  - Optionally: all US tracts (large — ~1GB download)

Data year: 2020 (matches the Census data vintage in Stage 8)
"""
from __future__ import annotations

import io
import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
from tqdm import tqdm

from scripts.geography.schema import open_spatialite, set_build_metadata

logger = logging.getLogger(__name__)

TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2020"
YEAR = "2020"

# Files to download — (url_template, local_name, description)
TIGER_FILES = [
    # National layers
    (f"{TIGER_BASE}/STATE/tl_2020_us_state.zip",
     "tl_2020_us_state.zip", "US States"),
    (f"{TIGER_BASE}/COUNTY/tl_2020_us_county.zip",
     "tl_2020_us_county.zip", "US Counties"),
    # Georgia-specific layers (FIPS 13)
    (f"{TIGER_BASE}/TRACT/tl_2020_13_tract.zip",
     "tl_2020_13_tract.zip", "Georgia Tracts"),
    (f"{TIGER_BASE}/PLACE/tl_2020_13_place.zip",
     "tl_2020_13_place.zip", "Georgia Places"),
    (f"{TIGER_BASE}/COUSUB/tl_2020_13_cousub.zip",
     "tl_2020_13_cousub.zip", "Georgia County Subdivisions"),
]


def download_tiger_files(
    data_dir: Path,
    force: bool = False,
) -> dict[str, Path]:
    """Download TIGER/Line shapefiles, skipping already-downloaded files.

    Returns dict of {local_name: local_path}
    """
    tiger_dir = data_dir / "tiger"
    tiger_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for url, local_name, description in TIGER_FILES:
        local_path = tiger_dir / local_name
        if local_path.exists() and not force:
            logger.info("Already downloaded: %s", local_name)
            downloaded[local_name] = local_path
            continue

        logger.info("Downloading %s from %s", description, url)
        _download_file(url, local_path)
        downloaded[local_name] = local_path

    return downloaded


def _download_file(url: str, dest: Path) -> None:
    """Download a file with a progress bar."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=dest.name, leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def load_states(
    conn: sqlite3.Connection,
    zip_path: Path,
) -> int:
    """Load US states into admin_geographies."""
    gdf = _read_shapefile(zip_path)
    gdf = gdf.to_crs("EPSG:4326")

    rows_inserted = 0
    with conn:
        for _, row in gdf.iterrows():
            state_fips = str(row["STATEFP"]).zfill(2)
            geoid = state_fips
            name = row["NAME"]

            wkt = row.geometry.wkt if row.geometry else None
            conn.execute(
                "INSERT OR REPLACE INTO admin_geographies "
                "(geoid, name, geo_type, state_fips, aland, "
                " api_for_clause, api_in_clause, geom) "
                "VALUES (?, ?, 'state', ?, ?, ?, ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (
                    geoid, name, state_fips,
                    float(row.get("ALAND", 0) or 0),
                    f"state:{state_fips}",
                    "",  # No 'in' clause for state-level
                    wkt,
                ),
            )
            rows_inserted += 1

    logger.info("Loaded %d states", rows_inserted)
    return rows_inserted


def load_counties(
    conn: sqlite3.Connection,
    zip_path: Path,
) -> int:
    """Load US counties into admin_geographies."""
    gdf = _read_shapefile(zip_path)
    gdf = gdf.to_crs("EPSG:4326")

    rows_inserted = 0
    with conn:
        for _, row in gdf.iterrows():
            state_fips = str(row["STATEFP"]).zfill(2)
            county_fips = str(row["COUNTYFP"]).zfill(3)
            geoid = f"{state_fips}{county_fips}"
            name = row["NAMELSAD"]  # "Fulton County"

            wkt = row.geometry.wkt if row.geometry else None
            conn.execute(
                "INSERT OR REPLACE INTO admin_geographies "
                "(geoid, name, geo_type, state_fips, county_fips, lsad, "
                " aland, api_for_clause, api_in_clause, geom) "
                "VALUES (?, ?, 'county', ?, ?, ?, ?, ?, ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (
                    geoid, name, state_fips, county_fips,
                    str(row.get("LSAD", "")),
                    float(row.get("ALAND", 0) or 0),
                    f"county:{county_fips}",
                    f"state:{state_fips}",
                    wkt,
                ),
            )
            rows_inserted += 1

    logger.info("Loaded %d counties", rows_inserted)
    return rows_inserted


def load_tracts(
    conn: sqlite3.Connection,
    zip_path: Path,
) -> int:
    """Load census tracts into admin_geographies."""
    gdf = _read_shapefile(zip_path)
    gdf = gdf.to_crs("EPSG:4326")

    rows_inserted = 0
    with conn:
        for _, row in gdf.iterrows():
            state_fips = str(row["STATEFP"]).zfill(2)
            county_fips = str(row["COUNTYFP"]).zfill(3)
            tract_ce = str(row["TRACTCE"]).zfill(6)
            geoid = f"{state_fips}{county_fips}{tract_ce}"
            name = f"Census Tract {row['NAME']}"

            wkt = row.geometry.wkt if row.geometry else None
            conn.execute(
                "INSERT OR REPLACE INTO admin_geographies "
                "(geoid, name, geo_type, state_fips, county_fips, "
                " aland, api_for_clause, api_in_clause, geom) "
                "VALUES (?, ?, 'tract', ?, ?, ?, ?, ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (
                    geoid, name, state_fips, county_fips,
                    float(row.get("ALAND", 0) or 0),
                    f"tract:{tract_ce}",
                    f"state:{state_fips}+county:{county_fips}",
                    wkt,
                ),
            )
            rows_inserted += 1

    logger.info("Loaded %d tracts", rows_inserted)
    return rows_inserted


def load_places(
    conn: sqlite3.Connection,
    zip_path: Path,
) -> int:
    """Load incorporated places (cities, towns, CDPs) into admin_geographies."""
    gdf = _read_shapefile(zip_path)
    gdf = gdf.to_crs("EPSG:4326")

    rows_inserted = 0
    with conn:
        for _, row in gdf.iterrows():
            state_fips = str(row["STATEFP"]).zfill(2)
            place_fips = str(row["PLACEFP"]).zfill(5)
            geoid = f"{state_fips}{place_fips}"
            name = row["NAME"]

            wkt = row.geometry.wkt if row.geometry else None
            conn.execute(
                "INSERT OR REPLACE INTO admin_geographies "
                "(geoid, name, geo_type, state_fips, place_fips, lsad, "
                " aland, api_for_clause, api_in_clause, geom) "
                "VALUES (?, ?, 'place', ?, ?, ?, ?, ?, ?, "
                "  CastToMultiPolygon(GeomFromText(?, 4326)))",
                (
                    geoid, name, state_fips, place_fips,
                    str(row.get("LSAD", "")),
                    float(row.get("ALAND", 0) or 0),
                    f"place:{place_fips}",
                    f"state:{state_fips}",
                    wkt,
                ),
            )
            rows_inserted += 1

    logger.info("Loaded %d places", rows_inserted)
    return rows_inserted


def _read_shapefile(zip_path: Path) -> gpd.GeoDataFrame:
    """Read a shapefile from a zip archive."""
    return gpd.read_file(f"zip://{zip_path}")
