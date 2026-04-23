"""Populate admin_geographies.population for CBSAs from ACS5.

The gazetteer's admin_geographies table has a population column that is
empty for CBSAs (TIGER shapefiles don't carry it). We fetch total
population (variable B01003_001E) for all CBSAs in one API call and
update the rows.

Used by Stage 8's top-N MSA filter: with population values available,
we can restrict fetches to the top 100 CBSAs + all in-scope GA CBSAs,
cutting ~85% of the msa geo-level storage.

Usage:
    python -m scripts.geography.populate_cbsa_population
    python -m scripts.geography.populate_cbsa_population --year 2023
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import urllib.request
import urllib.parse
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Preference order — try the most recent ACS5 vintage first, fall back.
# The API 404s instead of serving partial data when a vintage doesn't
# exist yet, so the fallback loop is simple.
DEFAULT_YEAR_PREFERENCE = [2023, 2022, 2021, 2020]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fetch_cbsa_populations(year: int) -> list[tuple[str, int]]:
    """Return list of (cbsa_geoid, population) from ACS5 for one year."""
    api_key = os.environ.get("CENSUS_API_KEY", "")
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    geo = "metropolitan statistical area/micropolitan statistical area"
    params = urllib.parse.urlencode({
        "get": "B01003_001E",
        "for": f"{geo}:*",
        **({"key": api_key} if api_key else {}),
    })
    url = f"{base}?{params}"
    logger.info("Fetching CBSA population from %s", url)

    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    if not data or len(data) < 2:
        raise RuntimeError(f"Empty ACS5 response for year {year}")

    headers = data[0]
    try:
        pop_idx = headers.index("B01003_001E")
        # Census API returns the geo column with underscores instead of spaces
        geo_idx = next(
            i for i, h in enumerate(headers)
            if "metropolitan statistical area" in h.lower()
        )
    except (ValueError, StopIteration) as e:
        raise RuntimeError(
            f"Unexpected ACS5 response shape: {headers}"
        ) from e

    out: list[tuple[str, int]] = []
    for row in data[1:]:
        raw_pop = row[pop_idx]
        geoid = row[geo_idx]
        if not geoid:
            continue
        try:
            pop = int(float(raw_pop)) if raw_pop is not None else None
        except (ValueError, TypeError):
            pop = None
        if pop is None or pop <= 0:
            continue
        out.append((geoid, pop))
    logger.info("  parsed %d CBSA populations", len(out))
    return out


def populate_cbsa_population(
    conn: sqlite3.Connection,
    year_preference: Optional[list[int]] = None,
) -> dict:
    """Populate admin_geographies.population for every CBSA row.

    Tries ACS5 vintages in ``year_preference`` order until one succeeds.
    Only updates geo_type IN ('msa', 'micro') rows. Idempotent.
    """
    years = year_preference or list(DEFAULT_YEAR_PREFERENCE)
    last_err: Optional[Exception] = None
    for year in years:
        try:
            rows = _fetch_cbsa_populations(year)
            if rows:
                break
        except Exception as e:
            logger.warning("ACS5 %d unavailable: %s", year, e)
            last_err = e
    else:
        raise RuntimeError(
            f"No ACS5 vintage returned CBSA populations: {last_err}"
        )

    logger.info("Using ACS5 %d population data", year)
    updated = 0
    with conn:
        for geoid, pop in rows:
            cursor = conn.execute(
                "UPDATE admin_geographies SET population = ? "
                "WHERE geoid = ? AND geo_type IN ('msa', 'micro')",
                (pop, geoid),
            )
            if cursor.rowcount > 0:
                updated += 1

    # Sanity check: how many CBSAs now have population?
    total = conn.execute(
        "SELECT COUNT(*) FROM admin_geographies "
        "WHERE geo_type IN ('msa', 'micro')"
    ).fetchone()[0]
    with_pop = conn.execute(
        "SELECT COUNT(*) FROM admin_geographies "
        "WHERE geo_type IN ('msa', 'micro') "
        "  AND population IS NOT NULL AND population > 0"
    ).fetchone()[0]

    logger.info(
        "CBSA population populate: %d updated; %d/%d CBSAs now have pop",
        updated, with_pop, total,
    )
    return {
        "year_used": year,
        "updated": updated,
        "with_population": with_pop,
        "total_cbsas": total,
    }


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--year", type=int, default=None,
        help=f"Specific ACS5 vintage to use. "
             f"Default tries {DEFAULT_YEAR_PREFERENCE}.",
    )
    p.add_argument(
        "--db", default=None,
        help="SpatiaLite DB path (default: data/geo/gazetteer.db)",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    from scripts.geography.schema import open_spatialite
    db_path = Path(args.db) if args.db else (
        _repo_root() / "data" / "geo" / "gazetteer.db"
    )
    conn = open_spatialite(db_path)
    try:
        year_pref = [args.year] if args.year else None
        summary = populate_cbsa_population(conn, year_preference=year_pref)
        print(f"\nSUMMARY: {summary}")
        return 0 if summary["updated"] > 0 else 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
