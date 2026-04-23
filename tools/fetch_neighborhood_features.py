"""Pre-compute feature vectors for Atlanta-metro neighborhoods.

Turns every ATL neighborhood in ``gazetteer.db:named_places`` (with
tract coverage in ``place_tract_map``) into a peer-ready feature
vector and appends it to ``data/metadata/peer_features.sqlite`` with
``geo_level='neighborhood'``.

Flow:
  1. For each unique table in the peer catalog, fetch tract-level
     ACS data for the anchor state (default: Georgia, state_fips=13)
     in one Census API call per table.
  2. Load the gazetteer's ATL neighborhood list + tract_geoids.
  3. For each neighborhood:
     - Restrict the state-wide tract rows to the neighborhood's tracts
     - Apply each catalog feature's derivation, using
       population-weighted means for medians and sums for counts
     - Serialize as a JSON feature vector, persist to peer_features
  4. Optionally: for neighborhoods without enough tract coverage
     (<2 tracts) skip — peer comparisons on 1-tract neighborhoods
     are too noisy.

Usage:
    export CENSUS_API_KEY=...
    python -m tools.fetch_neighborhood_features                # full
    python -m tools.fetch_neighborhood_features --dry-run      # 3 nbhs
    python -m tools.fetch_neighborhood_features --metro atlanta
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import httpx

from scripts.chatbot.peer_features_catalog import (
    AXES, FEATURES, Feature, tables_used, validate, variables_for_table,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CENSUS_BASE = "https://api.census.gov/data"
DEFAULT_STATE_FIPS = "13"     # Georgia
DEFAULT_METRO = "atlanta"
DEFAULT_VINTAGE = 2023
DEFAULT_MIN_TRACTS = 1        # include single-tract neighborhoods;
                              # for peer ranking the noise smooths
                              # out and excluding them would drop
                              # ~60% of ATL's named neighborhoods.
OUT_DB_PATH = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
MAX_VARS_PER_CALL = 48


# ---------------------------------------------------------------------------
# Census fetch — one call per table, state-wide tract rows
# ---------------------------------------------------------------------------

def _build_url(
    *, vintage: int, variables: list[str], state_fips: str,
    api_key: Optional[str],
) -> str:
    """State-wide tract-level pull: &for=tract:*&in=state:XX."""
    get = ",".join(["NAME"] + variables)
    parts = [
        f"{CENSUS_BASE}/{vintage}/acs/acs5?get={get}",
        f"&for=tract:*&in=state:{state_fips}",
    ]
    if api_key:
        parts.append(f"&key={api_key}")
    return "".join(parts)


def _fetch_table(
    client: httpx.Client, *,
    table: str, vintage: int, state_fips: str,
    api_key: Optional[str],
) -> dict[str, dict]:
    """Fetch a single table's needed variables for all tracts in the
    state. Returns ``{tract_geoid: {var: value, …}}``.

    Chunks variables when more than ``MAX_VARS_PER_CALL`` are needed.
    """
    vars_needed = sorted(variables_for_table(table))
    chunks = [
        vars_needed[i:i + MAX_VARS_PER_CALL]
        for i in range(0, len(vars_needed), MAX_VARS_PER_CALL)
    ]
    merged: dict[str, dict] = {}
    for chunk in chunks:
        url = _build_url(
            vintage=vintage, variables=chunk,
            state_fips=state_fips, api_key=api_key,
        )
        resp = client.get(url, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"{table}: HTTP {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        if not data or len(data) < 2:
            logger.warning("empty response for %s tract-state", table)
            continue
        header = data[0]
        for row in data[1:]:
            rec = dict(zip(header, row))
            tract_geoid = (
                rec.get("state", "") + rec.get("county", "")
                + rec.get("tract", "")
            )
            if tract_geoid not in merged:
                merged[tract_geoid] = {}
            merged[tract_geoid].update({v: rec.get(v) for v in chunk})
    return merged


# ---------------------------------------------------------------------------
# Derivation — aggregate tract rows to a single neighborhood value
# ---------------------------------------------------------------------------

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    # ACS annotation sentinels
    if v <= -1_000_000:
        return None
    return v


def _tract_pop(
    tract_data: dict[str, dict], tract_geoid: str,
) -> Optional[float]:
    """Return the tract's total population from B01003_001E, or None
    if missing. Used as weight for pop-weighted means on medians."""
    tract_row = tract_data.get("B01003", {}).get(tract_geoid)
    if not tract_row:
        return None
    return _to_float(tract_row.get("B01003_001E"))


def _derive_neighborhood_feature(
    feature: Feature,
    tract_geoids: list[str],
    tract_data: dict[str, dict[str, dict]],
) -> Optional[float]:
    """Aggregate one catalog feature across the neighborhood's tracts.

    Rules:
      - ``raw`` features (typically medians or totals):
          * names containing 'median'  → pop-weighted mean
          * everything else            → sum across tracts
      - ``ratio`` (num/denom)            → sum num, sum denom, divide
      - ``sum_ratio`` (Σnum / denom)     → sum every numerator, sum
                                           the denom, divide
    """
    per_table = tract_data.get(feature.table, {})
    if feature.derivation == "raw":
        vs_var = feature.variables[0]
        is_median = "median" in feature.name
        weighted_sum = 0.0
        weight_sum = 0.0
        plain_sum = 0.0
        plain_n = 0
        for tg in tract_geoids:
            row = per_table.get(tg)
            if not row:
                continue
            v = _to_float(row.get(vs_var))
            if v is None:
                continue
            if is_median:
                pop = _tract_pop(tract_data, tg) or 0.0
                if pop > 0:
                    weighted_sum += v * pop
                    weight_sum += pop
            else:
                plain_sum += v
                plain_n += 1
        if is_median:
            return weighted_sum / weight_sum if weight_sum > 0 else None
        return plain_sum if plain_n > 0 else None

    if feature.derivation == "ratio":
        num_var, den_var = feature.variables[0], feature.variables[1]
        num_sum, den_sum = 0.0, 0.0
        for tg in tract_geoids:
            row = per_table.get(tg)
            if not row:
                continue
            n = _to_float(row.get(num_var))
            d = _to_float(row.get(den_var))
            if n is None or d is None:
                continue
            num_sum += n
            den_sum += d
        return num_sum / den_sum if den_sum > 0 else None

    if feature.derivation == "sum_ratio":
        num_vars = feature.variables[:-1]
        den_var = feature.variables[-1]
        num_sum, den_sum = 0.0, 0.0
        for tg in tract_geoids:
            row = per_table.get(tg)
            if not row:
                continue
            d = _to_float(row.get(den_var))
            if d is None:
                continue
            # sum up numerator components (treat missing parts as 0)
            nk = sum(
                (_to_float(row.get(v)) or 0.0)
                for v in num_vars
            )
            num_sum += nk
            den_sum += d
        return num_sum / den_sum if den_sum > 0 else None

    raise ValueError(f"unknown derivation: {feature.derivation}")


# ---------------------------------------------------------------------------
# Gazetteer lookups — neighborhoods + their tract lists
# ---------------------------------------------------------------------------

def _load_neighborhoods(
    gaz: sqlite3.Connection, *, metro: Optional[str] = None,
    source: Optional[str] = "atl_opendata",
) -> list[dict]:
    """Return neighborhoods with non-empty tract lists.

    Each entry: {place_id, name, metro, state_fips, tract_geoids:[...]}.

    ``source`` defaults to ``'atl_opendata'``: the curated City of
    Atlanta neighborhood list (~248 entries like Buckhead, Midtown,
    Virginia-Highland). Pass ``source=None`` to include OSM-sourced
    subdivisions too (+3k entries, mostly tiny named subdivisions
    that are not useful peer candidates for grant queries).
    """
    where = ["np.place_type = 'neighborhood'"]
    params: list = []
    if metro:
        where.append("np.metro = ?")
        params.append(metro)
    if source:
        where.append("np.source = ?")
        params.append(source)
    sql = f"""
        SELECT np.place_id, np.name, np.metro,
               GROUP_CONCAT(DISTINCT ptm.tract_geoid) AS tract_csv
          FROM named_places np
          JOIN place_tract_map ptm ON ptm.place_id = np.place_id
         WHERE {' AND '.join(where)}
         GROUP BY np.place_id
    """
    out: list[dict] = []
    for r in gaz.execute(sql, params):
        tract_geoids = (r["tract_csv"] or "").split(",")
        tract_geoids = [t for t in tract_geoids if t]
        if not tract_geoids:
            continue
        # state fips = first 2 chars of any tract geoid
        state_fips = tract_geoids[0][:2] if tract_geoids else None
        out.append({
            "place_id": r["place_id"],
            "name": r["name"],
            "metro": r["metro"],
            "state_fips": state_fips,
            "tract_geoids": tract_geoids,
        })
    return out


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS peer_features (
    geo_level   TEXT NOT NULL,
    geo_id      TEXT NOT NULL,
    state_fips  TEXT,
    geo_name    TEXT NOT NULL,
    vintage     INTEGER NOT NULL,
    population  INTEGER,
    features_json TEXT NOT NULL,
    fetched_at  REAL NOT NULL,
    PRIMARY KEY (geo_level, geo_id, vintage)
);
CREATE INDEX IF NOT EXISTS idx_peer_pop
    ON peer_features(geo_level, population);
CREATE INDEX IF NOT EXISTS idx_peer_state
    ON peer_features(state_fips);
"""


def _open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA_SQL)
    con.commit()
    return con


def _insert(
    con: sqlite3.Connection, rows: list[dict], vintage: int,
) -> None:
    now = time.time()
    con.executemany(
        """
        INSERT OR REPLACE INTO peer_features
          (geo_level, geo_id, state_fips, geo_name, vintage,
           population, features_json, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (r["geo_level"], r["geo_id"], r["state_fips"],
             r["geo_name"], vintage, r["population"],
             r["features_json"], now)
            for r in rows
        ],
    )
    con.commit()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    *,
    vintage: int,
    state_fips: str,
    metro: Optional[str],
    out_db: Path,
    api_key: Optional[str],
    min_tracts: int,
    source: Optional[str] = "atl_opendata",
    dry_run: bool = False,
) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    issues = validate()
    if issues:
        for i in issues:
            logger.error("catalog issue: %s", i)
        raise SystemExit(1)

    # Gazetteer — SpatiaLite extension is not needed for this tool
    from scripts.chatbot.gazetteer_db import open_spatialite
    gaz_path = REPO_ROOT / "data" / "geo" / "gazetteer.db"
    gaz = open_spatialite(gaz_path)
    gaz.row_factory = sqlite3.Row

    neighborhoods = _load_neighborhoods(gaz, metro=metro, source=source)
    logger.info(
        "neighborhoods in scope: %d  "
        "(metro=%s, source=%s, min_tracts=%d)",
        len(neighborhoods), metro, source, min_tracts,
    )
    neighborhoods = [
        n for n in neighborhoods if len(n["tract_geoids"]) >= min_tracts
    ]
    logger.info(
        "after min_tracts filter: %d eligible neighborhoods",
        len(neighborhoods),
    )
    if dry_run:
        neighborhoods = neighborhoods[:3]
        logger.info("dry-run: capping at %d", len(neighborhoods))

    # Phase 1 — fetch tract-level ACS for every table we need.
    # Result: {table_id: {tract_geoid: {var: value, …}}}
    tables = sorted(tables_used())
    tract_data: dict[str, dict[str, dict]] = {}
    stats = {"calls": 0, "errors": 0, "rows": 0}
    logger.info(
        "fetching tract-level ACS5 for state=%s, %d tables...",
        state_fips, len(tables),
    )
    with httpx.Client(timeout=120.0) as client:
        for i, table in enumerate(tables, 1):
            try:
                data = _fetch_table(
                    client, table=table, vintage=vintage,
                    state_fips=state_fips, api_key=api_key,
                )
                tract_data[table] = data
                stats["calls"] += 1
                stats["rows"] += len(data)
                logger.info(
                    "  [%d/%d] %s → %d tracts",
                    i, len(tables), table, len(data),
                )
            except Exception as e:
                stats["errors"] += 1
                logger.warning(
                    "  [%d/%d] %s FAILED: %s",
                    i, len(tables), table, e,
                )

    # Phase 2 — aggregate per neighborhood + persist.
    con = _open_db(out_db) if not dry_run else None
    rows_out: list[dict] = []
    for nb in neighborhoods:
        features: dict[str, float] = {}
        tg = nb["tract_geoids"]
        for fname, feat in FEATURES.items():
            v = _derive_neighborhood_feature(feat, tg, tract_data)
            if v is not None:
                features[fname] = v
        pop = features.get("total_population")
        pop = int(pop) if pop is not None else None
        rows_out.append({
            "geo_level": "neighborhood",
            "geo_id": nb["place_id"],
            "state_fips": nb["state_fips"],
            "geo_name": nb["name"],
            "population": pop,
            "features_json": json.dumps(
                features, separators=(",", ":"),
            ),
        })

    logger.info(
        "derived features for %d neighborhoods, mean "
        "feature count = %.1f",
        len(rows_out),
        (sum(len(json.loads(r["features_json"])) for r in rows_out)
         / max(1, len(rows_out))),
    )
    if dry_run:
        logger.info("dry-run: skipping DB write")
        logger.info("sample: %s", rows_out[:2])
    else:
        _insert(con, rows_out, vintage)
        logger.info(
            "wrote %d neighborhood rows to %s",
            len(rows_out), out_db,
        )
        con.close()

    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vintage", type=int, default=DEFAULT_VINTAGE)
    p.add_argument("--state-fips", default=DEFAULT_STATE_FIPS)
    p.add_argument("--metro", default=DEFAULT_METRO,
                   help="Gazetteer metro filter (default 'atlanta'); "
                        "pass empty string for all metros.")
    p.add_argument("--out-db", type=Path, default=OUT_DB_PATH)
    p.add_argument("--min-tracts", type=int, default=DEFAULT_MIN_TRACTS)
    p.add_argument("--source", default="atl_opendata",
                   help="Gazetteer named_places.source filter. "
                        "Default 'atl_opendata' = curated ATL "
                        "neighborhood list (~248). Pass empty string "
                        "to include OSM subdivisions too.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        logger.warning("CENSUS_API_KEY not set — will hit unauth rate cap")
    run(
        vintage=args.vintage,
        state_fips=args.state_fips,
        metro=args.metro or None,
        out_db=args.out_db,
        api_key=api_key,
        min_tracts=args.min_tracts,
        source=args.source or None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
