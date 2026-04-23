"""Fetch peer-feature vectors from Census ACS5 for place / county / MSA.

Reads `scripts/chatbot/peer_features_catalog.py`, issues one Census API
call per (table, geo_level) pair, applies each feature's derivation
formula, and writes a wide feature-vector table to
`data/metadata/peer_features.sqlite`.

Totals for the current catalog: 107 features × 56 tables × 3 geo_levels
= ~168 API calls. Wall-clock ~5–10 min with a Census API key (the
default unauthenticated cap is 500 calls/day).

Output schema:

    CREATE TABLE peer_features (
        geo_level   TEXT NOT NULL,   -- 'place' | 'county' | 'msa'
        geo_id      TEXT NOT NULL,
        state_fips  TEXT,
        geo_name    TEXT NOT NULL,
        vintage     INTEGER NOT NULL,
        population  INTEGER,          -- hot column for size bucketing
        features_json TEXT NOT NULL,
        PRIMARY KEY (geo_level, geo_id, vintage)
    )

Usage:

    export CENSUS_API_KEY=...
    python -m tools.fetch_peer_features                      # all levels, 2023
    python -m tools.fetch_peer_features --vintage 2022
    python -m tools.fetch_peer_features --geo-levels place   # just places
    python -m tools.fetch_peer_features --dry-run            # fetch 1 table, no write
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
OUT_DB_PATH = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
CENSUS_BASE = "https://api.census.gov/data"

# Geo-level request shapes. Each entry gives the `&for=...` clause and
# the identifier columns the Census API returns alongside the data
# variables.
GEO_LEVELS = {
    "place": {
        "for_clause": "place:*",
        "in_clause": "state:*",
        "id_cols": ("state", "place"),
        "id_join": "{state}{place}",           # full geoid
    },
    "county": {
        "for_clause": "county:*",
        "in_clause": "state:*",
        "id_cols": ("state", "county"),
        "id_join": "{state}{county}",
    },
    "msa": {
        # Census API uses the long geography name with URL escaping.
        "for_clause": (
            "metropolitan%20statistical%20area/"
            "micropolitan%20statistical%20area:*"
        ),
        "in_clause": None,
        "id_cols": (
            "metropolitan statistical area/micropolitan statistical area",
        ),
        "id_join": "{metropolitan statistical area/micropolitan statistical area}",
    },
}

MAX_VARS_PER_CALL = 48      # Census API limit is 50; reserve 2 for NAME + geo id


# ---------------------------------------------------------------------------
# HTTP — one call per (table, geo_level). Chunks variables if needed.
# ---------------------------------------------------------------------------

def _build_url(
    *, vintage: int, dataset: str, variables: list[str],
    for_clause: str, in_clause: Optional[str],
    api_key: Optional[str],
) -> str:
    get = ",".join(["NAME"] + variables)
    parts = [f"{CENSUS_BASE}/{vintage}/{dataset}?get={get}",
             f"&for={for_clause}"]
    if in_clause:
        parts.append(f"&in={in_clause}")
    if api_key:
        parts.append(f"&key={api_key}")
    return "".join(parts)


def _fetch_one(
    client: httpx.Client, *,
    table: str, geo_level: str, vintage: int,
    api_key: Optional[str],
) -> list[dict]:
    """Return one list of {var: value, ...} dicts — one per geo entity.

    Chunks variables if the table needs more than MAX_VARS_PER_CALL.
    Results are joined on the id columns so the caller always gets one
    merged row per entity regardless of chunk count.
    """
    vars_needed = sorted(variables_for_table(table))
    geo = GEO_LEVELS[geo_level]
    chunks = [
        vars_needed[i:i + MAX_VARS_PER_CALL]
        for i in range(0, len(vars_needed), MAX_VARS_PER_CALL)
    ]

    merged: dict[str, dict] = {}
    for chunk in chunks:
        url = _build_url(
            vintage=vintage, dataset="acs/acs5",
            variables=chunk, for_clause=geo["for_clause"],
            in_clause=geo["in_clause"], api_key=api_key,
        )
        logger.debug("GET %s", url)
        resp = client.get(url)
        if resp.status_code != 200:
            raise RuntimeError(
                f"{table} / {geo_level} / {vintage}: "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{table} / {geo_level}: bad JSON — {e}"
            ) from e
        if not data or len(data) < 2:
            logger.warning(
                "empty response for %s / %s", table, geo_level,
            )
            continue
        header = data[0]
        for row in data[1:]:
            rec = dict(zip(header, row))
            key = geo["id_join"].format(**rec)
            if key not in merged:
                merged[key] = {
                    "geo_id": key,
                    "geo_name": rec.get("NAME", ""),
                    "state_fips": rec.get("state"),
                }
            merged[key].update({v: rec.get(v) for v in chunk})
    return list(merged.values())


# ---------------------------------------------------------------------------
# Derivation — apply each feature's formula to raw rows
# ---------------------------------------------------------------------------

def _to_float(x) -> Optional[float]:
    """Census returns strings; convert safely, treating -666666666 etc.
    (annotation sentinels) as missing."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    # ACS annotation sentinels: all very negative integers meaning
    # "suppressed" / "too few sample" / etc.
    if v <= -1_000_000:
        return None
    return v


def _derive(feature: Feature, row: dict) -> Optional[float]:
    vs = [_to_float(row.get(v)) for v in feature.variables]
    if feature.derivation == "raw":
        return vs[0]
    if feature.derivation == "ratio":
        num, den = vs[0], vs[1]
        if num is None or den is None or den == 0:
            return None
        return num / den
    if feature.derivation == "sum_ratio":
        nums = vs[:-1]
        den = vs[-1]
        if den is None or den == 0:
            return None
        clean_nums = [n for n in nums if n is not None]
        if not clean_nums:
            return None
        return sum(clean_nums) / den
    raise ValueError(f"unknown derivation: {feature.derivation}")


# ---------------------------------------------------------------------------
# Aggregation — one Feature catalog + many table fetches → one row per geo
# ---------------------------------------------------------------------------

def _aggregate_features(
    *, fetched_by_table: dict[str, list[dict]],
    geo_level: str,
) -> list[dict]:
    """Combine per-table fetch results into one record per geo entity,
    then compute each catalog feature.

    `fetched_by_table` is a dict keyed by table_id whose values are
    lists of {geo_id, geo_name, state_fips, <vars...>} rows (as
    emitted by _fetch_one).
    """
    # Build geo_id → merged-rows dict (union across all tables).
    by_geo: dict[str, dict] = {}
    for table, rows in fetched_by_table.items():
        for row in rows:
            gid = row["geo_id"]
            if gid not in by_geo:
                by_geo[gid] = {
                    "geo_id": gid,
                    "geo_name": row.get("geo_name", ""),
                    "state_fips": row.get("state_fips"),
                }
            # copy variable values (keep first non-null if duplicate)
            for k, v in row.items():
                if k in ("geo_id", "geo_name", "state_fips"):
                    continue
                by_geo[gid].setdefault(k, v)

    # Derive features.
    out: list[dict] = []
    for gid, rec in by_geo.items():
        features: dict[str, Optional[float]] = {}
        for fname, feat in FEATURES.items():
            features[fname] = _derive(feat, rec)
        pop_raw = features.get("total_population")
        pop = int(pop_raw) if pop_raw is not None else None
        out.append({
            "geo_level": geo_level,
            "geo_id": gid,
            "geo_name": rec.get("geo_name", ""),
            "state_fips": rec.get("state_fips"),
            "population": pop,
            "features_json": json.dumps(
                {k: v for k, v in features.items() if v is not None},
                separators=(",", ":"),
            ),
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
    geo_levels: list[str],
    out_db: Path,
    api_key: Optional[str],
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

    tables = sorted(tables_used())
    logger.info(
        "catalog: %d features / %d axes / %d tables",
        len(FEATURES), len(AXES), len(tables),
    )
    logger.info(
        "target: geo_levels=%s vintage=%d → %s",
        geo_levels, vintage, out_db,
    )

    if dry_run:
        tables = tables[:1]
        geo_levels = geo_levels[:1]
        logger.info("dry-run: fetching only %s / %s", tables, geo_levels)

    con = _open_db(out_db) if not dry_run else None

    stats = {"calls": 0, "rows_inserted": 0, "errors": 0}
    with httpx.Client(timeout=60.0) as client:
        for geo_level in geo_levels:
            logger.info("--- geo_level=%s ---", geo_level)
            fetched: dict[str, list[dict]] = {}
            for i, table in enumerate(tables, 1):
                try:
                    rows = _fetch_one(
                        client, table=table, geo_level=geo_level,
                        vintage=vintage, api_key=api_key,
                    )
                    fetched[table] = rows
                    stats["calls"] += 1
                    logger.info(
                        "  [%d/%d] %s → %d entities",
                        i, len(tables), table, len(rows),
                    )
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(
                        "  [%d/%d] %s FAILED: %s",
                        i, len(tables), table, e,
                    )
            if not fetched:
                logger.warning(
                    "no data for %s; skipping aggregation", geo_level,
                )
                continue
            agg = _aggregate_features(
                fetched_by_table=fetched, geo_level=geo_level,
            )
            logger.info(
                "%s: %d entities with derived features",
                geo_level, len(agg),
            )
            if dry_run:
                logger.info("dry-run: skipping DB write")
                logger.info("first 2 records: %s", agg[:2])
            else:
                _insert(con, agg, vintage)
                stats["rows_inserted"] += len(agg)

    if con:
        con.close()
    logger.info("done: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vintage", type=int, default=2023,
                   help="ACS5 vintage (default 2023).")
    p.add_argument(
        "--geo-levels", nargs="+",
        default=["place", "county", "msa"],
        choices=list(GEO_LEVELS),
        help="Geo levels to fetch (default: all three).",
    )
    p.add_argument("--out-db", type=Path, default=OUT_DB_PATH)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch one table / one level, print samples, don't write DB.",
    )
    return p.parse_args(argv)


def main():
    args = parse_args()
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        logger.warning(
            "CENSUS_API_KEY not set — unauthenticated calls are "
            "capped at 500/day. This run needs ~168."
        )
    run(
        vintage=args.vintage,
        geo_levels=args.geo_levels,
        out_db=args.out_db,
        api_key=api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
