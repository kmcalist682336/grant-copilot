"""Discover HMDA coverage directly from GCS parquet instead of hand-typing years.

Since this connector serves HMDA at tract level only (no rollup to county/
state/us), coverage only needs a `geo_level='tract'` row per year that
actually has data. This script connects to the same GCS bucket your test
script already validated (HMAC creds + httpfs secret) and asks the parquet
files themselves which years exist and how many rows each year has, instead
of you guessing/typing the year range by hand.

Usage:
    python -m tools.build_hmda_coverage \
        --gcs-glob "gs://<your-bucket>/variable_tree/table_id=hmda/year=*/variable=*/part-*.parquet" \
        --table-id HMDA_LAR \
        --dataset hmda \
        --geo-level tract \
        --out-csv data/staging/hmda/hmda_coverage.csv

Requires the same GCS HMAC secret already registered for your test
connection (via `CREATE SECRET ... TYPE gcs` or equivalent) to be visible to
this DuckDB session — if you set it up in a different script/session, you'll
need to re-create it here, e.g.:

    CREATE SECRET (
        TYPE gcs,
        KEY_ID '...',
        SECRET '...'
    );

Adjust --gcs-glob to your actual bucket/prefix if it differs from the
`variable_tree/table_id=<T>/year=<Y>/variable=<V>/part-*.parquet` layout.
"""
from __future__ import annotations

import os
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def discover_years(con: duckdb.DuckDBPyConnection, gcs_glob: str) -> list[tuple[int, int]]:
    """Returns [(year, row_count), ...] sorted by year, read straight from
    the parquet files via hive-style partition inference on the path."""
    sql = f"""
        SELECT year, COUNT(*) AS row_count
        FROM read_parquet('{gcs_glob}', hive_partitioning = 1)
        GROUP BY year
        ORDER BY year
    """
    return con.execute(sql).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gcs-glob", required=True,
                     help="Glob URI over the HMDA parquet partitions, e.g. "
                          "gs://bucket/variable_tree/table_id=hmda/year=*/variable=*/part-*.parquet")
    ap.add_argument("--table-id", default="HMDA_LAR")
    ap.add_argument("--dataset", default="hmda")
    ap.add_argument("--geo-level", default="tract",
                     help="Fixed geo_level for every coverage row (this connector is tract-only).")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    con.execute("SET threads = 1")
    con.execute("SET http_timeout = 300000")
    con.execute("SET http_retries = 5")
    con.execute("SET http_retry_wait_ms = 1000")
    
    load_dotenv()

    gcs_hmac_access_id = os.getenv("GCS_HMAC_ACCESS_ID")
    gcs_hmac_secret = os.getenv("GCS_HMAC_SECRET")

    if not gcs_hmac_access_id or not gcs_hmac_secret:
        raise RuntimeError(
            "Missing GCS_HMAC_ACCESS_ID or GCS_HMAC_SECRET in the .env file."
        )

    con.execute(
        """
        CREATE OR REPLACE SECRET my_gcs_secret (
            TYPE GCS,
            KEY_ID ?,
            SECRET ?
        )
        """,
        [gcs_hmac_access_id, gcs_hmac_secret],
    )

    logger.info("Scanning %s ...", args.gcs_glob)
    year_counts = discover_years(con, args.gcs_glob)

    if not year_counts:
        logger.error("No years found — check --gcs-glob and that your GCS "
                      "secret/credentials are registered in this session.")
        return 1

    checked_at = time.time()
    rows = [
        {
            "table_id": args.table_id,
            "dataset": args.dataset,
            "year": year,
            "geo_level": args.geo_level,
            "status": "success",
            "row_count": row_count,
            "source": "gcs",
            "checked_at": checked_at,
        }
        for year, row_count in year_counts
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} coverage rows to {args.out_csv}")
    for r in rows:
        print(f"  year={r['year']}  row_count={r['row_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
