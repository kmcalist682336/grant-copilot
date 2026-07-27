"""Load hand-authored HMDA metadata CSVs into the `tables` / `variables` /
`coverage` tables in the metadata DuckDB — Step 1 of NEW_DATASET_RUNBOOK.md /
Section 2 of DATASET_INTEGRATION.md, translated to DuckDB DDL.

This is the step BEFORE append_hmda_cards.py. append_hmda_cards.py (Step 4)
joins your question CSVs against these three tables, so this script must be
run first, and your table_id / dataset / variable_id values must match
exactly between here and your question CSVs.

Schemas below are copied verbatim (types translated to DuckDB) from
DATASET_INTEGRATION.md Section 2.1-2.3 — same columns, same primary keys,
same CHECK constraint on coverage.status.

Expected CSV columns:
    tables.csv    -> table_id, dataset, year, title, description, use_when,
                      avoid_when, universe, universe_plain, subject_area,
                      measure_type, max_depth, l1_domains_json
    variables.csv -> table_id, dataset, year, variable_id, label,
                      label_natural, description, column_name, depth,
                      is_total, is_commonly_queried, kind
    coverage.csv  -> table_id, dataset, year, geo_level, status, row_count,
                      source, checked_at

Usage:
    python -m tools.load_hmda_metadata --dry-run \
        --dataset gcs/hmda_v1 \
        --tables-csv data/staging/hmda/hmda_tables.csv \
        --variables-csv data/staging/hmda/hmda_variables.csv \
        --coverage-csv data/staging/hmda/hmda_coverage.csv

    python -m tools.load_hmda_metadata \
        --metadata-db data/metadata/search_index.duckdb \
        --dataset gcs/hmda_v1 \
        --tables-csv data/staging/hmda/hmda_tables.csv \
        --variables-csv data/staging/hmda/hmda_variables.csv \
        --coverage-csv data/staging/hmda/hmda_coverage.csv \
        -v
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DB = REPO_ROOT / "data" / "metadata" / "search_index.duckdb"


def _csv_view_sql(view_name: str, csv_path: Path) -> str:
    """DuckDB can't bind a prepared-statement parameter inside CREATE VIEW,
    so the path is inlined (escaped) rather than passed positionally."""
    escaped = str(csv_path).replace("'", "''")
    return (
        f"CREATE OR REPLACE TEMP VIEW {view_name} AS "
        f"SELECT * FROM read_csv_auto('{escaped}', header=true)"
    )


# ---------------------------------------------------------------------------
# Schema — verbatim from DATASET_INTEGRATION.md Section 2.1-2.3
# ---------------------------------------------------------------------------

_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS tables (
    table_id        TEXT NOT NULL,
    dataset         TEXT NOT NULL,       -- namespace, e.g. 'gcs/hmda_v1'
    year            INTEGER NOT NULL,
    title           TEXT,
    description     TEXT,
    use_when        TEXT,
    avoid_when      TEXT,
    universe        TEXT,
    universe_plain  TEXT,
    subject_area    TEXT,                -- e.g. 'Economics/Public Assistance'
    measure_type    TEXT,                -- median|count|percentage|currency|ratio|index
    max_depth       INTEGER,
    l1_domains_json TEXT,                -- JSON array
    PRIMARY KEY (table_id, dataset, year)
);

CREATE TABLE IF NOT EXISTS variables (
    table_id             TEXT NOT NULL,
    dataset              TEXT NOT NULL,
    year                 INTEGER NOT NULL,
    variable_id          TEXT NOT NULL,  -- the addressable cell
    label                TEXT,
    label_natural        TEXT,
    description          TEXT,
    column_name          TEXT,
    depth                INTEGER,        -- 1 = universe/total; 2+ = sub-breakdown
    is_total             INTEGER,        -- 1 for the universe/overall cell
    is_commonly_queried  INTEGER,        -- 1 for "real answer" cells
    kind                 TEXT,
    PRIMARY KEY (table_id, dataset, year, variable_id)
);

CREATE TABLE IF NOT EXISTS coverage (
    table_id   TEXT NOT NULL,
    dataset    TEXT NOT NULL,
    year       INTEGER NOT NULL,
    geo_level  TEXT NOT NULL,            -- 'tract'|'county'|'msa'|'state'|'us'|...
    status     TEXT NOT NULL CHECK (status IN ('success', 'no_data')),
    row_count  INTEGER,
    source     TEXT NOT NULL,
    checked_at DOUBLE NOT NULL,
    PRIMARY KEY (table_id, dataset, year, geo_level)
);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_METADATA_DDL)


# ---------------------------------------------------------------------------
# Loaders — each is an idempotent upsert scoped to --dataset
# ---------------------------------------------------------------------------

def load_tables(con: duckdb.DuckDBPyConnection, csv_path: Path, *, dataset: str) -> int:
    con.execute(_csv_view_sql("_staged_tables", csv_path))
    before = con.execute("SELECT COUNT(*) FROM tables WHERE dataset = ?", [dataset]).fetchone()[0]
    con.execute("""
        INSERT OR REPLACE INTO tables
        SELECT table_id, dataset, year, title, description, use_when, avoid_when,
               universe, universe_plain, subject_area, measure_type, max_depth,
               l1_domains_json
        FROM _staged_tables
        WHERE dataset = ?
    """, [dataset])
    after = con.execute("SELECT COUNT(*) FROM tables WHERE dataset = ?", [dataset]).fetchone()[0]
    return after - before


def load_variables(con: duckdb.DuckDBPyConnection, csv_path: Path, *, dataset: str) -> int:
    con.execute(_csv_view_sql("_staged_variables", csv_path))
    before = con.execute("SELECT COUNT(*) FROM variables WHERE dataset = ?", [dataset]).fetchone()[0]
    con.execute("""
        INSERT OR REPLACE INTO variables
        SELECT table_id, dataset, year, variable_id, label, label_natural,
               description, column_name, depth, is_total, is_commonly_queried, kind
        FROM _staged_variables
        WHERE dataset = ?
    """, [dataset])
    after = con.execute("SELECT COUNT(*) FROM variables WHERE dataset = ?", [dataset]).fetchone()[0]
    return after - before


def load_coverage(con: duckdb.DuckDBPyConnection, csv_path: Path, *, dataset: str) -> int:
    con.execute(_csv_view_sql("_staged_coverage", csv_path))
    before = con.execute("SELECT COUNT(*) FROM coverage WHERE dataset = ?", [dataset]).fetchone()[0]
    con.execute("""
        INSERT OR REPLACE INTO coverage
        SELECT table_id, dataset, year, geo_level, status, row_count, source,
               COALESCE(checked_at, ?)
        FROM _staged_coverage
        WHERE dataset = ?
    """, [time.time(), dataset])
    after = con.execute("SELECT COUNT(*) FROM coverage WHERE dataset = ?", [dataset]).fetchone()[0]
    return after - before


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    metadata_db: Path,
    dataset: str,
    tables_csv: Path | None = None,
    variables_csv: Path | None = None,
    coverage_csv: Path | None = None,
    dry_run: bool = False,
) -> dict:
    metadata_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(metadata_db))
    try:
        ensure_schema(con)

        if dry_run:
            stats = {}
            if tables_csv:
                con.execute(_csv_view_sql("_preview_tables", tables_csv))
                stats["table_rows_in_csv"] = con.execute(
                    "SELECT COUNT(*) FROM _preview_tables WHERE dataset = ?", [dataset]
                ).fetchone()[0]
            if variables_csv:
                con.execute(_csv_view_sql("_preview_variables", variables_csv))
                stats["variable_rows_in_csv"] = con.execute(
                    "SELECT COUNT(*) FROM _preview_variables WHERE dataset = ?", [dataset]
                ).fetchone()[0]
            if coverage_csv:
                con.execute(_csv_view_sql("_preview_coverage", coverage_csv))
                stats["coverage_rows_in_csv"] = con.execute(
                    "SELECT COUNT(*) FROM _preview_coverage WHERE dataset = ?", [dataset]
                ).fetchone()[0]
            logger.info("[dry-run] no rows written")
            return stats

        stats: dict = {}
        if tables_csv:
            n = load_tables(con, tables_csv, dataset=dataset)
            stats["tables_rows_written"] = n
            logger.info("tables rows written: %d", n)
        if variables_csv:
            n = load_variables(con, variables_csv, dataset=dataset)
            stats["variables_rows_written"] = n
            logger.info("variables rows written: %d", n)
        if coverage_csv:
            n = load_coverage(con, coverage_csv, dataset=dataset)
            stats["coverage_rows_written"] = n
            logger.info("coverage rows written: %d", n)

        stats["is_commonly_queried_count"] = con.execute(
            "SELECT COUNT(*) FROM variables WHERE dataset = ? AND is_commonly_queried = 1",
            [dataset],
        ).fetchone()[0]
        return stats
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--metadata-db", default=str(DEFAULT_METADATA_DB))
    ap.add_argument(
        "--dataset", required=True,
        help="Namespace for the HMDA data, e.g. 'gcs/hmda_v1' — must match "
             "the dataset value used in every CSV and in append_hmda_cards.py.",
    )
    ap.add_argument("--tables-csv", type=Path, default=None)
    ap.add_argument("--variables-csv", type=Path, default=None)
    ap.add_argument("--coverage-csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not (args.tables_csv or args.variables_csv or args.coverage_csv):
        logger.error("Provide at least one of --tables-csv / --variables-csv / --coverage-csv.")
        return 1

    stats = run(
        metadata_db=Path(args.metadata_db),
        dataset=args.dataset,
        tables_csv=args.tables_csv,
        variables_csv=args.variables_csv,
        coverage_csv=args.coverage_csv,
        dry_run=args.dry_run,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
