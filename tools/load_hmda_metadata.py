"""Load hand-authored HMDA metadata CSVs into the REAL search_index.db
(SQLite) — Step 1 of DATASET_INTEGRATION.md Section 4: populate the three
existing metadata tables (tables/variables/coverage), scoped by `dataset`.
No schema changes; these tables are shared with Census.

This replaces the earlier (incorrect) DuckDB version — same logic, correct
database engine and file.

Usage:
    python -m tools.load_hmda_metadata --dry-run \
        --dataset hmda \
        --tables-csv data/staging/hmda/hmda_tables.csv \
        --variables-csv data/staging/hmda/hmda_variables.csv \
        --coverage-csv data/staging/hmda/hmda_coverage.csv

    python -m tools.load_hmda_metadata \
        --metadata-db data/metadata/search_index.db \
        --dataset hmda \
        --tables-csv data/staging/hmda/hmda_tables.csv \
        --variables-csv data/staging/hmda/hmda_variables.csv \
        --coverage-csv data/staging/hmda/hmda_coverage.csv \
        -v
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DB = REPO_ROOT / "data" / "metadata" / "search_index.db"

TABLES_COLS = [
    "table_id", "dataset", "year", "title", "description", "use_when",
    "avoid_when", "universe", "universe_plain", "subject_area",
    "measure_type", "max_depth", "l1_domains_json",
]
VARIABLES_COLS = [
    "table_id", "dataset", "year", "variable_id", "label", "label_natural",
    "description", "column_name", "depth", "is_total",
    "is_commonly_queried", "kind",
]
COVERAGE_COLS = [
    "table_id", "dataset", "year", "geo_level", "status", "row_count",
    "source", "checked_at",
]


def _existing_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_generic(
    db: sqlite3.Connection, csv_path: Path, *, table: str,
    wanted_cols: list[str], dataset: str,
) -> int:
    rows = [r for r in _read_csv(csv_path) if r.get("dataset", "").strip() == dataset]
    if not rows:
        return 0
    cols = [c for c in wanted_cols if c in _existing_columns(db, table)]
    missing = set(wanted_cols) - set(cols)
    if missing:
        logger.warning("%s: columns not in live schema, skipped: %s", table, sorted(missing))

    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    values = [tuple((r.get(c) or None) for c in cols) for r in rows]
    db.executemany(sql, values)
    return len(values)


def run(
    metadata_db: Path, dataset: str, *,
    tables_csv: Path | None = None,
    variables_csv: Path | None = None,
    coverage_csv: Path | None = None,
    dry_run: bool = False,
) -> dict:
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    db = sqlite3.connect(str(metadata_db))
    db.execute("PRAGMA journal_mode=WAL")
    try:
        stats: dict = {}

        if dry_run:
            if tables_csv:
                stats["table_rows_in_csv"] = len(
                    [r for r in _read_csv(tables_csv) if r.get("dataset", "").strip() == dataset]
                )
            if variables_csv:
                stats["variable_rows_in_csv"] = len(
                    [r for r in _read_csv(variables_csv) if r.get("dataset", "").strip() == dataset]
                )
            if coverage_csv:
                stats["coverage_rows_in_csv"] = len(
                    [r for r in _read_csv(coverage_csv) if r.get("dataset", "").strip() == dataset]
                )
            logger.info("[dry-run] no rows written")
            return stats

        if tables_csv:
            n = _load_generic(db, tables_csv, table="tables", wanted_cols=TABLES_COLS, dataset=dataset)
            stats["tables_rows_written"] = n
            logger.info("tables rows written: %d", n)

        if variables_csv:
            n = _load_generic(db, variables_csv, table="variables", wanted_cols=VARIABLES_COLS, dataset=dataset)
            stats["variables_rows_written"] = n
            logger.info("variables rows written: %d", n)

        if coverage_csv:
            rows = [r for r in _read_csv(coverage_csv) if r.get("dataset", "").strip() == dataset]
            for r in rows:
                if not r.get("checked_at"):
                    r["checked_at"] = str(time.time())
            cols = [c for c in COVERAGE_COLS if c in _existing_columns(db, "coverage")]
            placeholders = ",".join("?" for _ in cols)
            sql = f"INSERT OR REPLACE INTO coverage ({','.join(cols)}) VALUES ({placeholders})"
            db.executemany(sql, [tuple((r.get(c) or None) for c in cols) for r in rows])
            stats["coverage_rows_written"] = len(rows)
            logger.info("coverage rows written: %d", len(rows))

        db.commit()

        stats["is_commonly_queried_count"] = db.execute(
            "SELECT COUNT(*) FROM variables WHERE dataset = ? AND is_commonly_queried = 1",
            (dataset,),
        ).fetchone()[0]
        return stats
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--metadata-db", default=str(DEFAULT_METADATA_DB))
    ap.add_argument("--dataset", required=True)
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
