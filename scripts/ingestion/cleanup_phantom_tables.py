"""One-shot cleanup: remove phantom table folders created by the pre-fix
Stage 1 implementation, which treated comma-separated group fields as
distinct table IDs.

This is safe to run against any existing data/processed tree. It identifies
any table folder whose name contains a comma (no real Census table ID does)
and removes the folder along with its state DB entries.

After running this, re-run Stage 1 normally. The raw cache is untouched, so
the re-run will be fast — it just re-partitions variables into the correct
folders.

Usage:
    python -m scripts.ingestion.cleanup_phantom_tables
    python -m scripts.ingestion.cleanup_phantom_tables --dry-run
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from scripts.utils.config import init_logging, load_pipeline_config
from scripts.utils.paths import PipelinePaths

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_phantom_table_dirs(paths: PipelinePaths) -> list[Path]:
    """Find all processed table directories whose name contains a comma.

    Real Census table IDs never contain commas — they're alphanumeric (e.g.,
    B01001, CP02, DP05, S0101, CP02PR).
    """
    processed_root = paths.processed_dir / "census"
    if not processed_root.exists():
        return []

    phantoms = []
    # Processed layout: census/<year>/<dataset_path>/tables/<table_id>/
    for tables_dir in processed_root.rglob("tables"):
        if not tables_dir.is_dir():
            continue
        for entry in tables_dir.iterdir():
            if entry.is_dir() and "," in entry.name:
                phantoms.append(entry)
    return phantoms


def find_phantom_raw_dirs(paths: PipelinePaths) -> list[Path]:
    """Same check for the raw/ tree — Stage 8 hasn't run yet, but be thorough."""
    raw_root = paths.raw_dir / "census"
    if not raw_root.exists():
        return []
    phantoms = []
    for tables_dir in raw_root.rglob("tables"):
        if not tables_dir.is_dir():
            continue
        for entry in tables_dir.iterdir():
            if entry.is_dir() and "," in entry.name:
                phantoms.append(entry)
    return phantoms


def cleanup_state_db(db_path: Path, dry_run: bool = False) -> int:
    """Remove state DB entries for entities whose entity_id contains a comma."""
    if not db_path.exists():
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        # Find phantom entries
        cur = conn.execute(
            "SELECT year, dataset_path, entity_id, stage FROM stage_status "
            "WHERE entity_type = 'table' AND entity_id LIKE '%,%'"
        )
        rows = cur.fetchall()
        for year, ds, eid, stage in rows:
            logger.info("  State DB phantom: %s::%s::%s [stage=%s]", year, ds, eid, stage)

        if dry_run:
            return len(rows)

        conn.execute(
            "DELETE FROM stage_status WHERE entity_type = 'table' AND entity_id LIKE '%,%'"
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove phantom CSV-group table folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed, but don't touch anything")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    cfg = load_pipeline_config(repo_root)
    init_logging(cfg)
    paths = PipelinePaths.from_config(repo_root, cfg)

    processed_phantoms = find_phantom_table_dirs(paths)
    raw_phantoms = find_phantom_raw_dirs(paths)

    logger.info("Found %d phantom folders in processed/", len(processed_phantoms))
    for p in processed_phantoms:
        logger.info("  %s", p)
    logger.info("Found %d phantom folders in raw/", len(raw_phantoms))
    for p in raw_phantoms:
        logger.info("  %s", p)

    state_count = cleanup_state_db(paths.status_db_path, dry_run=args.dry_run)
    logger.info("Phantom state DB entries: %d", state_count)

    if args.dry_run:
        logger.info("Dry run — no changes made. Re-run without --dry-run to apply.")
        return 0

    for phantom in processed_phantoms + raw_phantoms:
        logger.info("Removing %s", phantom)
        shutil.rmtree(phantom)

    logger.info("Cleanup complete. Re-run Stage 1 to rebuild with correct partitioning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
