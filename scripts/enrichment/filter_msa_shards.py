"""Filter existing MSA JSONL shards down to the allowed CBSA set.

Before the MSA filter landed in Stage 8, every `*__msa.jsonl` shard
contained rows for all ~925 CBSAs. This script reads each msa shard,
keeps only rows whose ``msa_code`` is in the current allowed set (top-N
by population + in-scope state CBSAs), and rewrites the file in place.

Atomic per-file: writes to a ``.tmp`` alongside the original, then
renames on success. A crash mid-write leaves the original intact.

Usage:
    python -m scripts.enrichment.filter_msa_shards
    python -m scripts.enrichment.filter_msa_shards --dry-run
    python -m scripts.enrichment.filter_msa_shards --top-n 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def filter_shard(
    path: Path,
    allowed_ids: set[str],
    dry_run: bool = False,
) -> dict:
    """Filter one MSA JSONL shard. Returns {total, kept, bytes_before, bytes_after}."""
    total = 0
    kept = 0
    bytes_before = path.stat().st_size
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        out_fh = None if dry_run else tmp_path.open("w", encoding="utf-8")
        with path.open("r", encoding="utf-8") as in_fh:
            for line in in_fh:
                total += 1
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupt line — skip in the filtered output
                    continue
                msa_code = rec.get("msa_code")
                if msa_code and str(msa_code) in allowed_ids:
                    kept += 1
                    if out_fh is not None:
                        out_fh.write(line)
        if out_fh is not None:
            out_fh.close()
            os.replace(tmp_path, path)
    except Exception:
        # Leave the original file untouched on any error
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    bytes_after = path.stat().st_size if not dry_run else bytes_before
    return {
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
    }


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    root = _repo_root()
    shards_dir = root / "data" / "curated" / "bigquery_loads" / "census_values"

    # Resolve the allowed CBSA set — same logic Stage 8 uses at runtime.
    from scripts.geography.schema import open_spatialite
    from scripts.ingestion.stage8_values import select_top_cbsas

    db_path = root / "data" / "geo" / "gazetteer.db"
    if not db_path.exists():
        print(f"Gazetteer not found at {db_path}", file=sys.stderr)
        return 1
    conn = open_spatialite(db_path)
    try:
        allowed = select_top_cbsas(
            conn,
            top_n=args.top_n,
            include_state_fips=args.extra_states,
        )
    finally:
        conn.close()

    if not allowed:
        print(
            "No allowed CBSAs resolved (populate_cbsa_population first). "
            "Aborting so we don't blow away every row.", file=sys.stderr,
        )
        return 1

    allowed_set = set(allowed)
    logger.info(
        "Allowed CBSAs: %d (top-%d by pop + states=%s)",
        len(allowed), args.top_n, ",".join(args.extra_states),
    )

    shards = sorted(shards_dir.glob("*__msa.jsonl"))
    if not shards:
        logger.info("No MSA shards found in %s", shards_dir)
        return 0

    total_before = 0
    total_after = 0
    total_rows = 0
    total_kept = 0
    for shard in shards:
        try:
            summary = filter_shard(shard, allowed_set, dry_run=args.dry_run)
        except Exception as e:
            logger.error("  FAILED to filter %s: %s", shard.name, e)
            continue
        total_before += summary["bytes_before"]
        total_after += summary["bytes_after"]
        total_rows += summary["total"]
        total_kept += summary["kept"]
        pct_kept = (
            100.0 * summary["kept"] / summary["total"]
            if summary["total"] else 0.0
        )
        logger.info(
            "  %-60s  %s/%s rows kept (%.1f%%)  "
            "%.1f MB → %.1f MB",
            shard.name, summary["kept"], summary["total"], pct_kept,
            summary["bytes_before"] / (1024 * 1024),
            summary["bytes_after"] / (1024 * 1024),
        )

    logger.info("=" * 70)
    logger.info(
        "TOTAL: %d shards, %d/%d rows kept (%.1f%%), "
        "%.1f GB → %.1f GB%s",
        len(shards), total_kept, total_rows,
        100.0 * total_kept / max(1, total_rows),
        total_before / (1024 ** 3),
        total_after / (1024 ** 3),
        " (DRY RUN — no files modified)" if args.dry_run else "",
    )
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--top-n", type=int, default=100,
                   help="Top-N CBSAs by population to keep (default 100)")
    p.add_argument("--extra-states", nargs="+", default=["13"],
                   metavar="STATE_FIPS",
                   help="State FIPS codes whose CBSAs are also kept "
                        "(default: 13 = Georgia)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without rewriting shards")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
