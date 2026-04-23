"""CLI entrypoint for running Stage 1 (metadata collection).

Usage:
    python -m scripts.ingestion.run_stage1
    python -m scripts.ingestion.run_stage1 --years 2020 2021
    python -m scripts.ingestion.run_stage1 --force-refresh
    python -m scripts.ingestion.run_stage1 --no-skip-complete
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.utils.census_api import CensusAPIClient, CensusAPIConfig
from scripts.utils.config import (
    init_logging,
    load_datasets_allowlist,
    load_pipeline_config,
    make_run_id,
    make_run_manifest,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker
from scripts.ingestion.stage1_metadata import STAGE_NAME, run_stage1

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Find the repo root by walking up from this file."""
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 (metadata collection) of the Census pipeline",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=None,
        help="Years to process (default: from pipeline config). "
        "Overrides scope.years from pipeline.yaml.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached raw files and re-fetch from Census API.",
    )
    parser.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Re-process datasets even if state DB marks them complete.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent dataset fetches. Defaults to "
        "census_api.max_concurrent_requests from pipeline config.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Process only the named dataset path (e.g., 'acs/acs5/profile'). "
        "Useful for debugging a single dataset.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override the detected repo root (for testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    pipeline_cfg = load_pipeline_config(repo_root)
    init_logging(pipeline_cfg)

    allowlist_cfg = load_datasets_allowlist(repo_root)
    paths = PipelinePaths.from_config(repo_root, pipeline_cfg)
    paths.ensure_dirs()

    # Resolve years
    years = args.years
    if years is None:
        years = [str(y) for y in pipeline_cfg.get("scope", {}).get("years", [])]
    else:
        years = [str(y) for y in years]
    if not years:
        logger.error("No years specified. Set scope.years in pipeline.yaml or pass --years.")
        return 2

    # If user specified a single dataset, mutate the allowlist accordingly
    if args.dataset:
        allowlist_cfg = {
            **allowlist_cfg,
            "mode": "allowlist",
            "datasets": [{"path": args.dataset, "tier": 1}],
        }

    # Resolve concurrency
    max_concurrent = args.max_concurrent
    if max_concurrent is None:
        max_concurrent = pipeline_cfg.get("census_api", {}).get("max_concurrent_requests", 4)

    # Build client and state tracker
    api_config = CensusAPIConfig.from_config(pipeline_cfg)
    client = CensusAPIClient(api_config)
    state = StateTracker(paths.status_db_path)

    # Write a run manifest
    run_id = make_run_id()
    manifest = make_run_manifest(
        run_id=run_id,
        stage=STAGE_NAME,
        pipeline_version=pipeline_cfg.get("pipeline_version", "unknown"),
        config_snapshot={
            "years": years,
            "force_refresh": args.force_refresh,
            "skip_if_complete": not args.no_skip_complete,
            "max_concurrent": max_concurrent,
            "dataset_override": args.dataset,
            "allowlist_mode": allowlist_cfg.get("mode"),
        },
    )
    manifest_path = paths.run_manifest_path(run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Run ID: %s (manifest: %s)", run_id, manifest_path)

    # Execute
    summary = run_stage1(
        years=years,
        allowlist_config=allowlist_cfg,
        client=client,
        paths=paths,
        state=state,
        max_concurrent_datasets=max_concurrent,
        force_refresh=args.force_refresh,
        skip_if_complete=not args.no_skip_complete,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Stage 1 complete")
    logger.info("  Datasets succeeded:  %d", summary["datasets_succeeded"])
    logger.info("  Datasets failed:     %d", summary["datasets_failed"])
    logger.info("  Datasets skipped:    %d", summary["datasets_skipped"])
    logger.info("  Tables written:      %d", summary["tables_written"])
    logger.info("=" * 60)

    # Status counts from the DB
    stage_summary = state.stage_summary(STAGE_NAME)
    logger.info("State DB status counts for %s: %s", STAGE_NAME, stage_summary)

    # Exit code reflects whether any dataset failed
    return 1 if summary["datasets_failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
