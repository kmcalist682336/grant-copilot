"""CLI entrypoint for Stages 2 and 3.

Stages 2 and 3 are fast (pure in-memory computation on cached Stage 1 data)
so we run them together in a single command by default. Either can be run
independently with --stage 2 or --stage 3.

Usage:
    python -m scripts.ingestion.run_stage23
    python -m scripts.ingestion.run_stage23 --stage 2
    python -m scripts.ingestion.run_stage23 --stage 3
    python -m scripts.ingestion.run_stage23 --dataset acs/acs5
    python -m scripts.ingestion.run_stage23 --no-skip-complete
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts.utils.config import (
    init_logging,
    load_pipeline_config,
    make_run_id,
    make_run_manifest,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker
from scripts.ingestion.stage2_structure import run_stage2
from scripts.ingestion.stage3_relationships import run_stage3

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stages 2 (structural analysis) and 3 (relationship mapping)"
    )
    parser.add_argument(
        "--stage", type=int, choices=[2, 3], default=None,
        help="Run only stage 2 or stage 3 (default: run both)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Process only datasets matching this string (e.g., 'acs/acs5')"
    )
    parser.add_argument(
        "--no-skip-complete", action="store_true",
        help="Reprocess tables already marked complete"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max concurrent datasets (default: from pipeline config)"
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="Override detected repo root"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    pipeline_cfg = load_pipeline_config(repo_root)
    init_logging(pipeline_cfg)

    paths = PipelinePaths.from_config(repo_root, pipeline_cfg)
    paths.ensure_dirs()

    max_concurrent = args.max_concurrent or pipeline_cfg.get(
        "census_api", {}
    ).get("max_concurrent_requests", 4)

    state = StateTracker(paths.status_db_path)
    skip = not args.no_skip_complete
    run_id = make_run_id()

    stages_to_run = [args.stage] if args.stage else [2, 3]

    for stage_num in stages_to_run:
        stage_name = f"stage{stage_num}"
        manifest = make_run_manifest(
            run_id=run_id,
            stage=stage_name,
            pipeline_version=pipeline_cfg.get("pipeline_version", "unknown"),
            config_snapshot={
                "stage": stage_num,
                "dataset_filter": args.dataset,
                "skip_if_complete": skip,
                "max_concurrent": max_concurrent,
            },
        )
        manifest_path = paths.run_manifest_path(f"{run_id}_{stage_name}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("=" * 60)
        logger.info("Starting Stage %d  (run_id=%s)", stage_num, run_id)
        logger.info("=" * 60)

        if stage_num == 2:
            years = [str(y) for y in pipeline_cfg.get("scope", {}).get("years", [])]
            summary = run_stage2(
                years=years,
                paths=paths,
                state=state,
                max_concurrent_datasets=max_concurrent,
                skip_if_complete=skip,
                dataset_filter=args.dataset,
            )
        else:
            years = [str(y) for y in pipeline_cfg.get("scope", {}).get("years", [])]
            summary = run_stage3(
                years=years,
                paths=paths,
                state=state,
                max_concurrent_datasets=max_concurrent,
                skip_if_complete=skip,
                dataset_filter=args.dataset,
            )

        logger.info("Stage %d summary: %s", stage_num, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
