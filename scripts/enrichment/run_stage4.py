"""CLI entrypoint for Stage 4: LLM-Based Enrichment.

IMPORTANT: vLLM must be running before this script is executed.
Start it with:

    python -m vllm.entrypoints.openai.api_server \\
        --model <path_to_model_weights> \\
        --port 8000 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.90 \\
        --guided-decoding-backend outlines

Usage:
    # Step 1: Always run preflight first
    python -m scripts.enrichment.run_stage4 --preflight

    # Step 2: If preflight passes, run full enrichment
    python -m scripts.enrichment.run_stage4

    # Debug a single dataset
    python -m scripts.enrichment.run_stage4 --dataset acs/acs5 --year 2020

    # Reprocess tables already marked complete
    python -m scripts.enrichment.run_stage4 --no-skip-complete
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from scripts.enrichment.llm_client import VLLMClient, VLLMClientConfig
from scripts.enrichment.prompts import PromptLoader
from scripts.enrichment.stage4_enrichment import run_stage4
from scripts.utils.config import init_logging, load_pipeline_config, make_run_id, make_run_manifest
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_inference_config(repo_root: Path) -> dict:
    path = repo_root / "config" / "inference.yaml"
    if not path.exists():
        logger.warning("inference.yaml not found at %s — using defaults", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: LLM-based table enrichment via vLLM"
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help=(
            "Run preflight test only: enrich a small number of tables "
            "to validate the pipeline before committing to the full run. "
            "Always run this first."
        ),
    )
    parser.add_argument(
        "--preflight-count", type=int, default=None,
        help="Number of tables to process in preflight (default: from inference.yaml)"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Process only datasets matching this string (e.g., 'acs/acs5')"
    )
    parser.add_argument(
        "--year", default=None,
        help="Process only this year (e.g., '2020')"
    )
    parser.add_argument(
        "--no-skip-complete", action="store_true",
        help="Reprocess tables already marked complete"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max concurrent tables (default: from inference.yaml)"
    )
    parser.add_argument(
        "--repo-root", default=None,
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    pipeline_cfg = load_pipeline_config(repo_root)
    init_logging(pipeline_cfg)
    inference_cfg = load_inference_config(repo_root)

    paths = PipelinePaths.from_config(repo_root, pipeline_cfg)
    paths.ensure_dirs()
    state = StateTracker(paths.status_db_path)

    # Build dataset filter combining --dataset and --year
    dataset_filter = args.dataset
    if args.year and args.dataset:
        dataset_filter = args.dataset  # year filtering handled separately
    elif args.year:
        dataset_filter = args.year  # crude but effective for path matching

    # Resolve settings from config, with CLI overrides
    batching_cfg = inference_cfg.get("batching", {})
    max_concurrent = args.max_concurrent or batching_cfg.get("max_concurrent_tables", 8)
    preflight_count = args.preflight_count or batching_cfg.get("preflight_table_count", 5)

    # Build client config
    client_cfg = VLLMClientConfig.from_config(inference_cfg)

    # Load prompts
    prompts_dir = repo_root / "prompts"
    loader = PromptLoader(prompts_dir=prompts_dir, version="v1")

    # Write run manifest
    run_id = make_run_id()
    manifest = make_run_manifest(
        run_id=run_id,
        stage="stage4_enriched",
        pipeline_version=pipeline_cfg.get("pipeline_version", "unknown"),
        config_snapshot={
            "preflight": args.preflight,
            "preflight_count": preflight_count,
            "dataset_filter": dataset_filter,
            "skip_if_complete": not args.no_skip_complete,
            "max_concurrent_tables": max_concurrent,
        },
    )
    manifest_path = paths.run_manifest_path(f"{run_id}_stage4")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Stage 4 starting (run_id=%s, preflight=%s)", run_id, args.preflight)

    async with VLLMClient(client_cfg) as client:
        # Health check — always do this before any work
        healthy, msg = await client.health_check()
        if not healthy:
            logger.error("vLLM health check FAILED: %s", msg)
            logger.error(
                "\nTo start vLLM:\n"
                "  python -m vllm.entrypoints.openai.api_server \\\n"
                "    --model <path_to_model> \\\n"
                "    --port 8000 \\\n"
                "    --max-model-len 32768 \\\n"
                "    --gpu-memory-utilization 0.90 \\\n"
                "    --guided-decoding-backend outlines"
            )
            return 1
        logger.info("vLLM health check passed: %s", msg)

        if args.preflight:
            logger.info("=" * 60)
            logger.info("PREFLIGHT TEST — processing %d tables", preflight_count)
            logger.info("Review output quality before running full enrichment.")
            logger.info("=" * 60)

        summary = await run_stage4(
            paths=paths,
            state=state,
            client=client,
            loader=loader,
            inference_cfg=inference_cfg,
            skip_if_complete=not args.no_skip_complete,
            max_concurrent_tables=max_concurrent,
            dataset_filter=dataset_filter,
            preflight_only=args.preflight,
            preflight_count=preflight_count,
        )

    # Print summary
    logger.info("=" * 60)
    if args.preflight:
        logger.info("PREFLIGHT COMPLETE")
        logger.info("  Tables enriched:  %d", summary["tables_ok"])
        logger.info("  Tables failed:    %d", summary["tables_failed"])
        if summary["tables_failed"] == 0:
            logger.info("")
            logger.info("✓ Preflight passed. Review the output files, then run:")
            logger.info("  python -m scripts.enrichment.run_stage4")
        else:
            logger.error("✗ Preflight had failures. Fix issues before full run.")
    else:
        logger.info("Stage 4 COMPLETE")
        logger.info("  Datasets:         %d", summary["datasets_processed"])
        logger.info("  Tables enriched:  %d", summary["tables_ok"])
        logger.info("  Tables failed:    %d", summary["tables_failed"])
        logger.info("  Tables skipped:   %d", summary["tables_skipped"])
    logger.info("=" * 60)

    return 1 if summary["tables_failed"] > 0 else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
