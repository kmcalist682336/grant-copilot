"""CLI entrypoint for Stage 7: Variable-Level Enrichment.

IMPORTANT: vLLM must be running before this script is executed with
prefix caching enabled:

    python -m vllm.entrypoints.openai.api_server \\
        --model models/gemma_weights/gemma-4-e4b-it \\
        --port 8000 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.90 \\
        --enable-prefix-caching

Stage 7 uses a SQLite cache to deduplicate LLM work across years — variables
with identical labels in multiple years share a single cache entry. Expect
very high cache hit rates after the first year is processed.

Usage:
    # Preflight — process 5 tables to validate
    python -m scripts.enrichment.run_stage7 --preflight

    # Full run
    python -m scripts.enrichment.run_stage7

    # Restrict to one dataset or year
    python -m scripts.enrichment.run_stage7 --dataset acs/acs5 --year 2020

    # Reprocess tables already marked complete (does NOT clear cache)
    python -m scripts.enrichment.run_stage7 --no-skip-complete

    # Clear the variable cache (forces regeneration from scratch)
    python -m scripts.enrichment.run_stage7 --clear-cache
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from scripts.enrichment.llm_client import VLLMClient, VLLMClientConfig
from scripts.enrichment.prompts import PromptLoader
from scripts.enrichment.stage7_variables import (
    STAGE_NAME, process_one_table,
)
from scripts.enrichment.variable_cache import VariableCache
from scripts.utils.config import (
    init_logging, load_pipeline_config, make_run_id, make_run_manifest,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Work discovery (shared with other stages)
# ---------------------------------------------------------------------------

def _discover_work_items(
    paths: PipelinePaths,
    dataset_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Return (year, dataset_path, table_id) tuples for every table that
    has stage2 AND stage4 outputs (the prerequisites for Stage 7)."""
    processed_root = paths.processed_dir / "census"
    work_items = []

    if not processed_root.exists():
        return work_items

    for year_dir in sorted(processed_root.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        if year_filter and year != year_filter:
            continue

        for s2_path in year_dir.rglob("stage2_structure.json"):
            table_dir = s2_path.parent
            if table_dir.name == "tables" or table_dir.parent.name != "tables":
                continue
            table_id = table_dir.name
            dataset_root = table_dir.parent.parent
            try:
                dataset_path = str(
                    dataset_root.relative_to(year_dir)
                ).replace("\\", "/")
            except ValueError:
                continue

            if dataset_filter and dataset_filter not in dataset_path:
                continue

            # Require stage4 output too
            s4_path = table_dir / "stage4_enriched.json"
            if not s4_path.exists():
                continue

            work_items.append((year, dataset_path, table_id))

    return sorted(work_items)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 7: Variable-Level Enrichment"
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Process N tables end-to-end to validate before full run",
    )
    parser.add_argument(
        "--preflight-count", type=int, default=5,
        help="Number of tables for preflight (default: 5)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter to datasets matching this string (e.g., 'acs/acs5')",
    )
    parser.add_argument(
        "--year", default=None,
        help="Filter to a specific year (e.g., '2020')",
    )
    parser.add_argument(
        "--no-skip-complete", action="store_true",
        help="Reprocess tables already marked complete (cache still used)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max concurrent tables (default: from inference.yaml)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the variable cache database before running",
    )
    parser.add_argument("--repo-root", default=None)
    return parser.parse_args(argv)


async def run_stage7(
    work_items: list[tuple[str, str, str]],
    client: VLLMClient,
    loader: PromptLoader,
    cache: VariableCache,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
    max_concurrent: int = 8,
) -> dict:
    semaphore = asyncio.Semaphore(max_concurrent)
    ok = failed = skipped = 0
    total_desc_hits = total_desc_misses = 0
    total_quest_hits = total_quest_misses = 0

    async def _bounded(year, dataset_path, table_id):
        async with semaphore:
            return await process_one_table(
                year, dataset_path, table_id,
                client, loader, cache, inference_cfg, paths, state,
                skip_if_complete=skip_if_complete,
            )

    tasks = [_bounded(*item) for item in work_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
        elif result.get("status") == "success":
            ok += 1
            total_desc_hits += result.get("desc_hits", 0)
            total_desc_misses += result.get("desc_misses", 0)
            total_quest_hits += result.get("quest_hits", 0)
            total_quest_misses += result.get("quest_misses", 0)
        elif result.get("status") == "skipped":
            skipped += 1
        else:
            failed += 1

    return {
        "tables_ok": ok,
        "tables_failed": failed,
        "tables_skipped": skipped,
        "total_desc_hits": total_desc_hits,
        "total_desc_misses": total_desc_misses,
        "total_quest_hits": total_quest_hits,
        "total_quest_misses": total_quest_misses,
    }


async def async_main(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    config = load_pipeline_config(repo_root)
    with (repo_root / "config" / "inference.yaml").open() as f:
        inference_cfg = yaml.safe_load(f)

    paths = PipelinePaths.from_config(repo_root, config)
    paths.ensure_dirs()

    run_id = make_run_id()
    init_logging(config)

    logger.info("=" * 60)
    logger.info("STAGE 7 — Variable-Level Enrichment")
    logger.info("=" * 60)

    cache_path = paths.state_dir / "variable_cache.sqlite"

    if args.clear_cache:
        if cache_path.exists():
            logger.warning("Clearing variable cache at %s", cache_path)
            cache_path.unlink()
            for suffix in ("-wal", "-shm"):
                extra = cache_path.with_suffix(cache_path.suffix + suffix)
                if extra.exists():
                    extra.unlink()

    cache = VariableCache(cache_path)
    stats = cache.stats()
    logger.info(
        "Variable cache: %d descriptions, %d question entries",
        stats["description_entries"], stats["question_entries"],
    )

    client_cfg = VLLMClientConfig.from_config(inference_cfg)

    async with VLLMClient(client_cfg) as client:
        if not await client.health_check():
            logger.error(
                "vLLM server is not healthy at %s:%d",
                client_cfg.host, client_cfg.port,
            )
            return 1
        logger.info(
            "vLLM server is healthy — serving model: %s", client.config.model_id
        )

        loader = PromptLoader(prompts_dir=repo_root / "prompts", version="v1")
        state = StateTracker(paths.status_db_path)

        work_items = _discover_work_items(
            paths, dataset_filter=args.dataset, year_filter=args.year,
        )
        if args.preflight:
            work_items = work_items[:args.preflight_count]
            logger.info("PREFLIGHT: processing %d tables", len(work_items))

        if not work_items:
            logger.warning("No tables to process — Stage 7 has no work")
            return 0

        logger.info("Total tables to process: %d", len(work_items))

        batching = inference_cfg.get("batching", {})
        max_concurrent = args.max_concurrent or batching.get(
            "max_concurrent_tables_s7", 8
        )

        run_manifest = make_run_manifest(
            run_id=run_id,
            stage=STAGE_NAME,
            pipeline_version=config.get("pipeline_version", "unknown"),
            config_snapshot=inference_cfg,
        )

        summary = await run_stage7(
            work_items, client, loader, cache, inference_cfg,
            paths, state,
            skip_if_complete=(not args.no_skip_complete),
            max_concurrent=max_concurrent,
        )

        run_manifest["summary"] = summary
        with paths.run_manifest_path(run_id).open("w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        logger.info("=" * 60)
        logger.info("Stage 7 complete")
        logger.info("  Tables: %d ok, %d failed, %d skipped",
                    summary["tables_ok"], summary["tables_failed"],
                    summary["tables_skipped"])
        total_desc = summary["total_desc_hits"] + summary["total_desc_misses"]
        total_quest = summary["total_quest_hits"] + summary["total_quest_misses"]
        if total_desc > 0:
            hit_rate = summary["total_desc_hits"] / total_desc
            logger.info(
                "  Descriptions: %d hits + %d misses (%.1f%% cache hit rate)",
                summary["total_desc_hits"], summary["total_desc_misses"],
                hit_rate * 100,
            )
        if total_quest > 0:
            hit_rate = summary["total_quest_hits"] / total_quest
            logger.info(
                "  Questions:    %d hits + %d misses (%.1f%% cache hit rate)",
                summary["total_quest_hits"], summary["total_quest_misses"],
                hit_rate * 100,
            )
        final_stats = cache.stats()
        logger.info(
            "  Final cache: %d descriptions, %d question entries",
            final_stats["description_entries"], final_stats["question_entries"],
        )
        logger.info("=" * 60)

    return 0 if summary["tables_failed"] == 0 else 1


def main() -> int:
    args = parse_args(sys.argv[1:])
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
