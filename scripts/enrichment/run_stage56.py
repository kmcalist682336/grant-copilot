"""CLI for Stages 5 and 6: Topic Tagging and Question Generation.

Stages 5 and 6 are driven by the same LLM infrastructure and are designed
to run in sequence: Stage 5 tags each table, then Stage 6 uses those tags
as additional context when generating questions.

Usage:
    # Run both stages in sequence (recommended)
    python -m scripts.enrichment.run_stage56

    # Run Stage 5 (tagging) only
    python -m scripts.enrichment.run_stage56 --stage 5

    # Run Stage 6 (questions) only (requires Stage 5 complete)
    python -m scripts.enrichment.run_stage56 --stage 6

    # Preflight: process 5 tables end-to-end before committing to full run
    python -m scripts.enrichment.run_stage56 --preflight

    # Target a specific dataset
    python -m scripts.enrichment.run_stage56 --dataset acs/acs5

    # Reprocess already-completed tables
    python -m scripts.enrichment.run_stage56 --no-skip-complete
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
from scripts.enrichment.stage5_tagging import (
    TopicVocabulary, ProposalAccumulator, tag_one_table,
    STAGE_NAME as STAGE5_NAME,
)
from scripts.enrichment.stage6_questions import (
    generate_questions_one_table,
    STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE,
    VARIED_SEED_FOCUSES, N_CANONICAL, N_VARIED_TOTAL, N_NEGATIVE,
)
from scripts.utils.config import init_logging, load_pipeline_config, make_run_id, make_run_manifest
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_inference_config(repo_root: Path) -> dict:
    path = repo_root / "config" / "inference.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stages 5 and 6: Topic Tagging and Question Generation"
    )
    parser.add_argument(
        "--stage", type=int, choices=[5, 6], default=None,
        help="Run only Stage 5 or Stage 6. Default: run both in sequence.",
    )
    parser.add_argument(
        "--mode", choices=["discovery", "production"], default="discovery",
        help=(
            "Stage 5 tagging mode (default: discovery). "
            "'discovery' uses the vocabulary as a preference and collects new tag "
            "proposals into data/state/tag_proposals.jsonl. Run this first, then "
            "review proposals with 'python -m scripts.diagnostics.review_tag_proposals', "
            "promote accepted tags to config/topic_vocabulary.yaml, then re-run "
            "with --mode production to apply the locked vocabulary."
        ),
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Process 5 tables end-to-end to validate before full run.",
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
        help="Reprocess tables already marked complete",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Max concurrent tables (default: from inference.yaml)",
    )
    parser.add_argument("--repo-root", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset discovery (shared with Stage 4)
# ---------------------------------------------------------------------------

def _discover_work_items(
    paths: PipelinePaths,
    dataset_filter: Optional[str] = None,
    year_filter: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Return (year, dataset_path, table_id) tuples for all processable tables."""
    processed_root = paths.processed_dir / "census"
    work_items = []

    for doc_path in sorted(processed_root.rglob("dataset_document.json")):
        parts = doc_path.relative_to(processed_root).parts
        if len(parts) < 3:
            continue
        year = parts[0]
        dataset_path = "/".join(parts[1:-1])

        if dataset_filter and dataset_filter not in dataset_path:
            continue
        if year_filter and year != year_filter:
            continue

        import json as _json
        try:
            with doc_path.open() as f:
                doc = _json.load(f)
            for table_id in doc.get("table_ids", []):
                work_items.append((year, dataset_path, table_id))
        except Exception:
            continue

    return work_items


# ---------------------------------------------------------------------------
# Stage 5 runner
# ---------------------------------------------------------------------------

async def run_stage5(
    work_items: list[tuple[str, str, str]],
    client: VLLMClient,
    loader: PromptLoader,
    vocab: TopicVocabulary,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    mode: str = "discovery",
    accumulator: Optional[ProposalAccumulator] = None,
    skip_if_complete: bool = True,
    max_concurrent: int = 8,
) -> dict:
    semaphore = asyncio.Semaphore(max_concurrent)
    ok = failed = skipped = 0

    async def _bounded(year, dataset_path, table_id):
        async with semaphore:
            if skip_if_complete and state.is_complete(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=STAGE5_NAME,
            ):
                return {"status": "skipped"}
            return await tag_one_table(
                year, dataset_path, table_id,
                client, loader, vocab, inference_cfg, paths, state,
                mode=mode,
                accumulator=accumulator,
            )

    tasks = [_bounded(*item) for item in work_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
        elif result.get("status") == "success":
            ok += 1
        elif result.get("status") == "skipped":
            skipped += 1
        else:
            failed += 1

    return {"tables_ok": ok, "tables_failed": failed, "tables_skipped": skipped}


# ---------------------------------------------------------------------------
# Stage 6 runner
# ---------------------------------------------------------------------------

async def run_stage6(
    work_items: list[tuple[str, str, str]],
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
    max_concurrent: int = 4,  # lower default — Stage 6 has 7 calls per table
) -> dict:
    semaphore = asyncio.Semaphore(max_concurrent)
    ok = failed = skipped = 0

    async def _bounded(year, dataset_path, table_id):
        async with semaphore:
            return await generate_questions_one_table(
                year, dataset_path, table_id,
                client, loader, inference_cfg, paths, state,
                skip_if_complete=skip_if_complete,
            )

    tasks = [_bounded(*item) for item in work_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
        elif result.get("status") == "success":
            ok += 1
        elif result.get("status") == "skipped":
            skipped += 1
        else:
            failed += 1

    return {"tables_ok": ok, "tables_failed": failed, "tables_skipped": skipped}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    pipeline_cfg = load_pipeline_config(repo_root)
    init_logging(pipeline_cfg)
    inference_cfg = _load_inference_config(repo_root)

    paths = PipelinePaths.from_config(repo_root, pipeline_cfg)
    paths.ensure_dirs()
    state = StateTracker(paths.status_db_path)

    run_stage5_flag = args.stage in (None, 5)
    run_stage6_flag = args.stage in (None, 6)

    dataset_filter = args.dataset
    year_filter = args.year
    skip = not args.no_skip_complete
    batching = inference_cfg.get("batching", {})
    max_concurrent_s5 = args.max_concurrent or batching.get("max_concurrent_tables", 8)
    max_concurrent_s6 = args.max_concurrent or batching.get("max_concurrent_tables_s6", 4)

    # Load vocabulary (Stage 5)
    vocab = TopicVocabulary.from_config_dir(repo_root / "config")
    logger.info(
        "Topic vocabulary: %d L1 domains, %d L2 subjects",
        len(vocab.l1_domains), len(vocab.l2_subjects),
    )

    loader = PromptLoader(prompts_dir=repo_root / "prompts", version="v1")
    client_cfg = VLLMClientConfig.from_config(inference_cfg)

    async with VLLMClient(client_cfg) as client:
        # Health check
        healthy, msg = await client.health_check()
        if not healthy:
            logger.error("vLLM health check FAILED: %s", msg)
            logger.error(
                "\nStart vLLM:\n"
                "  python -m vllm.entrypoints.openai.api_server \\\n"
                "    --model models/gemma_weights/gemma-4-e4b-it \\\n"
                "    --port 8000 --max-model-len 32768 \\\n"
                "    --gpu-memory-utilization 0.90"
            )
            return 1
        logger.info(msg)

        # Discover work
        all_items = _discover_work_items(paths, dataset_filter, year_filter)

        if args.preflight:
            all_items = all_items[:args.preflight_count]
            logger.info("PREFLIGHT: processing %d tables", len(all_items))

        logger.info("Total tables to process: %d", len(all_items))

        # ── Stage 5 ──────────────────────────────────────────────────────
        if run_stage5_flag:
            mode = args.mode
            logger.info("=" * 60)
            logger.info("STAGE 5 — Topic Tagging (%s mode)", mode.upper())
            if mode == "discovery":
                logger.info(
                    "  Vocabulary is a preference — new tags will be proposed"
                )
                logger.info(
                    "  Proposals → data/state/tag_proposals.jsonl"
                )
            else:
                logger.info("  Vocabulary is locked — no new tags accepted")
            logger.info("=" * 60)

            accumulator = None
            if mode == "discovery":
                proposals_path = paths.state_dir / "tag_proposals.jsonl"
                accumulator = ProposalAccumulator(proposals_path)

            s5_summary = await run_stage5(
                all_items, client, loader, vocab, inference_cfg,
                paths, state,
                mode=mode,
                accumulator=accumulator,
                skip_if_complete=skip,
                max_concurrent=max_concurrent_s5,
            )

            if mode == "discovery" and accumulator:
                accumulator.flush()
                if accumulator.total_proposals > 0:
                    logger.info("")
                    logger.info(
                        "Discovery complete: %d unique tag proposals collected.",
                        accumulator.total_proposals,
                    )
                    logger.info("Review with:")
                    logger.info(
                        "  python -m scripts.diagnostics.review_tag_proposals"
                    )
                    logger.info(
                        "Then add approved tags to config/topic_vocabulary.yaml"
                    )
                    logger.info("and re-run with --mode production.")
                else:
                    logger.info(
                        "Discovery complete: no new tag proposals — "
                        "vocabulary appears complete. Safe to run --mode production."
                    )

            logger.info(
                "Stage 5 (%s): %d ok, %d failed, %d skipped",
                mode,
                s5_summary["tables_ok"],
                s5_summary["tables_failed"],
                s5_summary["tables_skipped"],
            )

        # ── Stage 6 ──────────────────────────────────────────────────────
        if run_stage6_flag:
            logger.info("=" * 60)
            logger.info("STAGE 6 — Question Generation")
            logger.info(
                "  %d canonical + ~%d varied + %d negative per table",
                N_CANONICAL, N_VARIED_TOTAL, N_NEGATIVE,
            )
            logger.info("  %d seed focuses for varied batches", len(VARIED_SEED_FOCUSES))
            logger.info("=" * 60)
            s6_summary = await run_stage6(
                all_items, client, loader, inference_cfg,
                paths, state,
                skip_if_complete=skip,
                max_concurrent=max_concurrent_s6,
            )
            logger.info(
                "Stage 6 complete: %d ok, %d failed, %d skipped",
                s6_summary["tables_ok"],
                s6_summary["tables_failed"],
                s6_summary["tables_skipped"],
            )

    if args.preflight:
        logger.info("")
        logger.info("PREFLIGHT complete. Review outputs:")
        logger.info(
            "  find data/processed/census -name 'stage5_tags.json' | head -3"
        )
        logger.info(
            "  find data/processed/census -name 'stage6_questions_canonical.json'"
            " | head -3"
        )
        logger.info("")
        logger.info("If quality looks good, run the full pipeline:")
        logger.info("  python -m scripts.enrichment.run_stage56")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
