"""Stage 6: Synthetic Question Generation.

Reads stage4_enriched.json, stage5_tags.json, and stage3_relationships.json
per table and generates three question sets:

  stage6_questions_canonical.json  — 10 anchor questions (temp 0.3)
  stage6_questions_varied.json     — 20 varied questions with intent metadata
  stage6_questions_negative.json   — 10 negative examples with type and reason

Total: 40 questions per table × 16,529 tables ≈ 660K questions, 3 LLM calls per table.

Volume tiers (based on commonly_queried_variables count from Stage 3):
  full  — canonical + varied + negatives (tables with CQ > 0)
  light — canonical + negatives only (narrow single-concept tables, CQ == 0)
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.enrichment.llm_client import VLLMClient, LLMCallError
from scripts.enrichment.prompts import PromptLoader, build_sample_labels
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_CANONICAL = "stage6_questions_canonical"
STAGE_VARIED    = "stage6_questions_varied"
STAGE_NEGATIVE  = "stage6_questions_negative"

DEPENDS_ON_S3 = "stage3_relationships"
DEPENDS_ON_S4 = "stage4_enriched"
DEPENDS_ON_S5 = "stage5_tags"

# Number of questions per output
# 10 canonical + 20 varied (1 batch) + 10 negative = 40 per table
N_CANONICAL    = 10
N_VARIED_BATCH = 20
N_VARIED_TOTAL = 20   # 1 batch × 20
N_NEGATIVE     = 10

# Single varied batch — seed instructs the model to cover all registers itself
VARIED_SEED_FOCUSES = [
    "a mix of casual, formal, research, and comparison questions from Georgia "
    "residents, policy analysts, data journalists, and researchers — vary the "
    "geography (state, county, city, tract), register (colloquial to formal), "
    "and framing (counts, rates, trends, comparisons across Georgia geographies)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _format_tags(tags: list[dict]) -> str:
    if not tags:
        return "(none)"
    return ", ".join(t["tag"] for t in tags)


def _format_commonly_queried(s3: dict, s2: dict) -> str:
    """Format the commonly-queried variable list for prompt injection."""
    cq_ids: list[str] = s3.get("commonly_queried_variables", [])
    if not cq_ids:
        return "(none flagged)"

    inventory: list[dict] = s2.get("variable_inventory", [])
    id_to_label = {
        v["var_id"]: v.get("label_raw", "").replace("!!", " → ").strip(" →")
        for v in inventory
    }
    lines = []
    for var_id in cq_ids[:15]:  # cap at 15 for prompt length
        label = id_to_label.get(var_id, var_id)
        lines.append(f"  {var_id}: {label}")
    if len(cq_ids) > 15:
        lines.append(f"  ... ({len(cq_ids) - 15} more)")
    return "\n".join(lines)


def _load_context(
    year: str, dataset_path: str, table_id: str, paths: PipelinePaths
) -> Optional[dict]:
    s2 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, "stage2_structure"))
    s3 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S3))
    s4 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S4))
    s5 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S5))

    if not s2 or not s4:
        return None
    return {"s2": s2, "s3": s3 or {}, "s4": s4, "s5": s5 or {}}


def _is_light_tier(ctx: dict) -> bool:
    """Light tier = narrow table with no commonly-queried variables."""
    cq = ctx["s3"].get("commonly_queried_variables", [])
    return len(cq) == 0


def _deduplicate(questions: list[dict | str]) -> list[dict | str]:
    """Remove exact-text duplicates preserving order."""
    seen: set[str] = set()
    out = []
    for q in questions:
        text = q["text"] if isinstance(q, dict) else q
        normalized = text.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# Generation settings helpers
# ---------------------------------------------------------------------------

def _canonical_settings(inference_cfg: dict) -> dict:
    s6 = inference_cfg.get("stages", {}).get("stage6_questions", {})
    return s6.get("canonical", {
        "temperature": 0.3, "max_tokens": 2048,
        "thinking_mode": False, "top_p": 0.9,
    })


def _varied_settings(inference_cfg: dict) -> dict:
    s6 = inference_cfg.get("stages", {}).get("stage6_questions", {})
    return s6.get("varied", {
        "temperature": 0.85, "max_tokens": 4096,
        "thinking_mode": False, "top_p": 0.95,
    })


def _negative_settings(inference_cfg: dict) -> dict:
    s6 = inference_cfg.get("stages", {}).get("stage6_questions", {})
    return s6.get("negative", {
        "temperature": 0.7, "max_tokens": 2048,
        "thinking_mode": False, "top_p": 0.9,
    })


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

async def _generate_canonical(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> tuple[list[str], dict]:
    s2, s4, s5 = ctx["s2"], ctx["s4"], ctx["s5"]
    prompt = loader.render(
        "questions_canonical",
        title=s4.get("title") or "",
        description=s4.get("description") or "",
        universe_plain=s4.get("universe_plain") or "",
        tags=_format_tags(s5.get("tags", [])),
        measure_type=s2.get("measure_type") or "",
        temporal_stability=s4.get("temporal_stability") or "",
        sample_labels=build_sample_labels(s2.get("variable_inventory", [])),
        n_questions=N_CANONICAL,
    )
    resp = await client.generate(prompt, "questions_canonical",
                                 _canonical_settings(inference_cfg))
    questions = resp.content.get("questions", [])
    return questions, resp.provenance_block()


async def _generate_varied_batch(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    seed_focus: str,
    batch_index: int,
) -> tuple[list[dict], dict]:
    s2, s3, s4, s5 = ctx["s2"], ctx["s3"], ctx["s4"], ctx["s5"]
    prompt = loader.render(
        "questions_varied",
        title=s4.get("title") or "",
        description=s4.get("description") or "",
        universe_plain=s4.get("universe_plain") or "",
        tags=_format_tags(s5.get("tags", [])),
        measure_type=s2.get("measure_type") or "",
        commonly_queried=_format_commonly_queried(s3, s2),
        seed_focus=seed_focus,
        n_questions=N_VARIED_BATCH,
    )
    resp = await client.generate(prompt, "questions_varied",
                                 _varied_settings(inference_cfg))
    questions = resp.content.get("questions", [])
    for q in questions:
        q["_batch_index"] = batch_index
        q["_seed_focus_summary"] = seed_focus[:60]
    prov = resp.provenance_block()
    prov["batch_index"] = batch_index
    return questions, prov


async def _generate_negatives(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> tuple[list[dict], dict]:
    s2, s4, s5 = ctx["s2"], ctx["s4"], ctx["s5"]
    prompt = loader.render(
        "questions_negative",
        title=s4.get("title") or "",
        description=s4.get("description") or "",
        universe_plain=s4.get("universe_plain") or "",
        tags=_format_tags(s5.get("tags", [])),
        measure_type=s2.get("measure_type") or "",
        n_questions=N_NEGATIVE,
    )
    resp = await client.generate(prompt, "questions_negative",
                                 _negative_settings(inference_cfg))
    questions = resp.content.get("questions", [])
    return questions, resp.provenance_block()


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

async def generate_questions_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
) -> dict:
    """Generate all question sets for a single table."""

    # Check completion for all three sub-stages
    all_complete = all(
        state.is_complete(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=stage,
        )
        for stage in [STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE]
    )
    if skip_if_complete and all_complete:
        return {"status": "skipped", "table_id": table_id}

    ctx = _load_context(year, dataset_path, table_id, paths)
    if not ctx:
        msg = f"Missing stage2/stage4 for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        for stage in [STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE]:
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=stage,
                status=Status.FAILED_NO_DATA, error_message=msg,
            )
        return {"status": "failed", "reason": "missing_prerequisites"}

    analysis_type = ctx["s2"].get("analysis_type", "grouped")
    if analysis_type == "flat":
        for stage in [STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE]:
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=stage,
                status=Status.SUCCESS,
                metadata={"skipped": True, "reason": "flat_dataset"},
            )
        return {"status": "skipped", "reason": "flat_dataset"}

    light_tier = _is_light_tier(ctx)
    collected_at = datetime.now(timezone.utc).isoformat()
    table_label = f"{year}/{dataset_path}/{table_id}"

    try:
        need_canonical = not (skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_CANONICAL,
        ))
        need_varied = not light_tier and not (skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_VARIED,
        ))
        need_negative = not (skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NEGATIVE,
        ))

        # ── Run all needed calls concurrently ──────────────────────────
        # Canonical, varied, and negative are independent — no ordering
        # dependency between them. Running concurrently cuts wall time
        # from 3× latency to 1× max(latency).
        import asyncio as _asyncio

        tasks = {}
        if need_canonical:
            tasks["canonical"] = _asyncio.ensure_future(
                _generate_canonical(ctx, client, loader, inference_cfg)
            )
        if need_varied:
            tasks["varied"] = _asyncio.ensure_future(
                _generate_varied_batch(
                    ctx, client, loader, inference_cfg,
                    VARIED_SEED_FOCUSES[0], 0,
                )
            )
        if need_negative:
            tasks["negative"] = _asyncio.ensure_future(
                _generate_negatives(ctx, client, loader, inference_cfg)
            )

        results = {}
        if tasks:
            done = await _asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), done):
                if isinstance(result, Exception):
                    raise result
                results[key] = result

        # ── Write canonical ─────────────────────────────────────────────
        if need_canonical:
            canonical_qs, canonical_prov = results["canonical"]
            canonical_qs = _deduplicate(canonical_qs)
            _write_json_atomic(
                paths.processed_table_stage_file(
                    year, dataset_path, table_id, STAGE_CANONICAL),
                {
                    "schema_version": "1.0.0",
                    "stage": STAGE_CANONICAL,
                    "collected_at": collected_at,
                    "year": year,
                    "dataset_path": dataset_path,
                    "table_id": table_id,
                    "question_count": len(canonical_qs),
                    "questions": canonical_qs,
                    "_provenance": canonical_prov,
                }
            )
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=STAGE_CANONICAL,
                status=Status.SUCCESS,
                metadata={"question_count": len(canonical_qs)},
            )
            print(f"  [canonical]  {table_label} — {len(canonical_qs)} questions")

        # ── Write varied ────────────────────────────────────────────────
        if need_varied:
            varied_qs, varied_prov = results["varied"]
            varied_qs = _deduplicate(varied_qs)
            _write_json_atomic(
                paths.processed_table_stage_file(
                    year, dataset_path, table_id, STAGE_VARIED),
                {
                    "schema_version": "1.0.0",
                    "stage": STAGE_VARIED,
                    "collected_at": collected_at,
                    "year": year,
                    "dataset_path": dataset_path,
                    "table_id": table_id,
                    "question_count": len(varied_qs),
                    "batch_count": 1,
                    "questions": varied_qs,
                    "_provenance_batches": [varied_prov],
                }
            )
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=STAGE_VARIED,
                status=Status.SUCCESS,
                metadata={"question_count": len(varied_qs), "tier": "full"},
            )
            print(f"  [varied]     {table_label} — {len(varied_qs)} questions")

        elif light_tier and not (skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_VARIED,
        )):
            _write_json_atomic(
                paths.processed_table_stage_file(
                    year, dataset_path, table_id, STAGE_VARIED),
                {
                    "schema_version": "1.0.0",
                    "stage": STAGE_VARIED,
                    "collected_at": collected_at,
                    "year": year,
                    "dataset_path": dataset_path,
                    "table_id": table_id,
                    "question_count": 0,
                    "tier": "light",
                    "questions": [],
                }
            )
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=STAGE_VARIED,
                status=Status.SUCCESS,
                metadata={"skipped": True, "reason": "light_tier", "question_count": 0},
            )
            print(f"  [varied]     {table_label} — skipped (light tier)")

        # ── Write negative ──────────────────────────────────────────────
        if need_negative:
            negative_qs, negative_prov = results["negative"]
            negative_qs = _deduplicate(negative_qs)
            _write_json_atomic(
                paths.processed_table_stage_file(
                    year, dataset_path, table_id, STAGE_NEGATIVE),
                {
                    "schema_version": "1.0.0",
                    "stage": STAGE_NEGATIVE,
                    "collected_at": collected_at,
                    "year": year,
                    "dataset_path": dataset_path,
                    "table_id": table_id,
                    "question_count": len(negative_qs),
                    "questions": negative_qs,
                    "_provenance": negative_prov,
                }
            )
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=STAGE_NEGATIVE,
                status=Status.SUCCESS,
                metadata={"question_count": len(negative_qs)},
            )
            print(f"  [negative]   {table_label} — {len(negative_qs)} questions")

        tier = "light" if light_tier else "full"
        logger.debug("✓ %s (%s tier)", table_label, tier)
        return {"status": "success", "table_id": table_id, "tier": tier}

    except LLMCallError as e:
        logger.error("LLM call failed for %s: %s", table_label, e)
        for stage in [STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE]:
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=stage,
                status=Status.FAILED_OTHER, error_message=str(e),
            )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage="stage6_questions", error_type="LLMCallError",
            error_message=str(e),
        )
        return {"status": "failed", "reason": str(e)}

    except Exception as e:
        logger.exception("Unexpected error generating questions for %s", table_label)
        for stage in [STAGE_CANONICAL, STAGE_VARIED, STAGE_NEGATIVE]:
            state.set_status(
                year=year, dataset_path=dataset_path, entity_type="table",
                entity_id=table_id, stage=stage,
                status=Status.FAILED_OTHER, error_message=str(e),
            )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage="stage6_questions", error_type=type(e).__name__,
            error_message=str(e), traceback=traceback.format_exc(),
        )
        return {"status": "failed", "reason": str(e)}
