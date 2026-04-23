"""Stage 7: Variable-Level Enrichment.

For every estimate variable in every table, produces:
  - label_natural: concise natural-language phrase
  - description: 1-2 sentence plain-English description
  - column_name: snake_case CSV column identifier

For question-earning variables (per deterministic rules below), also produces:
  - 5 year-agnostic questions that would naturally retrieve the variable

CROSS-YEAR DEDUP:
Variables are keyed by (dataset_path, table_id, var_id, hash(label_raw))
and cached in a SQLite database. The same variable across years shares
a cache entry, so LLM generation happens once per unique fingerprint
regardless of how many years contain it. Bumping DESCRIPTION_PROMPT_VERSION
or QUESTION_PROMPT_VERSION invalidates cache entries.

QUESTION-EARNING RULES (a variable earns 5 questions if ANY are true):
  1. is_total == True                  — headline/total variables
  2. depth <= 2 AND is estimate        — natural user-intent level
  3. var_id in commonly_queried_variables  (from Stage 3)
  4. kind in {median, aggregate, ratio}  — summary stats at any depth

OUTPUTS PER TABLE (per year):
  stage7_variables.json — one entry per estimate variable with
    label_natural, description, column_name, associated MOE/annotation,
    and questions (for earning variables only)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.enrichment.llm_client import VLLMClient, LLMCallError
from scripts.enrichment.prompts import PromptLoader
from scripts.enrichment.variable_cache import (
    VariableCache, compute_fingerprint,
    DESCRIPTION_PROMPT_VERSION, QUESTION_PROMPT_VERSION,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage7_variables"
DEPENDS_ON_S2 = "stage2_structure"
DEPENDS_ON_S3 = "stage3_relationships"
DEPENDS_ON_S4 = "stage4_enriched"


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


def _label_path_from_raw(label_raw: str) -> list[str]:
    """Split a raw label like 'Estimate!!Total:!!Male:!!Under 5 years'
    into ['Total', 'Male', 'Under 5 years']. Strips 'Estimate' prefix
    and trailing colons."""
    parts = [p.strip().rstrip(":") for p in label_raw.split("!!")]
    # Drop 'Estimate' or 'Annotation' prefix
    if parts and parts[0].lower() in ("estimate", "annotation", "margin of error"):
        parts = parts[1:]
    return [p for p in parts if p]


def _get_annotation_var_id(var_id: str) -> Optional[str]:
    """Given an estimate var_id like B17001_003E, return annotation pair
    B17001_003EA if it exists. Returns None if the var_id does not end
    with E (decennial _N variables have no annotation pair)."""
    if var_id.endswith("E"):
        return var_id + "A"
    return None


# ---------------------------------------------------------------------------
# Question-earning rules
# ---------------------------------------------------------------------------

SUMMARY_STAT_KINDS = {"median", "aggregate", "ratio"}


def variable_earns_questions(
    var_entry: dict,
    commonly_queried: set[str],
) -> bool:
    """Determine if a variable earns its own set of generated questions.

    Earns 5 questions if ANY condition is true:
      1. is_total flag set
      2. depth <= 2 (natural user-intent level)
      3. flagged in Stage 3 commonly_queried_variables
      4. kind is median/aggregate/ratio (summary stats at any depth)
    """
    if var_entry.get("is_total"):
        return True
    if var_entry.get("depth", 99) <= 2:
        return True
    if var_entry["var_id"] in commonly_queried:
        return True
    if var_entry.get("kind") in SUMMARY_STAT_KINDS:
        return True
    return False


# ---------------------------------------------------------------------------
# Column name uniqueness
# ---------------------------------------------------------------------------

_COL_SAFE_RE = re.compile(r"[^a-z0-9_]")
_COL_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _sanitize_column_name(raw: str, max_len: int = 40) -> str:
    """Apply strict snake_case rules even if the LLM was sloppy."""
    if not raw:
        return "unnamed"
    cleaned = raw.lower().strip()
    cleaned = _COL_SAFE_RE.sub("_", cleaned)
    cleaned = _COL_MULTI_UNDERSCORE_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        return "unnamed"
    if cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned[:max_len].rstrip("_") or "unnamed"


def resolve_column_name_collisions(
    var_entries: list[dict],
    variable_inventory: list[dict],
) -> None:
    """In-place deduplication of column_name across a table's variables.

    Two-pass approach:
      1. Identify collisions (same column_name on different var_ids)
      2. For colliding entries, prepend the nearest distinguishing label
         segment. If that still collides, append the numeric var_id suffix.
    """
    inv_by_id = {v["var_id"]: v for v in variable_inventory}

    # Sanitize first
    for e in var_entries:
        e["column_name"] = _sanitize_column_name(e.get("column_name", ""))

    # Group by column_name to find collisions
    by_col: dict[str, list[dict]] = defaultdict(list)
    for e in var_entries:
        by_col[e["column_name"]].append(e)

    for col_name, group in by_col.items():
        if len(group) <= 1:
            continue

        # Attempt 1 — prepend nearest distinguishing label segment
        resolved = {}
        for entry in group:
            inv = inv_by_id.get(entry["var_id"], {})
            segments = inv.get("segments", [])
            # Find the segment that differs from siblings
            # Use var_id-based distinction if segments not available
            if segments:
                # Prepend the shallowest segment as a prefix
                prefix = _sanitize_column_name(segments[-1] if len(segments) >= 2
                                               else segments[0])
                if prefix and not entry["column_name"].startswith(prefix):
                    new_name = f"{prefix}_{entry['column_name']}"
                    entry["column_name"] = _sanitize_column_name(new_name)
            resolved[entry["var_id"]] = entry["column_name"]

        # Check if attempt 1 resolved everything
        new_cols = [e["column_name"] for e in group]
        if len(set(new_cols)) == len(group):
            continue

        # Attempt 2 — append numeric suffix from var_id
        for entry in group:
            # Extract trailing digits from var_id (e.g., B17001_003E -> 003)
            m = re.search(r"_(\d+)[A-Z]*$", entry["var_id"])
            suffix = m.group(1) if m else entry["var_id"][-4:]
            new_name = f"{entry['column_name']}_{suffix}"
            entry["column_name"] = _sanitize_column_name(new_name, max_len=48)


# ---------------------------------------------------------------------------
# Prompt input builders
# ---------------------------------------------------------------------------

def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _build_description_variable_list(chunk: list[dict]) -> str:
    """Format a chunk of variable inventory entries for prompt injection."""
    lines = []
    for v in chunk:
        path = _label_path_from_raw(v.get("label_raw", ""))
        path_str = " > ".join(path) if path else v.get("label_raw", "")
        lines.append(f"  {v['var_id']}: {path_str}")
    return "\n".join(lines)


def _build_question_variable_list(
    chunk: list[dict],
    descriptions_by_var: dict[str, dict],
) -> str:
    """Format question-earning variables with their natural labels."""
    lines = []
    for v in chunk:
        var_id = v["var_id"]
        desc = descriptions_by_var.get(var_id, {})
        label = desc.get("label_natural", v.get("label_raw", ""))
        lines.append(f"  {var_id}: {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core generation — batched
# ---------------------------------------------------------------------------

def _description_settings(inference_cfg: dict) -> dict:
    s7 = inference_cfg.get("stages", {}).get("stage7_variables", {})
    return s7.get("description", {
        "temperature": 0.2, "max_tokens": 4096,
        "thinking_mode": False, "top_p": 0.9,
    })


def _question_settings(inference_cfg: dict) -> dict:
    s7 = inference_cfg.get("stages", {}).get("stage7_variables", {})
    return s7.get("questions", {
        "temperature": 0.8, "max_tokens": 4096,
        "thinking_mode": False, "top_p": 0.95,
    })


async def _generate_descriptions_chunk(
    chunk: list[dict],
    table_ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> list[dict]:
    """Generate label_natural + description + column_name for a chunk of vars."""
    prompt = loader.render(
        "variable_description_batch",
        table_id=table_ctx["table_id"],
        table_title=table_ctx.get("title", ""),
        table_description=table_ctx.get("description", ""),
        universe_plain=table_ctx.get("universe_plain", ""),
        measure_type=table_ctx.get("measure_type", ""),
        subject_area=table_ctx.get("subject_area", ""),
        variable_list=_build_description_variable_list(chunk),
    )
    resp = await client.generate(
        prompt, "variable_description_batch", _description_settings(inference_cfg)
    )
    out = resp.content.get("variables", [])
    # Defensively fill missing entries if the model returned fewer than expected
    out_by_id = {e["var_id"]: e for e in out if "var_id" in e}
    result = []
    for v in chunk:
        vid = v["var_id"]
        entry = out_by_id.get(vid)
        if entry:
            result.append(entry)
        else:
            # Fallback: deterministic description from label path
            path = _label_path_from_raw(v.get("label_raw", ""))
            label_natural = " ".join(path) if path else vid
            result.append({
                "var_id": vid,
                "label_natural": label_natural[:80],
                "description": f"Census variable {vid}: {label_natural}",
                "column_name": _sanitize_column_name(label_natural),
            })
    return result


async def _generate_questions_chunk(
    chunk: list[dict],
    descriptions_by_var: dict[str, dict],
    table_ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> list[dict]:
    prompt = loader.render(
        "variable_questions_batch",
        table_id=table_ctx["table_id"],
        table_title=table_ctx.get("title", ""),
        table_description=table_ctx.get("description", ""),
        universe_plain=table_ctx.get("universe_plain", ""),
        variable_list=_build_question_variable_list(chunk, descriptions_by_var),
    )
    resp = await client.generate(
        prompt, "variable_questions_batch", _question_settings(inference_cfg)
    )
    out = resp.content.get("variables", [])
    out_by_id = {e["var_id"]: e for e in out if "var_id" in e}
    result = []
    for v in chunk:
        vid = v["var_id"]
        entry = out_by_id.get(vid)
        if entry and entry.get("questions"):
            result.append(entry)
        else:
            # Fallback — use the label as a single retrieval anchor
            desc = descriptions_by_var.get(vid, {})
            label = desc.get("label_natural", vid)
            result.append({
                "var_id": vid,
                "questions": [
                    f"What is {label.lower()} in Georgia?",
                    f"How many {label.lower()} in Fulton County?",
                    f"{label} by county in Georgia",
                ],
            })
    return result


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

def _load_context(
    year: str, dataset_path: str, table_id: str, paths: PipelinePaths
) -> Optional[dict]:
    s2 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S2))
    s3 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S3))
    s4 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S4))
    if not s2 or not s4:
        return None
    return {"s2": s2, "s3": s3 or {}, "s4": s4}


def _build_table_ctx(ctx: dict, table_id: str) -> dict:
    s2, s4 = ctx["s2"], ctx["s4"]
    return {
        "table_id": table_id,
        "title": s4.get("title", ""),
        "description": s4.get("description", ""),
        "universe_plain": s4.get("universe_plain", ""),
        "measure_type": s2.get("measure_type", ""),
        "subject_area": s2.get("subject_area", ""),
    }


async def process_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    client: VLLMClient,
    loader: PromptLoader,
    cache: VariableCache,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
) -> dict:
    """Run Stage 7 for one table: generate descriptions + questions for every
    estimate variable, using cross-year cache to dedup."""

    if skip_if_complete and state.is_complete(
        year=year, dataset_path=dataset_path, entity_type="table",
        entity_id=table_id, stage=STAGE_NAME,
    ):
        return {"status": "skipped", "table_id": table_id}

    ctx = _load_context(year, dataset_path, table_id, paths)
    if not ctx:
        msg = f"Missing stage2/stage4 for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_NO_DATA, error_message=msg,
        )
        return {"status": "failed", "reason": "missing_prerequisites"}

    s2 = ctx["s2"]
    analysis_type = s2.get("analysis_type", "grouped")

    # Flat tables don't have estimate variables in the ACS sense — skip
    if analysis_type == "flat":
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.SUCCESS,
            metadata={"skipped": True, "reason": "flat_dataset"},
        )
        return {"status": "skipped", "reason": "flat_dataset"}

    inventory = s2.get("variable_inventory", [])
    if not inventory:
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.SUCCESS,
            metadata={"skipped": True, "reason": "empty_inventory"},
        )
        return {"status": "skipped", "reason": "empty_inventory"}

    table_ctx = _build_table_ctx(ctx, table_id)
    commonly_queried = set(ctx["s3"].get("commonly_queried_variables", []))
    table_label = f"{year}/{dataset_path}/{table_id}"

    try:
        # ── Step 1: Descriptions — check cache, generate misses ────────
        desc_fingerprints = {
            v["var_id"]: compute_fingerprint(
                DESCRIPTION_PROMPT_VERSION, dataset_path, table_id,
                v["var_id"], v.get("label_raw", ""),
            )
            for v in inventory
        }
        desc_cache_hits: dict[str, dict] = {}
        desc_to_generate: list[dict] = []
        for v in inventory:
            fp = desc_fingerprints[v["var_id"]]
            hit = cache.get_description(fp)
            if hit:
                desc_cache_hits[v["var_id"]] = hit
            else:
                desc_to_generate.append(v)

        n_desc_hits = len(desc_cache_hits)
        n_desc_gen = len(desc_to_generate)

        descriptions_by_var: dict[str, dict] = dict(desc_cache_hits)

        if desc_to_generate:
            chunk_size = inference_cfg.get("batching", {}).get(
                "stage7_description_chunk_size", 50
            )
            chunk_tasks = []
            for chunk in _chunks(desc_to_generate, chunk_size):
                chunk_tasks.append(
                    _generate_descriptions_chunk(
                        chunk, table_ctx, client, loader, inference_cfg,
                    )
                )
            all_results = await asyncio.gather(*chunk_tasks)

            to_cache = []
            for chunk_results in all_results:
                for entry in chunk_results:
                    vid = entry["var_id"]
                    descriptions_by_var[vid] = {
                        "label_natural": entry["label_natural"],
                        "description": entry["description"],
                        "column_name": entry["column_name"],
                        "model_id": client.config.model_id,
                    }
                    to_cache.append({
                        "fingerprint": desc_fingerprints[vid],
                        "dataset_path": dataset_path,
                        "table_id": table_id,
                        "var_id": vid,
                        "label_natural": entry["label_natural"],
                        "description": entry["description"],
                        "column_name": entry["column_name"],
                        "model_id": client.config.model_id,
                    })
            cache.put_descriptions_batch(to_cache)

        # ── Step 2: Resolve column name collisions within the table ────
        var_entries_for_collision = [
            {"var_id": vid, "column_name": data["column_name"]}
            for vid, data in descriptions_by_var.items()
        ]
        resolve_column_name_collisions(var_entries_for_collision, inventory)
        for entry in var_entries_for_collision:
            descriptions_by_var[entry["var_id"]]["column_name"] = entry["column_name"]

        # ── Step 3: Questions — for earning variables only, check cache ─
        earning_vars = [
            v for v in inventory
            if variable_earns_questions(v, commonly_queried)
        ]
        quest_fingerprints = {
            v["var_id"]: compute_fingerprint(
                QUESTION_PROMPT_VERSION, dataset_path, table_id,
                v["var_id"], v.get("label_raw", ""),
            )
            for v in earning_vars
        }
        questions_by_var: dict[str, list[str]] = {}
        quest_to_generate: list[dict] = []
        for v in earning_vars:
            fp = quest_fingerprints[v["var_id"]]
            hit = cache.get_questions(fp)
            if hit:
                questions_by_var[v["var_id"]] = hit
            else:
                quest_to_generate.append(v)

        n_quest_hits = len(questions_by_var)
        n_quest_gen = len(quest_to_generate)

        if quest_to_generate:
            chunk_size = inference_cfg.get("batching", {}).get(
                "stage7_question_chunk_size", 20
            )
            chunk_tasks = []
            for chunk in _chunks(quest_to_generate, chunk_size):
                chunk_tasks.append(
                    _generate_questions_chunk(
                        chunk, descriptions_by_var, table_ctx,
                        client, loader, inference_cfg,
                    )
                )
            all_results = await asyncio.gather(*chunk_tasks)

            to_cache = []
            for chunk_results in all_results:
                for entry in chunk_results:
                    vid = entry["var_id"]
                    questions = entry.get("questions", [])
                    questions_by_var[vid] = questions
                    to_cache.append({
                        "fingerprint": quest_fingerprints[vid],
                        "dataset_path": dataset_path,
                        "table_id": table_id,
                        "var_id": vid,
                        "questions": questions,
                        "model_id": client.config.model_id,
                    })
            cache.put_questions_batch(to_cache)

        # ── Step 4: Assemble output document ───────────────────────────
        variables_output = []
        for v in inventory:
            vid = v["var_id"]
            desc = descriptions_by_var.get(vid, {})
            moe_id = v.get("moe_var_id")
            annotation_id = _get_annotation_var_id(vid)
            associated = {}
            if moe_id:
                associated["moe"] = {
                    "var_id": moe_id,
                    "column_name": f"{desc.get('column_name', vid)}_moe",
                }
            if annotation_id:
                associated["annotation"] = {"var_id": annotation_id}

            var_out = {
                "var_id": vid,
                "label_raw": v.get("label_raw", ""),
                "label_natural": desc.get("label_natural", ""),
                "description": desc.get("description", ""),
                "column_name": desc.get("column_name", ""),
                "depth": v.get("depth", 0),
                "is_total": v.get("is_total", False),
                "kind": v.get("kind", "estimate"),
                "is_commonly_queried": vid in commonly_queried,
                "earns_questions": variable_earns_questions(v, commonly_queried),
                "questions": questions_by_var.get(vid, []),
            }
            if associated:
                var_out["associated"] = associated
            variables_output.append(var_out)

        # ── Step 5: Coverage check ─────────────────────────────────────
        expected_ids = {v["var_id"] for v in inventory}
        covered_ids = {v["var_id"] for v in variables_output}
        orphans = expected_ids - covered_ids
        if orphans:
            raise RuntimeError(
                f"Coverage check failed: {len(orphans)} variables not in "
                f"output: {sorted(orphans)[:5]}"
            )

        # ── Step 6: Write output ───────────────────────────────────────
        output = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "year": year,
            "dataset_path": dataset_path,
            "table_id": table_id,
            "variable_count": len(variables_output),
            "earning_variable_count": sum(
                1 for v in variables_output if v["earns_questions"]
            ),
            "variables": variables_output,
            "_cache_stats": {
                "description_cache_hits": n_desc_hits,
                "description_cache_misses": n_desc_gen,
                "question_cache_hits": n_quest_hits,
                "question_cache_misses": n_quest_gen,
            },
        }
        out_path = paths.processed_table_stage_file(
            year, dataset_path, table_id, STAGE_NAME
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(out_path, output)

        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.SUCCESS,
            metadata={
                "variable_count": len(variables_output),
                "earning_variable_count": output["earning_variable_count"],
                "desc_cache_hit_rate": (
                    n_desc_hits / len(inventory) if inventory else 0
                ),
            },
        )
        print(
            f"  [stage7]  {table_label} — {len(variables_output)} vars "
            f"({n_desc_hits}h/{n_desc_gen}m desc, "
            f"{n_quest_hits}h/{n_quest_gen}m quest)"
        )
        return {
            "status": "success", "table_id": table_id,
            "variable_count": len(variables_output),
            "desc_hits": n_desc_hits, "desc_misses": n_desc_gen,
            "quest_hits": n_quest_hits, "quest_misses": n_quest_gen,
        }

    except LLMCallError as e:
        logger.error("LLM call failed for %s: %s", table_label, e)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_OTHER, error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type="LLMCallError", error_message=str(e),
        )
        return {"status": "failed", "reason": str(e)}

    except Exception as e:
        logger.exception("Unexpected error for %s", table_label)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_OTHER, error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type=type(e).__name__,
            error_message=str(e), traceback=traceback.format_exc(),
        )
        return {"status": "failed", "reason": str(e)}
