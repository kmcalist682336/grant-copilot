"""Stage 4: LLM-Based Enrichment.

Reads stage1_metadata.json, stage2_structure.json, and stage3_relationships.json
per table and uses Gemma 4 26B A4B via vLLM to generate:

  For grouped tables (ACS, Decennial, PEP):
    - Plain-English description of what the table measures
    - use_when / avoid_when guidance
    - Natural-language universe rewrite
    - Domain glossary (key Census terms defined)
    - Data quality caveats
    - Temporal stability classification

  For flat tables (pdb, geoinfo):
    - Plain-English description
    - Dataset-level caveat

All outputs are written to stage4_enriched.json alongside the earlier
stage files. Every LLM-generated field carries a _provenance block
recording the model, prompt version, and generation settings used.

vLLM must be running before this stage executes:
    python -m vllm.entrypoints.openai.api_server \\
        --model <model_path> --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.enrichment.llm_client import VLLMClient, VLLMClientConfig, LLMCallError
from scripts.enrichment.prompts import PromptLoader, build_sample_labels, build_flat_sample_labels
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage4_enriched"
DEPENDS_ON_S1 = "stage1_metadata"
DEPENDS_ON_S2 = "stage2_structure"
DEPENDS_ON_S3 = "stage3_relationships"


# ---------------------------------------------------------------------------
# Loaders
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


def _load_table_context(
    year: str, dataset_path: str, table_id: str, paths: PipelinePaths
) -> Optional[dict]:
    """Load all prior stage outputs for a table into a single context dict."""
    s1 = _load_json(paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON_S1))
    s2 = _load_json(paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON_S2))
    s3 = _load_json(paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON_S3))

    if not s1 or not s2:
        return None  # s3 is optional — Stage 3 may not have run for flat/microdata

    return {"s1": s1, "s2": s2, "s3": s3 or {}}


# ---------------------------------------------------------------------------
# Generation settings helpers
# ---------------------------------------------------------------------------

def _generative_settings(inference_cfg: dict) -> dict:
    s4 = inference_cfg.get("stages", {}).get("stage4_enrichment", {})
    return s4.get("generative", {
        "temperature": 0.3, "max_tokens": 1024,
        "thinking_mode": False, "top_p": 0.9,
    })


def _classification_settings(inference_cfg: dict) -> dict:
    s4 = inference_cfg.get("stages", {}).get("stage4_enrichment", {})
    return s4.get("classification", {
        "temperature": 0.1, "max_tokens": 256,
        "thinking_mode": True, "top_p": 0.9,
    })


# ---------------------------------------------------------------------------
# Grouped table enrichment
# ---------------------------------------------------------------------------

async def enrich_grouped_table(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> dict:
    """Run the full enrichment suite for a standard grouped table."""
    s1, s2 = ctx["s1"], ctx["s2"]

    title = s1.get("title") or ""
    universe = s1.get("universe") or ""
    dataset_path = s1.get("dataset_path", "")
    year = s1.get("year", "")
    measure_type = s2.get("measure_type", "")
    subject_area = s2.get("subject_area") or ""
    analysis_type = s2.get("analysis_type", "grouped")
    inventory = s2.get("variable_inventory", [])
    sample_labels = build_sample_labels(inventory)
    gen = _generative_settings(inference_cfg)
    cls = _classification_settings(inference_cfg)

    # --- Prompt 1: Description ---
    desc_prompt = loader.render(
        "table_description",
        title=title, universe=universe, dataset_path=dataset_path,
        year=year, dataset_structure=analysis_type, measure_type=measure_type,
        variable_count=len(inventory), sample_labels=sample_labels,
    )

    # --- Prompt 2: Use/Avoid ---
    # We need description first — run prompt 1 before 2
    desc_resp = await client.generate(desc_prompt, "table_description", gen)
    description = desc_resp.content.get("description", "")

    use_avoid_prompt = loader.render(
        "table_use_avoid",
        title=title, description=description, universe=universe,
        dataset_path=dataset_path, year=year, measure_type=measure_type,
        subject_area=subject_area, sample_labels=sample_labels,
    )

    # --- Prompt 3: Universe rewrite ---
    universe_prompt = loader.render(
        "universe_rewrite",
        title=title, dataset_path=dataset_path, year=year, universe=universe,
    )

    # --- Prompt 4: Temporal stability ---
    stability_prompt = loader.render(
        "temporal_stability",
        title=title, description=description, subject_area=subject_area,
        measure_type=measure_type,
    )

    # Prompts 2, 3, 4 can run concurrently
    batch_results = await client.generate_batch([
        (use_avoid_prompt, "table_use_avoid", gen),
        (universe_prompt, "universe_rewrite", gen),
        (stability_prompt, "temporal_stability", cls),
    ])

    use_avoid_resp, universe_resp, stability_resp = batch_results

    # Handle any failures in the batch
    for label, resp in [
        ("use_avoid", use_avoid_resp),
        ("universe_rewrite", universe_resp),
        ("temporal_stability", stability_resp),
    ]:
        if isinstance(resp, Exception):
            logger.warning("Prompt '%s' failed: %s", label, resp)

    universe_plain = (
        universe_resp.content.get("universe_plain", universe)
        if not isinstance(universe_resp, Exception) else universe
    )

    # --- Prompt 5: Glossary and caveats (needs universe_plain) ---
    glossary_prompt = loader.render(
        "glossary_and_caveats",
        title=title, universe_plain=universe_plain, description=description,
        dataset_path=dataset_path, year=year, measure_type=measure_type,
        sample_labels=sample_labels,
    )
    glossary_resp = await client.generate(glossary_prompt, "glossary_and_caveats", gen)

    # --- Assemble output ---
    def _prov(resp, fallback_name: str) -> dict:
        if isinstance(resp, Exception):
            return {"error": str(resp), "prompt_name": fallback_name}
        return resp.provenance_block()

    return {
        "description": description,
        "_provenance_description": desc_resp.provenance_block(),

        "use_when": (
            use_avoid_resp.content.get("use_when", "")
            if not isinstance(use_avoid_resp, Exception) else ""
        ),
        "avoid_when": (
            use_avoid_resp.content.get("avoid_when", "")
            if not isinstance(use_avoid_resp, Exception) else ""
        ),
        "_provenance_use_avoid": _prov(use_avoid_resp, "table_use_avoid"),

        "universe_plain": universe_plain,
        "_provenance_universe": _prov(universe_resp, "universe_rewrite"),

        "temporal_stability": (
            stability_resp.content.get("temporal_stability", "unknown")
            if not isinstance(stability_resp, Exception) else "unknown"
        ),
        "stability_rationale": (
            stability_resp.content.get("stability_rationale", "")
            if not isinstance(stability_resp, Exception) else ""
        ),
        "_provenance_stability": _prov(stability_resp, "temporal_stability"),

        "glossary": (
            glossary_resp.content.get("glossary", [])
            if not isinstance(glossary_resp, Exception) else []
        ),
        "caveats": (
            glossary_resp.content.get("caveats", [])
            if not isinstance(glossary_resp, Exception) else []
        ),
        "_provenance_glossary_caveats": _prov(glossary_resp, "glossary_and_caveats"),
    }


# ---------------------------------------------------------------------------
# Flat table enrichment
# ---------------------------------------------------------------------------

async def enrich_flat_table(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
) -> dict:
    """Run simplified enrichment for flat-structure datasets."""
    s1, s2 = ctx["s1"], ctx["s2"]
    title = s1.get("title") or ""
    dataset_path = s1.get("dataset_path", "")
    year = s1.get("year", "")
    dataset_structure = s2.get("analysis_type", "flat")
    variables = s1.get("variables", {})
    variable_count = len(variables)
    sample_labels = build_flat_sample_labels(variables)
    gen = _generative_settings(inference_cfg)

    prompt = loader.render(
        "flat_dataset_description",
        title=title, dataset_path=dataset_path, year=year,
        dataset_structure=dataset_structure, variable_count=variable_count,
        sample_labels=sample_labels,
    )
    resp = await client.generate(prompt, "flat_dataset_description", gen)

    return {
        "description": resp.content.get("description", ""),
        "caveats": [resp.content.get("caveat", "")] if resp.content.get("caveat") else [],
        "use_when": "",
        "avoid_when": "",
        "universe_plain": "",
        "temporal_stability": "stable",
        "stability_rationale": "Flat planning/reference dataset — updated on census cycle.",
        "glossary": [],
        "_provenance_description": resp.provenance_block(),
    }


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

async def enrich_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
) -> dict:
    """Run Stage 4 enrichment for a single table."""
    ctx = _load_table_context(year, dataset_path, table_id, paths)
    if not ctx:
        msg = f"Missing stage1/stage2 for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_NO_DATA, error_message=msg,
        )
        return {"status": "failed", "reason": "missing_prerequisites"}

    analysis_type = ctx["s2"].get("analysis_type", "grouped")
    collected_at = datetime.now(timezone.utc).isoformat()

    try:
        if analysis_type == "flat":
            enrichment = await enrich_flat_table(ctx, client, loader, inference_cfg)
        else:
            enrichment = await enrich_grouped_table(ctx, client, loader, inference_cfg)

        output = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "collected_at": collected_at,
            "source": "census",
            "year": year,
            "dataset_path": dataset_path,
            "table_id": table_id,
            "title": ctx["s1"].get("title"),
            "universe": ctx["s1"].get("universe"),
            "analysis_type": analysis_type,
            **enrichment,
        }

        out_path = paths.processed_table_stage_file(
            year, dataset_path, table_id, STAGE_NAME
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(out_path, output)

        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME, status=Status.SUCCESS,
            metadata={"analysis_type": analysis_type},
        )
        return {"status": "success", "table_id": table_id}

    except LLMCallError as e:
        logger.error("LLM call failed for %s/%s/%s: %s", year, dataset_path, table_id, e)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME, status=Status.FAILED_OTHER,
            error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type="LLMCallError", error_message=str(e),
        )
        return {"status": "failed", "reason": str(e)}

    except Exception as e:
        logger.exception("Unexpected error enriching %s/%s/%s", year, dataset_path, table_id)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME, status=Status.FAILED_OTHER,
            error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type=type(e).__name__, error_message=str(e),
            traceback=traceback.format_exc(),
        )
        return {"status": "failed", "reason": str(e)}


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

async def enrich_dataset(
    year: str,
    dataset_path: str,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
    max_concurrent: int = 8,
) -> dict:
    """Enrich all tables in a dataset concurrently."""
    dataset_doc_path = paths.processed_dataset_document(year, dataset_path)
    dataset_doc = _load_json(dataset_doc_path)
    if not dataset_doc:
        logger.warning("No dataset_document for %s/%s — skipping", year, dataset_path)
        return {"status": "failed", "reason": "missing_dataset_document"}

    table_ids = dataset_doc.get("table_ids", [])
    if not table_ids:
        return {"status": "success", "tables_ok": 0, "tables_skipped": 0}

    # Filter already-complete tables
    pending = []
    skipped = 0
    for table_id in table_ids:
        if skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path,
            entity_type="table", entity_id=table_id, stage=STAGE_NAME,
        ):
            skipped += 1
        else:
            pending.append(table_id)

    if not pending:
        logger.info("%s/%s: all %d tables already complete", year, dataset_path, skipped)
        return {"status": "success", "tables_ok": 0, "tables_skipped": skipped}

    logger.info(
        "%s/%s: enriching %d tables (%d skipped), concurrency=%d",
        year, dataset_path, len(pending), skipped, max_concurrent,
    )

    # Process tables in concurrent batches
    ok = failed = 0
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(table_id: str) -> dict:
        async with semaphore:
            return await enrich_one_table(
                year, dataset_path, table_id,
                client, loader, inference_cfg, paths, state,
            )

    tasks = [_bounded(tid) for tid in pending]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
            logger.exception("Unhandled exception in table enrichment: %s", result)
        elif result.get("status") == "success":
            ok += 1
        else:
            failed += 1

    logger.info(
        "%s/%s Stage 4: %d ok, %d failed, %d skipped",
        year, dataset_path, ok, failed, skipped,
    )
    return {
        "status": "success" if failed == 0 else "partial",
        "tables_ok": ok,
        "tables_failed": failed,
        "tables_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

async def run_stage4(
    paths: PipelinePaths,
    state: StateTracker,
    client: VLLMClient,
    loader: PromptLoader,
    inference_cfg: dict,
    *,
    skip_if_complete: bool = True,
    max_concurrent_tables: int = 8,
    dataset_filter: Optional[str] = None,
    preflight_only: bool = False,
    preflight_count: int = 5,
) -> dict:
    """Run Stage 4 enrichment across all datasets.

    If preflight_only=True, processes only preflight_count tables across
    the first dataset found. Use this to validate the pipeline before
    committing to the full run.
    """
    processed_root = paths.processed_dir / "census"
    if not processed_root.exists():
        logger.error("No processed data at %s — run Stages 1-3 first", processed_root)
        return {"status": "error"}

    # Discover work items
    work_items: list[tuple[str, str]] = []
    for doc_path in sorted(processed_root.rglob("dataset_document.json")):
        parts = doc_path.relative_to(processed_root).parts
        if len(parts) < 3:
            continue
        year = parts[0]
        dataset_path = "/".join(parts[1:-1])
        if dataset_filter and dataset_filter not in dataset_path:
            continue
        work_items.append((year, dataset_path))

    if preflight_only:
        logger.info("PREFLIGHT MODE: processing %d tables to validate pipeline", preflight_count)
        work_items = work_items[:1]  # Just the first dataset

    logger.info("Stage 4: %d (year, dataset) pairs to process", len(work_items))

    all_results = []
    total_ok = total_failed = total_skipped = 0

    for year, dataset_path in work_items:
        if preflight_only:
            # Override table list to just the first N
            dataset_doc = _load_json(paths.processed_dataset_document(year, dataset_path))
            table_ids = (dataset_doc or {}).get("table_ids", [])[:preflight_count]
            if not table_ids:
                continue

            pf_ok = pf_failed = 0
            for table_id in table_ids:
                result = await enrich_one_table(
                    year, dataset_path, table_id,
                    client, loader, inference_cfg, paths, state,
                )
                if result["status"] == "success":
                    pf_ok += 1
                    logger.info("  ✓ %s/%s/%s", year, dataset_path, table_id)
                else:
                    pf_failed += 1
                    logger.error("  ✗ %s/%s/%s: %s", year, dataset_path, table_id,
                                 result.get("reason"))
            all_results.append({
                "year": year, "dataset_path": dataset_path,
                "tables_ok": pf_ok, "tables_failed": pf_failed,
            })
            total_ok += pf_ok
            total_failed += pf_failed
        else:
            result = await enrich_dataset(
                year, dataset_path, client, loader, inference_cfg, paths, state,
                skip_if_complete=skip_if_complete,
                max_concurrent=max_concurrent_tables,
            )
            result["year"] = year
            result["dataset_path"] = dataset_path
            all_results.append(result)
            total_ok += result.get("tables_ok", 0)
            total_failed += result.get("tables_failed", 0)
            total_skipped += result.get("tables_skipped", 0)

    return {
        "datasets_processed": len(all_results),
        "tables_ok": total_ok,
        "tables_failed": total_failed,
        "tables_skipped": total_skipped,
        "results": all_results,
        "preflight_mode": preflight_only,
    }
