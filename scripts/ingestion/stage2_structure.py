"""Stage 2: Structural Analysis.

Reads stage1_metadata.json per table and produces stage2_structure.json
containing the table's dimensional structure, headline variables, measure
type, subject area, and a parsed variable hierarchy.

Also reads the parent dataset_document.json to carry forward dataset_structure
so downstream stages don't need to open two files.

For flat and microdata datasets the analysis is simplified — there is no
label hierarchy to parse, so we produce a flat variable inventory instead.
"""
from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from scripts.utils.label_parser import (
    MeasureKind,
    MeasureType,
    ParsedLabel,
    VariableIDParts,
    get_moe_sibling,
    get_subject_area,
    infer_measure_type,
    parse_label,
    parse_variable_id,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2_structure"
DEPENDS_ON = "stage1_metadata"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stage1(
    year: str, dataset_path: str, table_id: str, paths: PipelinePaths
) -> Optional[dict]:
    """Load stage1_metadata.json for a table. Returns None if missing."""
    p = paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_document(
    year: str, dataset_path: str, paths: PipelinePaths
) -> Optional[dict]:
    """Load dataset_document.json for the parent dataset. Returns None if missing."""
    p = paths.processed_dataset_document(year, dataset_path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Variable classification helpers
# ---------------------------------------------------------------------------

def _classify_variable_kind(var_id: str, label: str) -> str:
    """Classify a variable by what kind of value it represents.

    Returns one of: estimate, moe, annotation, percent_estimate, percent_moe
    """
    parts = parse_variable_id(var_id)
    if parts:
        if parts.is_annotation:
            return "annotation"
        if parts.is_moe:
            return "percent_moe" if parts.is_percent else "moe"
        if parts.is_estimate:
            return "percent_estimate" if parts.is_percent else "estimate"
    # Fall back to label-based detection for non-standard IDs
    label_lower = label.lower()
    if "margin of error" in label_lower:
        return "moe"
    if "annotation" in label_lower:
        return "annotation"
    return "estimate"


# ---------------------------------------------------------------------------
# Core structural analysis for grouped (standard) tables
# ---------------------------------------------------------------------------

def analyze_grouped_table(
    table_id: str,
    variables: dict[str, dict],
    title: Optional[str],
    universe: Optional[str],
    dataset_path: str,
) -> dict:
    """Analyze a standard grouped table with a label hierarchy.

    Returns a dict describing the table's structure. This is the main
    output for ACS, Decennial, and PEP tables.
    """
    # --- Step 1: Parse every variable label ---
    parsed: dict[str, ParsedLabel] = {}
    variable_kinds: dict[str, str] = {}

    for var_id, meta in variables.items():
        label = meta.get("label", "")
        parsed[var_id] = parse_label(label)
        variable_kinds[var_id] = _classify_variable_kind(var_id, label)

    # --- Step 2: Separate estimates from MOE/annotation ---
    estimate_vars = {v: p for v, p in parsed.items()
                     if variable_kinds[v] in ("estimate", "percent_estimate")}
    moe_vars = {v for v in parsed if variable_kinds[v] in ("moe", "percent_moe")}
    annotation_vars = {v for v in parsed if variable_kinds[v] == "annotation"}

    # --- Step 3: Find headline variables ---
    # Root total: the shallowest estimate variable, usually _001E
    # Ties broken by sequence number ascending
    headline_variables: list[str] = []
    root_total: Optional[str] = None

    # Sort by (depth, sequence number) to find shallowest first
    def _sort_key(var_id: str) -> tuple:
        p = parse_variable_id(var_id)
        seq = int(p.sequence_number) if p else 9999
        return (parsed[var_id].depth, seq)

    sorted_estimates = sorted(estimate_vars.keys(), key=_sort_key)

    if sorted_estimates:
        root_total = sorted_estimates[0]
        headline_variables.append(root_total)

        # Top-level subtotals: depth == 1 (one level below root)
        # These are the primary dimension breakdowns (e.g., Male/Female totals)
        for var_id in sorted_estimates[1:]:
            pl = estimate_vars[var_id]
            if pl.depth == 1 and pl.is_subtotal:
                headline_variables.append(var_id)

    # --- Step 4: Infer dimensional structure ---
    # The "dimensions" of the table are the distinct segment positions
    # across all estimate variables. E.g., a table crossing Sex × Age × Poverty
    # will have 3 dimension levels.
    max_depth = max((p.depth for p in estimate_vars.values()), default=0)

    # Collect distinct values at each depth level to name dimensions
    # (heuristic — works well for ACS detailed, less so for profile)
    dimension_values: dict[int, set[str]] = defaultdict(set)
    for pl in estimate_vars.values():
        for i, seg in enumerate(pl.segments):
            dimension_values[i].add(seg)

    # Infer dimension names from position and content
    dimensions = _infer_dimensions(dimension_values, max_depth, dataset_path)

    # --- Step 5: Detect section headers (profile/subject tables) ---
    section_headers = sorted({
        pl.section_header
        for pl in estimate_vars.values()
        if pl.section_header
    })

    # --- Step 6: Subject area and measure type ---
    subject_area = get_subject_area(table_id)
    measure_type = infer_measure_type(table_id, title)

    # --- Step 7: Build variable inventory (estimates only — MOE linked) ---
    variable_inventory = _build_variable_inventory(
        estimate_vars, moe_vars, parsed, variable_kinds, variables
    )

    return {
        "analysis_type": "grouped",
        "measure_type": measure_type.value,
        "subject_area": subject_area,
        "max_depth": max_depth,
        "dimensions": dimensions,
        "section_headers": section_headers,
        "headline_variables": headline_variables,
        "root_total_variable": root_total,
        "variable_counts": {
            "total": len(variables),
            "estimate": len(estimate_vars),
            "moe": len(moe_vars),
            "annotation": len(annotation_vars),
        },
        "variable_inventory": variable_inventory,
    }


def _infer_dimensions(
    dimension_values: dict[int, set[str]],
    max_depth: int,
    dataset_path: str,
) -> list[dict]:
    """Infer dimension names and value counts from the label structure."""
    dimensions = []
    for i in range(max_depth):
        values = dimension_values.get(i, set())
        # Heuristic name: if position 0 has just "Total", it's the universe
        # If position 0 has >1 value it's the first classification dimension
        if i == 0 and len(values) <= 2:
            name = "total"
        else:
            name = f"dimension_{i + 1}"

        dimensions.append({
            "position": i,
            "name": name,
            "distinct_value_count": len(values),
            # Store up to 20 sample values to aid LLM enrichment in Stage 4
            "sample_values": sorted(values)[:20],
        })
    return dimensions


def _build_variable_inventory(
    estimate_vars: dict[str, ParsedLabel],
    moe_vars: set[str],
    parsed: dict[str, ParsedLabel],
    variable_kinds: dict[str, str],
    raw_variables: dict[str, dict],
) -> list[dict]:
    """Build a flat inventory of estimate variables with structural annotations."""
    inventory = []
    for var_id, pl in estimate_vars.items():
        raw_meta = raw_variables.get(var_id, {})
        moe_id = get_moe_sibling(var_id)
        has_moe = moe_id in moe_vars if moe_id else False

        inventory.append({
            "var_id": var_id,
            "label_raw": raw_meta.get("label", ""),
            "segments": pl.segments,
            "depth": pl.depth,
            "is_subtotal": pl.is_subtotal,
            "is_total": pl.is_total,
            "measure_kind": pl.measure_kind.value,
            "kind": variable_kinds[var_id],
            "moe_var_id": moe_id if has_moe else None,
            "predicate_type": raw_meta.get("predicateType", ""),
        })

    # Sort by variable ID for stable output
    inventory.sort(key=lambda x: x["var_id"])
    return inventory


# ---------------------------------------------------------------------------
# Simplified analysis for flat and microdata datasets
# ---------------------------------------------------------------------------

def analyze_flat_table(
    table_id: str,
    variables: dict[str, dict],
    title: Optional[str],
    dataset_path: str,
) -> dict:
    """Analyze a flat-structure dataset (pdb, geoinfo, etc.).

    Flat datasets have no label hierarchy — each variable is an independent
    data field. We produce a simple type inventory instead of a hierarchy.
    """
    inventory = []
    for var_id, meta in sorted(variables.items()):
        predicate_type = meta.get("predicateType", "")
        label = meta.get("label", "")

        # Infer data type from predicateType field
        if predicate_type == "int":
            data_type = "integer"
        elif predicate_type == "float":
            data_type = "float"
        elif predicate_type == "string":
            data_type = "string"
        else:
            data_type = "unknown"

        # Rough semantic classification from label content
        label_lower = label.lower()
        if any(w in label_lower for w in ("percent", "pct", "rate", "%")):
            semantic = "percentage"
        elif any(w in label_lower for w in ("median", "mean", "average")):
            semantic = "summary_statistic"
        elif any(w in label_lower for w in ("count", "number", "total", "population")):
            semantic = "count"
        elif any(w in label_lower for w in ("name", "code", "fips", "id")):
            semantic = "identifier"
        else:
            semantic = "unknown"

        inventory.append({
            "var_id": var_id,
            "label_raw": label,
            "concept": meta.get("concept", ""),
            "data_type": data_type,
            "semantic": semantic,
        })

    return {
        "analysis_type": "flat",
        "measure_type": MeasureType.UNKNOWN.value,
        "subject_area": get_subject_area(table_id),
        "variable_counts": {"total": len(variables)},
        "variable_inventory": inventory,
    }


def analyze_microdata_dataset(
    dataset_path: str,
    variables: dict[str, dict],
) -> dict:
    """Analyze a microdata dataset (CPS, PUMS).

    Microdata variables have no table grouping. We classify them into
    weight, identifier, and substantive categories.
    """
    weights, identifiers, substantive, other = [], [], [], []

    for var_id, meta in sorted(variables.items()):
        label = meta.get("label", "").lower()
        if "weight" in label or var_id.upper().endswith("WGT"):
            weights.append(var_id)
        elif any(w in label for w in ("identifier", "serial", "id number", "unique")):
            identifiers.append(var_id)
        elif meta.get("predicateType") in ("int", "float"):
            substantive.append(var_id)
        else:
            other.append(var_id)

    return {
        "analysis_type": "microdata",
        "measure_type": MeasureType.UNKNOWN.value,
        "subject_area": None,
        "variable_counts": {
            "total": len(variables),
            "weights": len(weights),
            "identifiers": len(identifiers),
            "substantive": len(substantive),
            "other": len(other),
        },
        "weight_variables": sorted(weights),
        "identifier_variables": sorted(identifiers),
    }


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

def process_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    dataset_structure: str,
    paths: PipelinePaths,
    state: StateTracker,
) -> dict:
    """Run Stage 2 analysis on a single table."""
    stage1 = load_stage1(year, dataset_path, table_id, paths)
    if not stage1:
        msg = f"stage1_metadata.json not found for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        state.set_status(
            year=year, dataset_path=dataset_path,
            entity_type="table", entity_id=table_id,
            stage=STAGE_NAME, status=Status.FAILED_NO_DATA,
            error_message=msg,
        )
        return {"status": "failed", "reason": "missing_stage1"}

    variables = stage1.get("variables", {})
    title = stage1.get("title")
    universe = stage1.get("universe")
    collected_at = datetime.now(timezone.utc).isoformat()

    # Choose analysis strategy based on dataset structure
    if dataset_structure == "flat":
        analysis = analyze_flat_table(table_id, variables, title, dataset_path)
    elif dataset_structure == "microdata":
        # For microdata, table_id is synthetic — the variables are the dataset
        analysis = analyze_microdata_dataset(dataset_path, variables)
    else:
        analysis = analyze_grouped_table(
            table_id, variables, title, universe, dataset_path
        )

    output = {
        "schema_version": "1.0.0",
        "stage": STAGE_NAME,
        "collected_at": collected_at,
        "source": "census",
        "year": year,
        "dataset_path": dataset_path,
        "dataset_structure": dataset_structure,
        "table_id": table_id,
        "title": title,
        "universe": universe,
        **analysis,
    }

    out_path = paths.processed_table_stage_file(
        year, dataset_path, table_id, STAGE_NAME
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_path, output)

    state.set_status(
        year=year, dataset_path=dataset_path,
        entity_type="table", entity_id=table_id,
        stage=STAGE_NAME, status=Status.SUCCESS,
        metadata={
            "analysis_type": analysis.get("analysis_type"),
            "measure_type": analysis.get("measure_type"),
            "variable_count": len(variables),
        },
    )
    return {"status": "success", "table_id": table_id}


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

def run_stage2_for_dataset(
    year: str,
    dataset_path: str,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
) -> dict:
    """Run Stage 2 for all tables in a dataset."""
    dataset_doc = load_dataset_document(year, dataset_path, paths)
    if not dataset_doc:
        logger.warning(
            "dataset_document.json not found for %s/%s — skipping",
            year, dataset_path,
        )
        return {"status": "failed", "reason": "missing_dataset_document"}

    dataset_structure = dataset_doc.get("dataset_structure", "grouped")
    table_ids = dataset_doc.get("table_ids", [])

    if not table_ids:
        logger.info("%s/%s: no tables to process", year, dataset_path)
        return {"status": "success", "tables_processed": 0}

    tables_ok = tables_failed = tables_skipped = 0

    for table_id in sorted(table_ids):
        if skip_if_complete and state.is_complete(
            year=year, dataset_path=dataset_path,
            entity_type="table", entity_id=table_id,
            stage=STAGE_NAME,
        ):
            tables_skipped += 1
            continue

        try:
            result = process_one_table(
                year, dataset_path, table_id, dataset_structure, paths, state
            )
            if result["status"] == "success":
                tables_ok += 1
            else:
                tables_failed += 1
        except Exception as e:
            tables_failed += 1
            logger.exception(
                "Unexpected error in Stage 2 for %s/%s/%s",
                year, dataset_path, table_id,
            )
            state.set_status(
                year=year, dataset_path=dataset_path,
                entity_type="table", entity_id=table_id,
                stage=STAGE_NAME, status=Status.FAILED_OTHER,
                error_message=str(e),
            )
            state.log_error(
                year=year, dataset_path=dataset_path,
                entity_id=table_id, stage=STAGE_NAME,
                error_type=type(e).__name__, error_message=str(e),
                traceback=traceback.format_exc(),
            )

    logger.info(
        "%s/%s Stage 2: %d ok, %d failed, %d skipped",
        year, dataset_path, tables_ok, tables_failed, tables_skipped,
    )
    return {
        "status": "success" if tables_failed == 0 else "partial",
        "tables_ok": tables_ok,
        "tables_failed": tables_failed,
        "tables_skipped": tables_skipped,
    }


def run_stage2(
    years: list[str],
    paths: PipelinePaths,
    state: StateTracker,
    *,
    max_concurrent_datasets: int = 4,
    skip_if_complete: bool = True,
    dataset_filter: Optional[str] = None,
) -> dict:
    """Run Stage 2 across all years and datasets found in processed/."""
    processed_root = paths.processed_dir / "census"
    if not processed_root.exists():
        logger.error("No processed data found at %s — run Stage 1 first", processed_root)
        return {"status": "error", "reason": "no_processed_data"}

    # Discover (year, dataset_path) pairs from existing dataset_document.json files
    work_items: list[tuple[str, str]] = []
    for doc_path in sorted(processed_root.rglob("dataset_document.json")):
        # Path: processed/census/<year>/<dataset_path>/dataset_document.json
        parts = doc_path.relative_to(processed_root).parts
        if len(parts) < 3:
            continue
        year = parts[0]
        dataset_path = "/".join(parts[1:-1])
        if dataset_filter and dataset_filter not in dataset_path:
            continue
        work_items.append((year, dataset_path))

    logger.info("Stage 2: %d (year, dataset) pairs to process", len(work_items))

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_datasets) as pool:
        future_to_item = {
            pool.submit(
                run_stage2_for_dataset,
                year, dataset_path, paths, state,
                skip_if_complete=skip_if_complete,
            ): (year, dataset_path)
            for year, dataset_path in work_items
        }
        for future in as_completed(future_to_item):
            year, dataset_path = future_to_item[future]
            try:
                result = future.result()
                result["year"] = year
                result["dataset_path"] = dataset_path
                results.append(result)
            except Exception as e:
                logger.exception("Stage 2 failed for %s/%s", year, dataset_path)
                results.append({
                    "year": year, "dataset_path": dataset_path,
                    "status": "failed", "error": str(e),
                })

    ok = sum(1 for r in results if r.get("status") == "success")
    partial = sum(1 for r in results if r.get("status") == "partial")
    failed = sum(1 for r in results if r.get("status") == "failed")
    total_tables = sum(r.get("tables_ok", 0) for r in results)

    logger.info(
        "Stage 2 complete: %d datasets ok, %d partial, %d failed; %d tables analyzed",
        ok, partial, failed, total_tables,
    )
    return {
        "datasets_ok": ok,
        "datasets_partial": partial,
        "datasets_failed": failed,
        "tables_analyzed": total_tables,
        "results": results,
    }
