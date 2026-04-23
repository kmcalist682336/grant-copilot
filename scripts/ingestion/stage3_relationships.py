"""Stage 3: Relationship Mapping.

Reads stage1_metadata.json and stage2_structure.json per table and produces
stage3_relationships.json containing:

  Per-table (grouped datasets):
    - parent/child/sibling maps derived from the label hierarchy
    - denominator/numerator pairs for rate calculations
    - "commonly queried" variable flags (used by Stage 7 question generation
      to decide which variables earn their own questions)

  Per-dataset (all structure types):
    - dataset_level_context.json: how dataset-level variables relate to
      every table in the dataset
        geography_identifiers  → always attached to every query response
        required_predicates    → must be specified; carry over to every table
        optional_predicates    → can be used to filter any table
        record_identifiers     → available in every microdata row

Stage 3 depends on Stage 2 having run first for each table.
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
    get_moe_sibling,
    parse_label,
    parse_variable_id,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage3_relationships"
DEPENDS_ON_S1 = "stage1_metadata"
DEPENDS_ON_S2 = "stage2_structure"

# Phrases in variable labels that indicate commonly-queried concepts.
# Variables matching any of these earn their own question generation in Stage 7
# even if they're at depth >= 2 (i.e., crosstab cells).
#
# Covers both ACS detailed table label patterns (colon-terminated hierarchy)
# and ACS subject/profile table patterns (plain English, no colons).
# Matching is case-insensitive substring match against the full label and
# joined segment string.
COMMONLY_QUERIED_PHRASES: frozenset[str] = frozenset({
    # --- Poverty ---
    "below poverty level",
    "in poverty",
    "below 100 percent of the poverty level",
    "poverty rate",
    "income below poverty",
    "below the poverty",
    # --- Education ---
    "bachelor's degree or higher",
    "bachelor's degree",
    "high school graduate or higher",
    "high school diploma",
    "less than high school",
    "graduate or professional degree",
    "some college",
    "associate's degree",
    "no schooling completed",
    # --- Health insurance ---
    "no health insurance",
    "without health insurance coverage",
    "uninsured",
    "with health insurance",
    "health insurance coverage",
    "public coverage",
    "private coverage",
    "medicaid",
    "medicare",
    # --- Nativity / citizenship ---
    "foreign born",
    "not a u.s. citizen",
    "native born",
    "naturalized",
    "born outside the united states",
    # --- Employment ---
    "unemployed",
    "unemployment",
    "in labor force",
    "not in labor force",
    "employed",
    "civilian labor force",
    "in the labor force",
    # --- Housing tenure ---
    "owner-occupied",
    "renter-occupied",
    "owner occupied",
    "renter occupied",
    "homeowner",
    "renter",
    # --- Income (subject table patterns use plain values like "$10,000") ---
    "median household income",
    "median family income",
    "median earnings",
    "median income",
    "per capita income",
    "aggregate income",
    "less than $10,000",
    "less than $25,000",
    "less than $15,000",
    "$100,000 or more",
    "$75,000 or more",
    "$50,000 to $74,999",
    "$35,000 to $49,999",
    "$25,000 to $34,999",
    # --- Housing costs ---
    "median gross rent",
    "median home value",
    "median value",
    "median monthly costs",
    "gross rent as a percentage",
    "selected monthly owner costs",
    "housing costs",
    "cost burden",
    # --- Language ---
    "speaks english less than very well",
    "speak only english",
    "speaks only english",
    "english only",
    "language other than english",
    "speak english",
    "spanish",
    "limited english",
    # --- Disability ---
    "disability",
    "with a disability",
    "with any disability",
    "hearing difficulty",
    "vision difficulty",
    "cognitive difficulty",
    "ambulatory difficulty",
    "self-care difficulty",
    "independent living difficulty",
    # --- Veterans ---
    "veteran",
    "veterans",
    "service-connected disability",
    "period of service",
    # --- Sex ---
    "male",
    "female",
    # --- Age ---
    "under 18 years",
    "65 years and over",
    "under 5 years",
    "18 to 64 years",
    "18 years and over",
    "median age",
    # --- Race and ethnicity ---
    "white alone",
    "black or african american alone",
    "hispanic or latino",
    "asian alone",
    "american indian",
    "native hawaiian",
    "two or more races",
    "some other race",
    "non-hispanic white",
    "not hispanic or latino",
    # --- Food assistance ---
    "snap",
    "food stamps",
    "received snap",
    # --- Public assistance ---
    "public assistance",
    "social security",
    "supplemental security income",
    "cash public assistance",
    "retirement income",
    # --- Technology / internet (S2801) ---
    "no internet",
    "no computer",
    "broadband",
    "with a computer",
    "internet subscription",
    "dial-up",
    "cellular data plan",
    # --- Commuting / transportation ---
    "drove alone",
    "carpooled",
    "public transportation",
    "worked from home",
    "mean travel time",
    "travel time to work",
    "walked",
    # --- Family / household structure ---
    "married-couple",
    "single parent",
    "female householder",
    "male householder",
    "living alone",
    "nonfamily households",
    "family households",
    "average household size",
    "average family size",
    # --- Poverty-adjacent programs ---
    "free or reduced",
    "school lunch",
    "wic",
    # --- Geographic mobility ---
    "same house",
    "moved",
    "different state",
    "different county",
    "from abroad",
})


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


# ---------------------------------------------------------------------------
# Parent/child/sibling mapping from label hierarchy
# ---------------------------------------------------------------------------

def build_hierarchy_map(variable_inventory: list[dict]) -> dict:
    """Build parent/child/sibling relationships from the variable inventory.

    Uses segment paths as the basis for hierarchy — a variable's parent is
    the variable whose segment path is a prefix of the current one.

    Returns:
        {
          "parents":  {var_id: parent_var_id or None},
          "children": {var_id: [child_var_id, ...]},
          "siblings": {var_id: [sibling_var_id, ...]},
        }
    """
    # Build a lookup from segment tuple → var_id for matching
    # Only use estimate variables (not MOE)
    estimates = [v for v in variable_inventory if v.get("kind") == "estimate"]

    # Map from frozen segments tuple → var_id for O(1) parent lookup
    segments_to_var: dict[tuple, str] = {}
    for entry in estimates:
        segs = tuple(entry.get("segments", []))
        segments_to_var[segs] = entry["var_id"]

    parents: dict[str, Optional[str]] = {}
    children: dict[str, list[str]] = defaultdict(list)

    for entry in estimates:
        var_id = entry["var_id"]
        segs = tuple(entry.get("segments", []))

        # Try progressively shorter prefix paths to find the nearest parent
        parent_id = None
        for length in range(len(segs) - 1, 0, -1):
            candidate_segs = segs[:length]
            # Parent must be a subtotal node (it has children under it)
            candidate_id = segments_to_var.get(candidate_segs)
            if candidate_id and candidate_id != var_id:
                parent_id = candidate_id
                break

        parents[var_id] = parent_id
        if parent_id:
            children[parent_id].append(var_id)

    # Siblings: share the same parent
    siblings: dict[str, list[str]] = defaultdict(list)
    for var_id, parent_id in parents.items():
        if parent_id:
            sibs = [c for c in children[parent_id] if c != var_id]
            siblings[var_id] = sorted(sibs)

    return {
        "parents": parents,
        "children": {k: sorted(v) for k, v in children.items()},
        "siblings": dict(siblings),
    }


# ---------------------------------------------------------------------------
# Denominator / numerator pairing
# ---------------------------------------------------------------------------

def find_denominator_pairs(
    variable_inventory: list[dict],
    root_total_variable: Optional[str],
    universe: Optional[str],
) -> list[dict]:
    """Identify natural numerator→denominator pairs for rate calculations.

    The primary denominator for most tables is the root total (_001E).
    Secondary denominators are subtotals that are natural "wholes" for their
    child cells (e.g., "Total:!!Male:" is the denominator for all male
    age-group cells).

    Returns a list of pairs:
        [{"numerator": var_id, "denominator": var_id, "relationship": str}, ...]
    """
    pairs = []
    estimates = {v["var_id"]: v for v in variable_inventory if v.get("kind") == "estimate"}

    if not estimates or not root_total_variable:
        return pairs

    # Every non-total, non-subtotal estimate is a potential numerator
    # whose denominator is its nearest subtotal ancestor (or root)
    for var_id, entry in estimates.items():
        if var_id == root_total_variable:
            continue
        if entry.get("is_total") or entry.get("is_subtotal"):
            # Subtotals serve as denominators, not numerators (except vs. root)
            if var_id != root_total_variable:
                pairs.append({
                    "numerator": var_id,
                    "denominator": root_total_variable,
                    "relationship": "subtotal_of_total",
                })
        else:
            # Leaf cell — find the nearest subtotal ancestor as denominator
            segs = tuple(entry.get("segments", []))
            best_denom = root_total_variable
            # Walk up to find the most specific subtotal ancestor
            for length in range(len(segs) - 1, 0, -1):
                ancestor_segs = segs[:length]
                for cand_id, cand_entry in estimates.items():
                    if (tuple(cand_entry.get("segments", [])) == ancestor_segs
                            and cand_entry.get("is_subtotal")):
                        best_denom = cand_id
                        break
                if best_denom != root_total_variable:
                    break

            pairs.append({
                "numerator": var_id,
                "denominator": best_denom,
                "relationship": (
                    "leaf_of_subtotal"
                    if best_denom != root_total_variable
                    else "leaf_of_total"
                ),
            })

    return pairs


# ---------------------------------------------------------------------------
# Commonly-queried variable detection
# ---------------------------------------------------------------------------

def flag_commonly_queried(variable_inventory: list[dict]) -> set[str]:
    """Return the set of variable IDs that match commonly-queried phrases.

    These variables earn their own question generation in Stage 7 even if
    they appear deep in a crosstab hierarchy.
    """
    flagged = set()
    for entry in variable_inventory:
        if entry.get("kind") not in ("estimate", "percent_estimate"):
            continue
        label_lower = entry.get("label_raw", "").lower()
        segments_lower = " ".join(entry.get("segments", [])).lower()
        combined = label_lower + " " + segments_lower
        if any(phrase in combined for phrase in COMMONLY_QUERIED_PHRASES):
            flagged.add(entry["var_id"])
    return flagged


# ---------------------------------------------------------------------------
# Per-table Stage 3 processing
# ---------------------------------------------------------------------------

def process_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    dataset_structure: str,
    paths: PipelinePaths,
    state: StateTracker,
) -> dict:
    """Run Stage 3 relationship mapping on a single table."""
    s1_path = paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON_S1)
    s2_path = paths.processed_table_stage_file(year, dataset_path, table_id, DEPENDS_ON_S2)

    s1 = _load_json(s1_path)
    s2 = _load_json(s2_path)

    if not s1 or not s2:
        msg = f"Missing stage1 or stage2 for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        state.set_status(
            year=year, dataset_path=dataset_path,
            entity_type="table", entity_id=table_id,
            stage=STAGE_NAME, status=Status.FAILED_NO_DATA,
            error_message=msg,
        )
        return {"status": "failed", "reason": "missing_prerequisite"}

    collected_at = datetime.now(timezone.utc).isoformat()
    analysis_type = s2.get("analysis_type", "grouped")

    if analysis_type != "grouped":
        # Flat and microdata tables have no internal hierarchy to map
        output = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "collected_at": collected_at,
            "year": year,
            "dataset_path": dataset_path,
            "dataset_structure": dataset_structure,
            "table_id": table_id,
            "analysis_type": analysis_type,
            "note": (
                "Relationship mapping not applicable for "
                f"{analysis_type} datasets"
            ),
            "hierarchy": {},
            "denominator_pairs": [],
            "commonly_queried_variables": [],
        }
    else:
        inventory = s2.get("variable_inventory", [])
        root_total = s2.get("root_total_variable")
        universe = s1.get("universe")

        hierarchy = build_hierarchy_map(inventory)
        denom_pairs = find_denominator_pairs(inventory, root_total, universe)
        commonly_queried = sorted(flag_commonly_queried(inventory))

        output = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "collected_at": collected_at,
            "year": year,
            "dataset_path": dataset_path,
            "dataset_structure": dataset_structure,
            "table_id": table_id,
            "analysis_type": analysis_type,
            "root_total_variable": root_total,
            "hierarchy": hierarchy,
            "denominator_pairs": denom_pairs,
            "denominator_pair_count": len(denom_pairs),
            "commonly_queried_variables": commonly_queried,
            "commonly_queried_count": len(commonly_queried),
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
            "commonly_queried_count": len(
                output.get("commonly_queried_variables", [])
            ),
            "denominator_pair_count": len(
                output.get("denominator_pairs", [])
            ),
        },
    )
    return {"status": "success", "table_id": table_id}


# ---------------------------------------------------------------------------
# Dataset-level context file
# ---------------------------------------------------------------------------

def build_dataset_level_context(
    year: str,
    dataset_path: str,
    dataset_doc: dict,
    table_ids: list[str],
    paths: PipelinePaths,
) -> dict:
    """Build a dataset_level_context.json summarizing how dataset-level
    variables relate to every table in the dataset.

    This is the Stage 3 counterpart to the per-table relationship files.
    It describes which variables "attach" to every table query rather than
    belonging to any specific table.
    """
    dlv = dataset_doc.get("dataset_level_variables", {})
    qr = dataset_doc.get("query_requirements", {})

    # Strip the _role field for cleaner output — role is already encoded
    # by which bucket the variable is in
    def _clean(vars_dict: dict) -> dict:
        return {
            var_id: {k: v for k, v in meta.items() if k != "_role"}
            for var_id, meta in vars_dict.items()
        }

    geo_ids = _clean(dlv.get("geography_identifiers", {}))
    rec_ids = _clean(dlv.get("record_identifiers", {}))
    req_preds = _clean(dlv.get("required_predicates", {}))
    opt_preds = _clean(dlv.get("optional_predicates", {}))
    time_dims = _clean(dlv.get("time_dimensions", {}))

    return {
        "schema_version": "1.0.0",
        "stage": STAGE_NAME,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "dataset_path": dataset_path,
        "dataset_structure": dataset_doc.get("dataset_structure", "grouped"),
        "table_count": len(table_ids),
        # These variables are always returned alongside any table query
        "always_attached": {
            "geography_identifiers": geo_ids,
            "record_identifiers": rec_ids,
            "time_dimensions": time_dims,
        },
        # These affect how queries must be structured
        "query_constraints": {
            "has_required_predicates": qr.get("has_required_predicates", False),
            "required_predicates": req_preds,
            "optional_predicates": opt_preds,
            "is_microdata": qr.get("is_microdata", False),
            "is_flat": qr.get("is_flat", False),
        },
        # Stage 6 note: question generation should incorporate required
        # predicates into generated questions for this dataset. A question
        # about PEP data should mention the specific population filter
        # (e.g., "Hispanic population" not just "population").
        "stage6_note": (
            "Include required_predicate dimensions in generated questions"
            if qr.get("has_required_predicates")
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------

def run_stage3_for_dataset(
    year: str,
    dataset_path: str,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    skip_if_complete: bool = True,
) -> dict:
    """Run Stage 3 for all tables in a dataset, plus the dataset-level context."""
    # Load dataset document for structure info and dataset-level variables
    dataset_doc_path = paths.processed_dataset_document(year, dataset_path)
    dataset_doc = _load_json(dataset_doc_path)
    if not dataset_doc:
        logger.warning(
            "dataset_document.json not found for %s/%s — skipping",
            year, dataset_path,
        )
        return {"status": "failed", "reason": "missing_dataset_document"}

    dataset_structure = dataset_doc.get("dataset_structure", "grouped")
    table_ids = dataset_doc.get("table_ids", [])

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
                "Unexpected error in Stage 3 for %s/%s/%s",
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

    # Write dataset-level context file regardless of table failures
    ctx = build_dataset_level_context(
        year, dataset_path, dataset_doc, table_ids, paths
    )
    ctx_path = (
        paths.processed_dataset_dir(year, dataset_path)
        / "stage3_dataset_context.json"
    )
    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(ctx_path, ctx)

    logger.info(
        "%s/%s Stage 3: %d ok, %d failed, %d skipped",
        year, dataset_path, tables_ok, tables_failed, tables_skipped,
    )
    return {
        "status": "success" if tables_failed == 0 else "partial",
        "tables_ok": tables_ok,
        "tables_failed": tables_failed,
        "tables_skipped": tables_skipped,
    }


def run_stage3(
    years: list[str],
    paths: PipelinePaths,
    state: StateTracker,
    *,
    max_concurrent_datasets: int = 4,
    skip_if_complete: bool = True,
    dataset_filter: Optional[str] = None,
) -> dict:
    """Run Stage 3 across all years and datasets found in processed/."""
    processed_root = paths.processed_dir / "census"
    if not processed_root.exists():
        logger.error("No processed data found at %s — run Stages 1-2 first", processed_root)
        return {"status": "error", "reason": "no_processed_data"}

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

    logger.info("Stage 3: %d (year, dataset) pairs to process", len(work_items))

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent_datasets) as pool:
        future_to_item = {
            pool.submit(
                run_stage3_for_dataset,
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
                logger.exception("Stage 3 failed for %s/%s", year, dataset_path)
                results.append({
                    "year": year, "dataset_path": dataset_path,
                    "status": "failed", "error": str(e),
                })

    ok = sum(1 for r in results if r.get("status") == "success")
    partial = sum(1 for r in results if r.get("status") == "partial")
    failed = sum(1 for r in results if r.get("status") == "failed")
    total_tables = sum(r.get("tables_ok", 0) for r in results)

    logger.info(
        "Stage 3 complete: %d datasets ok, %d partial, %d failed; %d tables mapped",
        ok, partial, failed, total_tables,
    )
    return {
        "datasets_ok": ok,
        "datasets_partial": partial,
        "datasets_failed": failed,
        "tables_mapped": total_tables,
        "results": results,
    }
