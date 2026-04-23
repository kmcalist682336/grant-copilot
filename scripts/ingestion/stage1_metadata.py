"""Stage 1: Metadata Collection.

This stage collects raw metadata from the Census API for every (year, dataset)
pair in our allowlist, then partitions the metadata into per-table outputs.

Structure:
  1a. Dataset discovery — walk the year endpoint, filter via allowlist,
      record which datasets we intend to process.
  1b. Per-dataset metadata fetch — for each dataset, pull and cache:
          variables.json, geography.json, groups.json, examples.json, tags.json,
          and the dataset's entry in the year catalog.
  1c. Per-table partitioning — for each table in the dataset, emit a
      stage1_metadata.json file containing its title, universe, variable list,
      supported geography levels, and release metadata.

All raw API responses are cached under data/raw/. Stage outputs go to
data/processed/. Status is tracked in the SQLite state DB so re-runs skip
completed work.
"""
from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from scripts.utils.census_api import (
    CensusAPIClient,
    CensusAPIError,
    CensusNotFoundError,
    CensusPermanentError,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage1_metadata"

# Files we try to fetch per dataset. Some are optional — missing files are
# recorded but don't fail the dataset.
DATASET_METADATA_FILES = {
    "variables.json": {"required": True},
    "geography.json": {"required": True},
    "groups.json": {"required": True},
    "examples.json": {"required": False},
    "tags.json": {"required": False},
}


# ---------------------------------------------------------------------------
# Data classes representing the planned work and its results
# ---------------------------------------------------------------------------


@dataclass
class DatasetTarget:
    """A (year, dataset_path) pair we plan to process."""

    year: str
    dataset_path: str
    tier: int = 1
    notes: Optional[str] = None
    dataset_structure: str = "grouped"  # grouped | flat | microdata

    @property
    def key(self) -> str:
        return f"{self.year}::{self.dataset_path}"


@dataclass
class DatasetMetadata:
    """All raw metadata for one dataset after Stage 1b."""

    target: DatasetTarget
    catalog_entry: Optional[dict] = None      # from year_catalog's dataset list
    variables: Optional[dict] = None          # full variables.json response
    geography: Optional[dict] = None          # full geography.json response
    groups: Optional[dict] = None             # full groups.json response
    examples: Optional[dict] = None           # optional
    tags: Optional[dict] = None               # optional
    fetch_errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1a: Discovery
# ---------------------------------------------------------------------------


def discover_datasets(
    year: str,
    client: CensusAPIClient,
    paths: PipelinePaths,
    *,
    force_refresh: bool = False,
) -> list[dict]:
    """Fetch and cache the year-level catalog of all available datasets.

    Returns the parsed catalog's `dataset` list, each entry being a dict
    containing `c_dataset` (path components), `title`, `description`, and
    links to the variables/geography/groups endpoints.
    """
    year_url = f"{client.config.base_url}/{year}/"
    cache_path = paths.raw_year_catalog(year)

    logger.info("Fetching year catalog for %s from %s", year, year_url)
    catalog = client.fetch_json(year_url, cache_path=cache_path, force_refresh=force_refresh)

    datasets = catalog.get("dataset", [])
    logger.info("Year %s catalog contains %d dataset entries", year, len(datasets))
    return datasets


def _matches_allowlist_entry(dataset_path: str, allowlist_path: str) -> bool:
    """Check whether a dataset path matches an allowlist entry.

    Matching is exact on the dataset path. This is more conservative than
    prefix matching — if a user wants acs/acs5 and all its sub-datasets,
    they should list them individually or use mode=all with skip_datasets.
    """
    return dataset_path == allowlist_path


def _matches_skip_entry(dataset_path: str, skip_entry: str) -> bool:
    """Prefix match against a skip entry."""
    return dataset_path == skip_entry or dataset_path.startswith(skip_entry + "/")


def _extract_dataset_path(catalog_entry: dict) -> Optional[str]:
    """Extract the dataset path from a catalog entry's c_dataset field."""
    components = catalog_entry.get("c_dataset")
    if not components or not isinstance(components, list):
        return None
    return "/".join(components)


def _infer_dataset_structure(dataset_path: str) -> str:
    """Infer the structural type of a dataset from its path.

    Used in mode:all where the allowlist doesn't explicitly tag datasets.
    Explicit tags in datasets_allowlist.yaml always take precedence over
    this inference.

    Dataset structures:
      grouped   — standard Census structure: variables belong to named tables
                  via the 'group' field. ACS, Decennial, PEP.
      flat      — all variables are dataset-level, no table grouping.
                  Planning Database (pdb), some specialty datasets.
      microdata — person/household-level records with survey weights and
                  row identifiers. CPS, ACS PUMS.
    """
    path = dataset_path.lower()
    if path.startswith("cps/") or path.startswith("acs/pums"):
        return "microdata"
    if path.startswith("pdb/"):
        return "flat"
    return "grouped"


def plan_dataset_targets(
    year: str,
    catalog_datasets: list[dict],
    allowlist_config: dict,
) -> tuple[list[DatasetTarget], list[dict]]:
    """Filter the year's catalog down to datasets we actually want to process.

    Returns:
        (targets, skipped) where targets is the list of DatasetTarget to
        process and skipped is the list of catalog entries that were filtered
        out (with reasons attached for logging/reporting).
    """
    mode = allowlist_config.get("mode", "allowlist")
    skip_datasets = allowlist_config.get("skip_datasets", [])
    allowlist = allowlist_config.get("datasets", [])

    # Build lookup for allowlist entries with their tier/notes/years_override
    allowlist_by_path: dict[str, dict] = {
        entry["path"]: entry for entry in allowlist if "path" in entry
    }

    targets: list[DatasetTarget] = []
    skipped: list[dict] = []

    for entry in catalog_datasets:
        ds_path = _extract_dataset_path(entry)
        if not ds_path:
            skipped.append({"entry": entry, "reason": "no dataset path in catalog entry"})
            continue

        # Global skip list always wins
        if any(_matches_skip_entry(ds_path, s) for s in skip_datasets):
            skipped.append({"path": ds_path, "reason": "matches skip_datasets"})
            continue

        # Mode-specific filtering
        if mode == "allowlist":
            if ds_path not in allowlist_by_path:
                skipped.append({"path": ds_path, "reason": "not in allowlist"})
                continue
            entry_cfg = allowlist_by_path[ds_path]
            # Respect years_override if present
            years_override = entry_cfg.get("years_override")
            if years_override and year not in [str(y) for y in years_override]:
                skipped.append({
                    "path": ds_path,
                    "reason": f"year {year} not in years_override {years_override}",
                })
                continue
            targets.append(DatasetTarget(
                year=year,
                dataset_path=ds_path,
                tier=entry_cfg.get("tier", 1),
                notes=entry_cfg.get("notes"),
                dataset_structure=entry_cfg.get("dataset_structure", "grouped"),
            ))
        elif mode == "all":
            targets.append(DatasetTarget(
                year=year,
                dataset_path=ds_path,
                dataset_structure=_infer_dataset_structure(ds_path),
            ))
        else:
            raise ValueError(f"Unknown allowlist mode: {mode}")

    logger.info(
        "Year %s: %d datasets planned, %d skipped (mode=%s)",
        year, len(targets), len(skipped), mode,
    )
    return targets, skipped


# ---------------------------------------------------------------------------
# Stage 1b: Per-dataset metadata fetch
# ---------------------------------------------------------------------------


def fetch_dataset_metadata(
    target: DatasetTarget,
    client: CensusAPIClient,
    paths: PipelinePaths,
    catalog_entry: Optional[dict],
    *,
    force_refresh: bool = False,
) -> DatasetMetadata:
    """Fetch and cache all metadata files for a single dataset."""
    result = DatasetMetadata(target=target, catalog_entry=catalog_entry)

    # Also persist the catalog entry on disk for provenance
    if catalog_entry is not None:
        catalog_path = paths.raw_dataset_file(target.year, target.dataset_path, "catalog_entry.json")
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with catalog_path.open("w", encoding="utf-8") as f:
            json.dump(catalog_entry, f, indent=2, ensure_ascii=False)

    base_url = f"{client.config.base_url}/{target.year}/{target.dataset_path}"

    for filename, spec in DATASET_METADATA_FILES.items():
        url = f"{base_url}/{filename}"
        cache_path = paths.raw_dataset_file(target.year, target.dataset_path, filename)
        try:
            data = client.fetch_json(url, cache_path=cache_path, force_refresh=force_refresh)
            _assign_metadata_field(result, filename, data)
        except CensusNotFoundError:
            if spec["required"]:
                result.fetch_errors[filename] = "404 Not Found (required file missing)"
                logger.warning(
                    "Required metadata %s missing for %s",
                    filename, target.dataset_path,
                )
            else:
                logger.debug(
                    "Optional metadata %s not present for %s",
                    filename, target.dataset_path,
                )
        except CensusAPIError as e:
            result.fetch_errors[filename] = str(e)
            logger.warning(
                "Error fetching %s for %s: %s",
                filename, target.dataset_path, e,
            )

    return result


def _assign_metadata_field(result: DatasetMetadata, filename: str, data: dict) -> None:
    mapping = {
        "variables.json": "variables",
        "geography.json": "geography",
        "groups.json": "groups",
        "examples.json": "examples",
        "tags.json": "tags",
    }
    field_name = mapping.get(filename)
    if field_name:
        setattr(result, field_name, data)


# ---------------------------------------------------------------------------
# Stage 1c: Per-table partitioning and output
# ---------------------------------------------------------------------------


# ---- Variable classification constants ---------------------------------

# API-mechanics variables — these are Census API query parameters that appear
# in variables.json for documentation purposes only. Not meaningful as data.
API_MECHANICS_VARIABLES = {"for", "in"}

# Known geography identifier variables that appear at the dataset level.
# These identify the geographic unit of a row rather than measuring anything.
# Includes standard FIPS geography component codes (STATE, COUNTY, etc.) that
# Census exposes as alternative geography specifiers alongside the for=/in= API.
# Also includes ANSI-variant names (suffix NS) from the geoinfo dataset.
GEOGRAPHY_IDENTIFIER_VARIABLES = {
    # Standard cross-dataset identifiers
    "NAME",         # Human-readable area name, e.g., "Fulton County, Georgia"
    "GEO_ID",       # Full geographic identifier string
    "ucgid",        # Uniform Census Geography Identifier
    "GEOCOMP",      # Geographic component (urban/rural within a geography)
    "SUMLEVEL",     # Summary level code
    # FIPS geography component codes — appear in ACS, geoinfo, acs/flows, etc.
    "NATION",       # Nation
    "STATE",        # State FIPS code
    "COUNTY",       # County FIPS code
    "TRACT",        # Census tract
    "BLKGRP",       # Block group
    "TBLKGRP",      # Tribal block group
    "TTRACT",       # Tribal tract
    "PLACE",        # Place FIPS code
    "PLACEREM",     # Place remainder
    "COUSUB",       # County subdivision
    "CONCIT",       # Consolidated city
    "AIANHH",       # American Indian Area / Alaska Native Area / Hawaiian Home Land
    "AIHHTL",       # American Indian Area (off-reservation trust land)
    "AIRES",        # American Indian Reservation and off-reservation trust land
    "AIARO",        # American Indian Area / off-reservation
    "ANRC",         # Alaska Native Regional Corporation
    "CBSA",         # Core-Based Statistical Area (metro/micro)
    "CSA",          # Combined Statistical Area
    "METDIV",       # Metropolitan Division
    "NECTA",        # New England City and Town Area
    "NECTADIV",     # NECTA Division
    "CNECTA",       # Combined NECTA
    "CD",           # Congressional District
    "SLDL",         # State Legislative District (Lower Chamber)
    "SLDU",         # State Legislative District (Upper Chamber)
    "SDELM",        # School District (Elementary)
    "SDSEC",        # School District (Secondary)
    "SDUNI",        # School District (Unified)
    "PUMA",         # Public Use Microdata Area
    "UA",           # Urban Area
    "ZCTA",         # ZIP Code Tabulation Area
    "ZCTA5",        # 5-digit ZCTA
    "REGION",       # Census Region
    "DIVISION",     # Census Division
    "PRINCITY",     # Principal City
    "SUBMCD",       # Subminor Civil Division
    "TRISUBREM",    # American Indian Tribal Subdivision/Remainder
    "GTCO",         # FIPS County Code (alternate form in some datasets)
    "ST",           # State code (alternate form)
    # ANSI-variant identifiers (from geoinfo dataset, suffix NS = ANSI)
    "AIANHHNS",     # AIANHH (ANSI)
    "CONCITNS",     # Consolidated City (ANSI)
    "PLACENS",      # Place (ANSI)
    "AITSNS",       # American Indian Tribal Subdivision (ANSI)
    "ANRCNS",       # ANRC (ANSI)
}

# Record-level identifiers — appear in microdata/PUMS datasets where rows
# represent individual housing units or persons rather than geographies.
# Not geography identifiers per se, but serve the same "row identity" role.
RECORD_IDENTIFIER_VARIABLES = {
    "SERIALNO",     # Housing unit / GQ person serial number (ACS PUMS)
    "CONCAT_ID",    # Concatenation ID (geographic join key)
    "RECORD_TYPE",  # Record type flag (microdata datasets)
}

# Known time/period dimension variables
TIME_DIMENSION_VARIABLES = {
    "YEAR", "DATE_", "time", "TIME", "PERIOD", "YYYYMM",
    "H_YEAR",       # Year of survey — Household (CPS)
    "FILEDATE",     # File creation date
}


# ---- Variable-role classification for dataset-level (ungrouped) variables --

ROLE_GEOGRAPHY_IDENTIFIER = "geography_identifier"
ROLE_RECORD_IDENTIFIER = "record_identifier"
ROLE_REQUIRED_PREDICATE = "required_predicate"
ROLE_OPTIONAL_PREDICATE = "optional_predicate"
ROLE_TIME_DIMENSION = "time_dimension"
ROLE_ANOMALY = "anomaly"


def classify_ungrouped_variable(var_id: str, meta: dict) -> str:
    """Classify a variable that has no group (or group='N/A') into a role.

    Decision order (first match wins):
      1. Name-based geography identifier (deterministic)
      2. Name-based record identifier (microdata row keys)
      3. Name-based time dimension (deterministic)
      4. concept == "Geography" (Census explicitly marks these)
      5. predicateOnly / values / validValues metadata signals
      6. Anomaly (no pattern matched — logged at WARNING for review)

    Name-based rules take precedence over metadata-based rules to ensure
    variables like NAME and GEO_ID are always classified correctly even if
    they happen to have other metadata that would match a later rule.
    """
    # 1. Name-based geography identifiers
    if var_id in GEOGRAPHY_IDENTIFIER_VARIABLES:
        return ROLE_GEOGRAPHY_IDENTIFIER

    # 2. Name-based record identifiers (microdata row keys)
    if var_id in RECORD_IDENTIFIER_VARIABLES:
        return ROLE_RECORD_IDENTIFIER

    # 3. Name-based time dimensions
    if var_id in TIME_DIMENSION_VARIABLES:
        return ROLE_TIME_DIMENSION

    # 4. Census explicitly marks geography component variables via concept field.
    #    This catches the full set of FIPS geography codes that appear in ACS,
    #    geoinfo, and acs/flows datasets — variables like STATE, COUNTY, REGION,
    #    DIVISION, CBSA, etc. that have concept="Geography" but no predicateOnly.
    if meta.get("concept", "").strip().lower() == "geography":
        return ROLE_GEOGRAPHY_IDENTIFIER

    # 5. Predicate/filter signals from Census metadata
    is_predicate_only = meta.get("predicateOnly") is True
    is_required = meta.get("required") == "true" or meta.get("required") is True
    has_enum = bool(meta.get("values") or meta.get("validValues"))

    if is_predicate_only or has_enum:
        return ROLE_REQUIRED_PREDICATE if is_required else ROLE_OPTIONAL_PREDICATE

    # 6. No pattern matched
    return ROLE_ANOMALY


def _parse_group_field(group_value: str) -> list[str]:
    """Parse a variable's 'group' field into a list of table IDs.

    Most variables have a single table ID (e.g., "B01001"), but some variables
    — typically shared metadata variables like GEO_ID — list multiple table
    IDs in a single comma-separated string (e.g., "CP02,CP03,CP04,CP05,CP02PR").
    We split these so the variable is assigned to each referenced table.

    Returns an empty list for empty values or the 'N/A' sentinel.
    """
    if not group_value or group_value == "N/A":
        return []
    parts = [p.strip() for p in group_value.split(",")]
    return [p for p in parts if p]


def partition_variables_by_table(
    variables_payload: dict,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Partition variables from variables.json into table-level and dataset-level.

    Census variables fall into three categories:
      1. Table-level data variables — these have a 'group' field naming one
         or more tables. Split by group (handling CSV group values).
      2. Dataset-level variables — these have no group (or group='N/A').
         They include geography identifiers (NAME, GEO_ID), filter predicates
         (AGEGROUP, HISP, SEX in PEP), and time dimensions (YEAR, DATE_).
      3. API-mechanics variables (for, in) — these document the Census API's
         own query parameters and are not meaningful as data. Discarded here.

    Returns:
        (by_table, dataset_level_variables) where
            by_table[table_id][var_id] = variable_metadata
            dataset_level_variables[var_id] = {**variable_metadata, "_role": role}
    """
    variables = variables_payload.get("variables", {})
    by_table: dict[str, dict[str, dict]] = defaultdict(dict)
    dataset_level: dict[str, dict] = {}

    for var_id, meta in variables.items():
        if var_id in API_MECHANICS_VARIABLES:
            continue

        group_value = meta.get("group", "")
        table_ids = _parse_group_field(group_value)

        if not table_ids:
            # Dataset-level variable — classify by role
            role = classify_ungrouped_variable(var_id, meta)
            dataset_level[var_id] = {**meta, "_role": role}
            continue

        # Assign to each referenced table
        for table_id in table_ids:
            by_table[table_id][var_id] = meta

    return dict(by_table), dataset_level


def summarize_dataset_level_variables(
    dataset_level: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """Group dataset-level variables by their classified role.

    Returns a dict keyed by role with per-role dicts of {var_id: metadata}.
    """
    by_role: dict[str, dict[str, dict]] = defaultdict(dict)
    for var_id, meta in dataset_level.items():
        role = meta.get("_role", ROLE_ANOMALY)
        by_role[role][var_id] = meta
    return dict(by_role)


# Valid dataset_structure values — used for validation and documentation
DATASET_STRUCTURES = {
    "grouped":   "Standard Census structure — variables belong to named tables",
    "flat":      "No table grouping — all variables are dataset-level (e.g., pdb)",
    "microdata": "Person/household records with survey weights (e.g., CPS, PUMS)",
}


def extract_geography_levels(geography_payload: Optional[dict]) -> list[dict]:
    """Extract the list of supported geography levels from geography.json.

    The geography.json has the shape:
        {"fips": [{"name": "state", "geoLevelDisplay": "040", ...}, ...],
         "default": [...]}

    Returns the fips array verbatim, or [] if geography couldn't be fetched.
    """
    if not geography_payload:
        return []
    return geography_payload.get("fips", [])


def build_table_metadata(
    *,
    target: DatasetTarget,
    table_id: str,
    table_variables: dict[str, dict],
    groups_payload: Optional[dict],
    geography_levels: list[dict],
    catalog_entry: Optional[dict],
    collected_at: str,
) -> dict:
    """Build the stage1_metadata.json content for a single table."""
    # Find the table's entry in groups.json
    group_entry = None
    if groups_payload:
        for g in groups_payload.get("groups", []):
            if g.get("name") == table_id:
                group_entry = g
                break

    # Release metadata — best-effort extraction from catalog entry
    release_meta = {}
    if catalog_entry:
        release_meta = {
            "modified": catalog_entry.get("modified"),
            "temporal": catalog_entry.get("temporal"),
            "spatial": catalog_entry.get("spatial"),
            "documentation_url": catalog_entry.get("c_documentationLink"),
            "contact_email": (catalog_entry.get("contactPoint") or {}).get("hasEmail"),
            "publisher": (catalog_entry.get("publisher") or {}).get("name"),
            "license": catalog_entry.get("license"),
            "identifier": catalog_entry.get("identifier"),
        }

    return {
        "schema_version": "1.0.0",
        "stage": STAGE_NAME,
        "collected_at": collected_at,
        "source": "census",
        "year": target.year,
        "dataset_path": target.dataset_path,
        "table_id": table_id,
        "title": (group_entry or {}).get("description") or (group_entry or {}).get("name"),
        "universe": (group_entry or {}).get("universe"),
        "variable_count_total": len(table_variables),
        "variable_count_estimate": sum(1 for v in table_variables if v.endswith("E")),
        "variable_count_moe": sum(1 for v in table_variables if v.endswith("M")),
        "variable_count_annotation": sum(
            1 for v in table_variables if v.endswith(("EA", "MA", "NA"))
        ),
        "variables": table_variables,
        "geography_levels": geography_levels,
        "release": release_meta,
    }


def build_dataset_document_stub(
    *,
    dataset_meta: DatasetMetadata,
    table_ids: list[str],
    dataset_level_variables: dict[str, dict],
    collected_at: str,
) -> dict:
    """Build a dataset-level document with top-level metadata.

    This is a stub — later stages enrich it further. It's written to
    processed/<dataset>/dataset_document.json and contains the
    DCAT-sourced fields, our derived structural info, and any variables
    that exist at the dataset level rather than being associated with a
    specific table (geography identifiers, required filter predicates, etc.).
    """
    target = dataset_meta.target
    ce = dataset_meta.catalog_entry or {}
    geo_levels = extract_geography_levels(dataset_meta.geography)

    # Group dataset-level variables by their classified role
    variables_by_role = summarize_dataset_level_variables(dataset_level_variables)

    # Identify required predicates — these impose query requirements
    required_predicates = list(variables_by_role.get(ROLE_REQUIRED_PREDICATE, {}).keys())
    has_required_predicates = bool(required_predicates)

    # dataset_structure comes from the target (set by allowlist or inference)
    dataset_structure = dataset_meta.target.dataset_structure

    return {
        "schema_version": "1.0.0",
        "stage": STAGE_NAME,
        "collected_at": collected_at,
        "source": "census",
        "year": target.year,
        "dataset_path": target.dataset_path,
        "dataset_key": target.key,
        "title": ce.get("title"),
        "description_official": ce.get("description"),
        "temporal": ce.get("temporal"),
        "spatial": ce.get("spatial"),
        "is_aggregate": ce.get("c_isAggregate"),
        "is_cube": ce.get("c_isCube"),
        "is_available": ce.get("c_isAvailable"),
        "publisher": (ce.get("publisher") or {}).get("name"),
        "contact_email": (ce.get("contactPoint") or {}).get("hasEmail"),
        "documentation_url": ce.get("c_documentationLink"),
        "license": ce.get("license"),
        "identifier": ce.get("identifier"),
        "modified": ce.get("modified"),
        "dataset_structure": dataset_structure,
        "geography_levels": geo_levels,
        "geography_level_count": len(geo_levels),
        "table_count": len(table_ids),
        "table_ids": sorted(table_ids),
        "dataset_level_variables": {
            "geography_identifiers": variables_by_role.get(ROLE_GEOGRAPHY_IDENTIFIER, {}),
            "record_identifiers": variables_by_role.get(ROLE_RECORD_IDENTIFIER, {}),
            "required_predicates": variables_by_role.get(ROLE_REQUIRED_PREDICATE, {}),
            "optional_predicates": variables_by_role.get(ROLE_OPTIONAL_PREDICATE, {}),
            "time_dimensions": variables_by_role.get(ROLE_TIME_DIMENSION, {}),
            "anomalies": variables_by_role.get(ROLE_ANOMALY, {}),
        },
        "query_requirements": {
            "has_required_predicates": has_required_predicates,
            "required_predicates": required_predicates,
            "dataset_structure": dataset_structure,
            "is_microdata": dataset_structure == "microdata",
            "is_flat": dataset_structure == "flat",
        },
        "fetch_errors": dataset_meta.fetch_errors,
    }


def emit_stage1_outputs(
    dataset_meta: DatasetMetadata,
    paths: PipelinePaths,
    state: StateTracker,
) -> tuple[int, int]:
    """Write per-table stage1_metadata.json files and the dataset document.

    Returns (tables_written, tables_failed).
    """
    target = dataset_meta.target
    collected_at = datetime.now(timezone.utc).isoformat()

    if dataset_meta.variables is None:
        raise ValueError(
            f"Cannot emit Stage 1 outputs for {target.key}: variables.json was not fetched"
        )

    # Partition variables into table-level and dataset-level
    by_table, dataset_level_vars = partition_variables_by_table(dataset_meta.variables)
    geography_levels = extract_geography_levels(dataset_meta.geography)

    # Log classification summary
    variables_by_role = summarize_dataset_level_variables(dataset_level_vars)
    if dataset_level_vars:
        logger.info(
            "%s: %d dataset-level variables — %s",
            target.key,
            len(dataset_level_vars),
            ", ".join(
                f"{role}={len(vars_)}"
                for role, vars_ in variables_by_role.items()
            ),
        )

    # Surface anomalies prominently — these are variables without a group
    # that don't match any known pattern and warrant operator review
    anomalies = variables_by_role.get(ROLE_ANOMALY, {})
    if anomalies:
        logger.warning(
            "%s: %d anomalous variables (no group and no matching classification "
            "pattern) — review needed: %s",
            target.key,
            len(anomalies),
            sorted(anomalies.keys()),
        )

    # Emit per-table files
    tables_written = 0
    tables_failed = 0

    for table_id, table_vars in sorted(by_table.items()):
        try:
            table_doc = build_table_metadata(
                target=target,
                table_id=table_id,
                table_variables=table_vars,
                groups_payload=dataset_meta.groups,
                geography_levels=geography_levels,
                catalog_entry=dataset_meta.catalog_entry,
                collected_at=collected_at,
            )
            out_path = paths.processed_table_stage_file(
                target.year, target.dataset_path, table_id, STAGE_NAME,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(out_path, table_doc)
            state.set_status(
                year=target.year,
                dataset_path=target.dataset_path,
                entity_type="table",
                entity_id=table_id,
                stage=STAGE_NAME,
                status=Status.SUCCESS,
                metadata={"variable_count": len(table_vars)},
            )
            tables_written += 1
        except Exception as e:
            tables_failed += 1
            logger.exception("Failed to emit Stage 1 output for table %s::%s",
                             target.key, table_id)
            state.set_status(
                year=target.year,
                dataset_path=target.dataset_path,
                entity_type="table",
                entity_id=table_id,
                stage=STAGE_NAME,
                status=Status.FAILED_OTHER,
                error_message=str(e),
            )
            state.log_error(
                year=target.year,
                dataset_path=target.dataset_path,
                entity_id=table_id,
                stage=STAGE_NAME,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )

    # Emit dataset-level document
    dataset_doc = build_dataset_document_stub(
        dataset_meta=dataset_meta,
        table_ids=list(by_table.keys()),
        dataset_level_variables=dataset_level_vars,
        collected_at=collected_at,
    )
    dataset_doc_path = paths.processed_dataset_document(target.year, target.dataset_path)
    dataset_doc_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(dataset_doc_path, dataset_doc)

    return tables_written, tables_failed


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write a JSON file atomically via temp file + rename."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process_one_dataset(
    target: DatasetTarget,
    catalog_entry: Optional[dict],
    client: CensusAPIClient,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    force_refresh: bool = False,
) -> dict:
    """Process a single dataset end-to-end through Stage 1."""
    logger.info("Processing dataset %s", target.key)

    state.set_status(
        year=target.year,
        dataset_path=target.dataset_path,
        entity_type="dataset",
        entity_id="",
        stage=STAGE_NAME,
        status=Status.IN_PROGRESS,
    )

    try:
        dataset_meta = fetch_dataset_metadata(
            target, client, paths, catalog_entry, force_refresh=force_refresh,
        )

        # Fail the dataset if any required file is missing
        if dataset_meta.variables is None:
            raise CensusPermanentError(
                f"variables.json missing for {target.key}; cannot continue. "
                f"Fetch errors: {dataset_meta.fetch_errors}"
            )

        tables_written, tables_failed = emit_stage1_outputs(dataset_meta, paths, state)

        result_metadata = {
            "tables_written": tables_written,
            "tables_failed": tables_failed,
            "has_geography": dataset_meta.geography is not None,
            "has_groups": dataset_meta.groups is not None,
            "fetch_errors": dataset_meta.fetch_errors,
        }

        state.set_status(
            year=target.year,
            dataset_path=target.dataset_path,
            entity_type="dataset",
            entity_id="",
            stage=STAGE_NAME,
            status=Status.SUCCESS,
            metadata=result_metadata,
        )
        logger.info(
            "Completed dataset %s: %d tables written, %d failed",
            target.key, tables_written, tables_failed,
        )
        return {"target": target, "status": "success", **result_metadata}

    except Exception as e:
        logger.exception("Failed to process dataset %s", target.key)
        state.set_status(
            year=target.year,
            dataset_path=target.dataset_path,
            entity_type="dataset",
            entity_id="",
            stage=STAGE_NAME,
            status=Status.FAILED_API_ERROR if isinstance(e, CensusAPIError) else Status.FAILED_OTHER,
            error_message=str(e),
        )
        state.log_error(
            year=target.year,
            dataset_path=target.dataset_path,
            entity_id="",
            stage=STAGE_NAME,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc(),
        )
        return {"target": target, "status": "failed", "error": str(e)}


def run_stage1(
    years: Iterable[str],
    allowlist_config: dict,
    client: CensusAPIClient,
    paths: PipelinePaths,
    state: StateTracker,
    *,
    max_concurrent_datasets: int = 4,
    force_refresh: bool = False,
    skip_if_complete: bool = True,
) -> dict:
    """Run Stage 1 across all years and datasets in the allowlist.

    Args:
        years: Years to process.
        allowlist_config: Loaded datasets_allowlist.yaml contents.
        client: Census API client.
        paths: Pipeline path resolver.
        state: State tracker.
        max_concurrent_datasets: How many datasets to fetch concurrently.
        force_refresh: If True, bypass raw/ cache and re-fetch.
        skip_if_complete: If True, skip datasets already marked SUCCESS.

    Returns:
        Summary dict with counts and per-dataset results.
    """
    results: list[dict] = []
    skipped_datasets: list[dict] = []

    for year in years:
        year = str(year)

        # Stage 1a: discover and filter
        catalog_entries = discover_datasets(year, client, paths, force_refresh=force_refresh)
        targets, skipped = plan_dataset_targets(year, catalog_entries, allowlist_config)
        skipped_datasets.extend(skipped)

        # Build a lookup of catalog entries by dataset path for passing through
        catalog_by_path: dict[str, dict] = {}
        for entry in catalog_entries:
            ds_path = _extract_dataset_path(entry)
            if ds_path:
                catalog_by_path[ds_path] = entry

        # Filter out completed datasets if requested
        work_items = []
        for t in targets:
            if skip_if_complete and state.is_complete(
                year=t.year, dataset_path=t.dataset_path,
                entity_type="dataset", entity_id="", stage=STAGE_NAME,
            ):
                logger.info("Skipping %s (already complete)", t.key)
                continue
            work_items.append(t)

        if not work_items:
            logger.info("Year %s: nothing to do", year)
            continue

        # Stage 1b+1c: fetch and partition, concurrently
        with ThreadPoolExecutor(max_workers=max_concurrent_datasets) as pool:
            future_to_target = {
                pool.submit(
                    process_one_dataset,
                    target,
                    catalog_by_path.get(target.dataset_path),
                    client,
                    paths,
                    state,
                    force_refresh=force_refresh,
                ): target
                for target in work_items
            }
            for future in as_completed(future_to_target):
                results.append(future.result())

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_tables = sum(r.get("tables_written", 0) for r in results if r["status"] == "success")

    summary = {
        "datasets_processed": len(results),
        "datasets_succeeded": successful,
        "datasets_failed": failed,
        "datasets_skipped": len(skipped_datasets),
        "tables_written": total_tables,
        "results": results,
    }
    logger.info(
        "Stage 1 summary: %d datasets succeeded, %d failed, %d tables written",
        successful, failed, total_tables,
    )
    return summary
