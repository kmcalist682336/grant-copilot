"""Stage 8 runner — orchestrates data ingestion across all tables.

Usage:
    python -m scripts.enrichment.run_stage8

    # Only fetch specific geo levels
    python -m scripts.enrichment.run_stage8 --geo-levels tract county

    # Only process specific datasets
    python -m scripts.enrichment.run_stage8 --dataset acs/acs5

    # Only process specific year
    python -m scripts.enrichment.run_stage8 --year 2022

    # Force re-fetch even if already complete
    python -m scripts.enrichment.run_stage8 --no-skip-complete

    # Preflight — process first N tables only
    python -m scripts.enrichment.run_stage8 --preflight --preflight-count 10

    # Fetch tract+county data for all default peer metros
    python -m scripts.enrichment.run_stage8 --peer-metros default

    # Fetch specific peer metros by CBSA code
    python -m scripts.enrichment.run_stage8 --peer-metros 16740 34980 26420

    # Full run including all peer metros
    python -m scripts.enrichment.run_stage8 --geo-levels tract county state --peer-metros default
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from scripts.ingestion.stage8_values import (
    TARGET_GEO_LEVELS,
    DEFAULT_PEER_METROS,
    FetchResult,
    GeoLevelSpec,
    CoverageReport,
    resolve_supported_geo_levels,
    build_peer_metro_specs,
    unpivot_response,
    validate_variable_coverage,
    load_georgia_zctas,
    write_jsonl,
    append_jsonl,
    jsonl_shard_path,
)
from scripts.utils.census_api import CensusAPIConfig, CensusAPIClient
from scripts.utils.config import (
    init_logging, load_pipeline_config, make_run_id, make_run_manifest,
)
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)
STAGE_NAME = "stage8_values"


# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    year: str
    dataset_path: str
    table_id: str
    geo_specs: list[GeoLevelSpec]       # Georgia geo levels to fetch
    peer_specs: list[GeoLevelSpec]      # peer metro specs to fetch


# ---------------------------------------------------------------------------
# Work discovery
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_work_items(
    paths: PipelinePaths,
    state: StateTracker,
    geo_level_filter: Optional[list[str]],
    dataset_filter: Optional[str],
    year_filter: Optional[str],
    skip_if_complete: bool,
    peer_cbsa_codes: Optional[list[str]] = None,
    exclude_geo_levels: Optional[set[str]] = None,
    skip_failed: bool = False,
    msa_cbsa_ids: Optional[list[str]] = None,
) -> list[WorkItem]:
    """Find all (year, dataset, table) combinations that need Stage 8 processing."""
    # Get all Stage 7 successes
    s7_records = state.records_for_stage(
        stage="stage7_variables",
        status=Status.SUCCESS,
    )
    # Filter to table entity type
    s7_records = [r for r in s7_records if r.entity_type == "table"]
    logger.info("Stage 7 successes: %d tables", len(s7_records))

    work_items = []
    skipped_complete = 0
    skipped_no_geo = 0

    for record in s7_records:
        year = record.year
        dataset_path = record.dataset_path
        table_id = record.entity_id

        # Apply filters
        if dataset_filter and dataset_path != dataset_filter:
            continue
        if year_filter and str(year) != str(year_filter):
            continue

        # Load Stage 1 metadata to get geography_levels
        stage1_path = paths.processed_table_stage_file(
            year, dataset_path, table_id, "stage1_metadata"
        )
        if not stage1_path.exists():
            continue

        try:
            stage1 = json.loads(stage1_path.read_text())
        except Exception as e:
            logger.warning("Could not read stage1 for %s/%s/%s: %s",
                          year, dataset_path, table_id, e)
            continue

        geography_levels = stage1.get("geography_levels", [])
        all_specs = resolve_supported_geo_levels(
            geography_levels, msa_cbsa_ids=msa_cbsa_ids,
        )

        if geo_level_filter:
            all_specs = [s for s in all_specs if s.geo_level in geo_level_filter]

        if exclude_geo_levels:
            all_specs = [s for s in all_specs if s.geo_level not in exclude_geo_levels]

        if not all_specs:
            skipped_no_geo += 1
            continue

        # Filter to specs not yet completed (and optionally not yet failed).
        # skip_if_complete: SUCCESS is skipped (default resume behavior).
        # skip_failed:     also skip any terminal FAILED_* state, so
        #                   restarting the run doesn't re-retry calls we
        #                   already know returned nothing useful.
        if skip_if_complete:
            pending_specs = []
            for spec in all_specs:
                entity_id = f"{table_id}::{spec.geo_level}"
                record = state.get_status(
                    year=str(year),
                    dataset_path=dataset_path,
                    entity_type="table_geo",
                    entity_id=entity_id,
                    stage=STAGE_NAME,
                )
                if record is not None:
                    if record.status == Status.SUCCESS:
                        continue
                    if skip_failed and record.status in (
                        Status.FAILED_NO_DATA,
                        Status.FAILED_API_ERROR,
                        Status.FAILED_OTHER,
                    ):
                        continue
                pending_specs.append(spec)
            if not pending_specs:
                skipped_complete += 1
                continue
            specs_to_fetch = pending_specs
        else:
            specs_to_fetch = all_specs

        work_items.append(WorkItem(
            year=str(year),
            dataset_path=dataset_path,
            table_id=table_id,
            geo_specs=specs_to_fetch,
            peer_specs=build_peer_metro_specs(
                cbsa_codes=peer_cbsa_codes or [],
                geo_levels=["tract", "county"],
                geography_levels=geography_levels,
            ) if peer_cbsa_codes else [],
        ))

    logger.info(
        "Work items: %d tables, skipped %d complete, %d no-geo",
        len(work_items), skipped_complete, skipped_no_geo,
    )
    return work_items


# ---------------------------------------------------------------------------
# Fetch one (table, geo_level)
# ---------------------------------------------------------------------------

async def _fetch_one_geo_level(
    client: CensusAPIClient,
    item: WorkItem,
    spec: GeoLevelSpec,
    raw_values_dir: Path,
) -> tuple[list, Optional[str]]:
    """Fetch raw Census API response for one (table, geo_level).

    Uses asyncio.to_thread to run the synchronous Census client
    without blocking the event loop.

    Returns:
        (raw_response, error_message)
        raw_response is [] on failure.
    """
    # Check disk cache first — avoids re-fetching on resume
    cache_path = raw_values_dir / f"{spec.cache_key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached and len(cached) > 1:  # header + at least one data row
                logger.debug("Cache hit: %s/%s/%s/%s",
                            item.year, item.dataset_path,
                            item.table_id, spec.geo_level)
                return cached, None
        except Exception:
            pass  # corrupt cache — re-fetch

    # Build full Census API URL
    base = f"https://api.census.gov/data/{item.year}/{item.dataset_path}"
    params = f"get=group({item.table_id})&for={spec.for_clause}"
    if spec.in_clause:
        params += f"&in={spec.in_clause}"
    url = f"{base}?{params}"

    label = (f"{item.year}/{item.dataset_path}/{item.table_id}"
             f"  [{spec.cache_key}]")

    def _do_fetch() -> tuple[list, Optional[str]]:
        from scripts.utils.census_api import (
            CensusNotFoundError, CensusPermanentError,
            CensusTransientError, CensusRateLimitError,
            CensusConnectionError,
        )
        try:
            response = client.fetch_json(url, cache_path=cache_path)
            return response, None
        except CensusRateLimitError as e:
            # Print throttle errors immediately — always visible
            print(
                f"\n  ⚡ THROTTLED  {label}\n"
                f"     Census API rate limit hit — will retry\n"
                f"     {e}",
                flush=True,
            )
            return [], f"rate_limit: {e}"
        except CensusConnectionError as e:
            # API unreachable (DNS/TCP/TLS). Fail-fast path already took
            # care of its own short retry; mark as connection error so
            # restarts can distinguish "network was down" from "table has
            # no data at this geography."
            return [], f"connection_error: {e}"
        except CensusNotFoundError:
            return [], "404: data not available at this geography level"
        except CensusPermanentError as e:
            return [], f"permanent_error: {e}"
        except CensusTransientError as e:
            return [], f"transient_error: {e}"
        except Exception as e:
            return [], f"unexpected: {e}"

    # Run the fetch with a watchdog that prints every 5s if still waiting
    async def _watchdog(interval: float = 5.0) -> None:
        elapsed = 0.0
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            print(
                f"  ... {label}  still fetching  ({elapsed:.0f}s elapsed)",
                flush=True,
            )

    fetch_task = asyncio.create_task(asyncio.to_thread(_do_fetch))
    watchdog_task = asyncio.create_task(_watchdog())

    try:
        result = await fetch_task
    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    return result


# ---------------------------------------------------------------------------
# Process one table
# ---------------------------------------------------------------------------

async def process_one_table(
    item: WorkItem,
    client: CensusAPIClient,
    paths: PipelinePaths,
    state: StateTracker,
    ga_zcta_set: set[str],
    curated_dir: Path,
    geo_semaphore: asyncio.Semaphore,
) -> dict:
    """Fetch and unpivot data for all geo levels of one table."""
    raw_values_dir = paths.raw_table_values_dir(
        item.year, item.dataset_path, item.table_id
    )
    raw_values_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 variable inventory once — used for coverage validation
    stage1_path = paths.processed_table_stage_file(
        item.year, item.dataset_path, item.table_id, "stage1_metadata"
    )
    stage1_variables = []
    n_expected_vars = 0
    if stage1_path.exists():
        try:
            stage1 = json.loads(stage1_path.read_text())
            stage1_variables = stage1.get("variables", [])
            # Count estimate variables for the progress print. ACS ends in
            # 'E', decennial ends in 'N'; both annotations 'EA'/'NA' skipped.
            from scripts.ingestion.stage8_values import _is_estimate_suffix
            n_expected_vars = sum(
                1 for v in stage1_variables
                if (
                    isinstance(v, str)
                    and _is_estimate_suffix(v.upper())
                ) or (
                    isinstance(v, dict)
                    and _is_estimate_suffix(v.get("var_id", "").upper())
                )
            )
        except Exception:
            pass

    all_specs = item.geo_specs + item.peer_specs

    # Progress print — one line per table so you know what's starting
    print(
        f"  --> {item.year}/{item.dataset_path}/{item.table_id}"
        f"  {n_expected_vars} vars"
        f"  {len(all_specs)} geo levels",
        flush=True,
    )

    summary = {
        "table_id": item.table_id,
        "geo_results": {},
        "coverage": {},
    }

    # Process all geo levels concurrently within this table
    # Each spec is an independent API call — no ordering dependency
    async def _process_one_spec(spec: GeoLevelSpec) -> dict:
        """Fetch, validate, unpivot, and write one (table, geo_level)."""
        if spec.peer_cbsa_code and spec.peer_state_fips:
            entity_id = (f"{item.table_id}::{spec.geo_level}"
                        f"::peer_{spec.peer_cbsa_code}_{spec.peer_state_fips}")
        else:
            entity_id = f"{item.table_id}::{spec.geo_level}"

        state.set_status(
            year=item.year,
            dataset_path=item.dataset_path,
            entity_type="table_geo",
            entity_id=entity_id,
            stage=STAGE_NAME,
            status=Status.IN_PROGRESS,
        )

        try:
            raw_response, error = await _fetch_one_geo_level(
                client, item, spec, raw_values_dir,
            )

            if error or not raw_response:
                fail_status = (
                    Status.FAILED_NO_DATA
                    if error and ("404" in error or "no data" in error.lower())
                    else Status.FAILED_API_ERROR
                )
                state.set_status(
                    year=item.year, dataset_path=item.dataset_path,
                    entity_type="table_geo", entity_id=entity_id,
                    stage=STAGE_NAME, status=fail_status,
                    error_message=error or "empty response",
                )
                return {"spec": spec, "status": fail_status.value, "rows": 0}

            # Validate variable coverage
            coverage = validate_variable_coverage(
                raw_response=raw_response,
                table_id=item.table_id,
                stage1_variables=stage1_variables,
                geo_level=spec.geo_level,
                year=int(item.year),
                dataset_path=item.dataset_path,
            )

            if not coverage.is_complete:
                print(
                    f"\n{'='*60}\n"
                    f"  ⚠  VARIABLE MISMATCH ALERT\n"
                    f"{'='*60}\n"
                    f"  Table:    {item.year}/{item.dataset_path}/{item.table_id}\n"
                    f"  Geo:      {spec.geo_level}\n"
                    f"  Expected: {coverage.expected_count} variables\n"
                    f"  Actual:   {coverage.actual_count} variables\n"
                    f"  Missing:  {len(coverage.missing_vars)} "
                    f"({coverage.coverage_pct:.1f}% coverage)\n"
                    f"  First missing: {coverage.missing_vars[:5]}\n"
                    f"{'='*60}\n",
                    flush=True,
                )
            else:
                peer_tag = (f"  [peer:{spec.peer_metro_name[:20]}]"
                           if spec.peer_cbsa_code else "")
                print(
                    f"  OK  {item.year}/{item.dataset_path}/{item.table_id}"
                    f"  [{spec.geo_level}]{peer_tag}  "
                    f"{coverage.actual_count}/{coverage.expected_count} vars  100%",
                    flush=True,
                )

            # Unpivot
            zcta_filter = ga_zcta_set if spec.geo_level == "zcta" else None
            long_rows = unpivot_response(
                raw_response=raw_response,
                year=int(item.year),
                dataset_path=item.dataset_path,
                table_id=item.table_id,
                geo_level=spec.geo_level,
                ga_zcta_set=zcta_filter,
                peer_cbsa_code=spec.peer_cbsa_code,
                stage1_variables=stage1_variables,
            )

            if not long_rows:
                state.set_status(
                    year=item.year, dataset_path=item.dataset_path,
                    entity_type="table_geo", entity_id=entity_id,
                    stage=STAGE_NAME, status=Status.FAILED_NO_DATA,
                    error_message="unpivot produced 0 rows",
                )
                return {"spec": spec, "status": "failed_no_data",
                        "rows": 0, "coverage": coverage.to_dict()}

            # Append to shard JSONL
            shard_path = jsonl_shard_path(
                curated_dir, int(item.year),
                item.dataset_path, spec.geo_level,
            )
            append_jsonl(long_rows, shard_path)

            state.set_status(
                year=item.year, dataset_path=item.dataset_path,
                entity_type="table_geo", entity_id=entity_id,
                stage=STAGE_NAME, status=Status.SUCCESS,
                metadata={
                    "row_count": len(long_rows),
                    "coverage_pct": coverage.coverage_pct,
                    "missing_var_count": len(coverage.missing_vars),
                    "missing_vars": coverage.missing_vars[:20],
                },
            )
            return {"spec": spec, "status": "success",
                    "rows": len(long_rows), "coverage": coverage.to_dict()}

        except Exception as e:
            logger.error(
                "Unexpected error %s/%s/%s/%s: %s",
                item.year, item.dataset_path, item.table_id,
                spec.geo_level, e, exc_info=True,
            )
            state.set_status(
                year=item.year, dataset_path=item.dataset_path,
                entity_type="table_geo", entity_id=entity_id,
                stage=STAGE_NAME, status=Status.FAILED_OTHER,
                error_message=str(e),
            )
            return {"spec": spec, "status": "failed_other", "rows": 0}

    # Launch geo-level fetches for this table, bounded by geo_semaphore.
    # Without this bound, a table with 9 geo levels holds 9 parsed JSON
    # responses in memory at once; compounded across 50 tables this drove
    # peak RAM to ~50GB in preflight.
    async def _bounded_spec(spec: GeoLevelSpec) -> dict:
        async with geo_semaphore:
            return await _process_one_spec(spec)

    spec_results = await asyncio.gather(
        *[_bounded_spec(spec) for spec in all_specs],
        return_exceptions=True,
    )

    for result in spec_results:
        if isinstance(result, Exception):
            logger.error("Spec task exception: %s", result)
            continue
        spec = result["spec"]
        summary["geo_results"][spec.cache_key] = {
            "status": result["status"],
            "rows": result.get("rows", 0),
        }
        if "coverage" in result:
            summary["coverage"][spec.cache_key] = result["coverage"]

    return summary


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_stage8(
    work_items: list[WorkItem],
    client: CensusAPIClient,
    paths: PipelinePaths,
    state: StateTracker,
    ga_zcta_set: set[str],
    max_concurrent: int = 50,
    max_geo_concurrent: int = 3,
) -> dict:
    """Run Stage 8 across all work items with bounded concurrency.

    The outer semaphore bounds how many tables are active at once.
    The inner (geo) semaphore bounds how many geo-level API calls run
    concurrently *within* a single table. Peak in-flight HTTP requests
    ≈ max_concurrent × max_geo_concurrent — this is the dominant driver
    of memory, since parsed JSON responses can be tens of MB each.
    """
    curated_dir = paths.curated_dir

    total_ok = 0
    total_failed = 0
    total_rows = 0
    geo_level_rows: dict[str, int] = {}

    semaphore = asyncio.Semaphore(max_concurrent)
    geo_semaphore = asyncio.Semaphore(max_geo_concurrent)

    async def _bounded(item: WorkItem) -> dict:
        async with semaphore:
            return await process_one_table(
                item, client, paths, state, ga_zcta_set, curated_dir,
                geo_semaphore,
            )

    tasks = [_bounded(item) for item in work_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Table task failed: %s", result)
            total_failed += 1
            continue

        table_ok = any(
            v["status"] == "success"
            for v in result["geo_results"].values()
        )
        if table_ok:
            total_ok += 1
        else:
            total_failed += 1

        for geo_level, geo_result in result["geo_results"].items():
            rows = geo_result.get("rows", 0)
            total_rows += rows
            geo_level_rows[geo_level] = geo_level_rows.get(geo_level, 0) + rows

        # Progress logging every 500 tables
        if (i + 1) % 500 == 0:
            logger.info(
                "Progress: %d/%d tables, %d rows so far",
                i + 1, len(work_items), total_rows,
            )

    return {
        "tables_ok": total_ok,
        "tables_failed": total_failed,
        "total_rows": total_rows,
        "rows_by_geo_level": geo_level_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 8: Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Peer metros available (use CBSA codes with --peer-metros):
{"".join(f"  {code}  {name}" + chr(10) for code, (name, _) in sorted(DEFAULT_PEER_METROS.items()))}
Example:
  # Fetch default peer set (all 15 metros)
  python -m scripts.enrichment.run_stage8 --peer-metros default

  # Fetch specific metros
  python -m scripts.enrichment.run_stage8 --peer-metros 16740 34980 26420
        """,
    )
    parser.add_argument("--dataset", default=None,
                        help="Only process this dataset path (e.g. acs/acs5)")
    parser.add_argument("--year", default=None,
                        help="Only process this year")
    parser.add_argument("--geo-levels", nargs="+",
                        choices=TARGET_GEO_LEVELS, default=None,
                        help="Only fetch these geo levels")
    parser.add_argument("--no-zcta", action="store_true",
                        help="Skip ZCTA-level fetches. ZCTA responses are always "
                             "national-scope (Census API does not support state-filtered "
                             "ZCTA queries post-2019) and are genuinely slow/large, "
                             "often dominating the wall-clock of a Stage 8 run.")
    parser.add_argument("--no-msa", action="store_true",
                        help="Skip MSA-level fetches. Useful when running Stage 8 "
                             "in parallel with filter_msa_shards cleanup, or when "
                             "MSA work is being handled in a separate targeted run.")
    parser.add_argument(
        "--peer-metros", nargs="+", default=None, metavar="CBSA_CODE",
        help=(
            "Fetch tract+county data for peer metros. "
            "Use 'default' to fetch all 15 default peer metros, "
            "or pass specific CBSA codes."
        ),
    )
    parser.add_argument(
        "--msa-top-n", type=int, default=100,
        help="Restrict MSA geo-level fetches to the top-N CBSAs by "
             "population (plus all GA-containing CBSAs). Requires "
             "admin_geographies.population populated via "
             "scripts.geography.populate_cbsa_population. Set to 0 to "
             "fetch all ~925 CBSAs nationally (old behavior; ~24× more "
             "storage for msa shards).",
    )
    parser.add_argument(
        "--msa-extra-states", nargs="+", default=["13"], metavar="STATE_FIPS",
        help="State FIPS codes whose CBSAs are unconditionally included "
             "in MSA fetches (even if they fall outside top-N). Default: "
             "13 (Georgia). Pass multiple for multi-state scope.",
    )
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max concurrent table fetches (default: 50 with API key, "
                             "8 without). Each table fetches all geo levels in parallel.")
    parser.add_argument("--max-geo-concurrent", type=int, default=3,
                        help="Max concurrent geo-level fetches within a single table "
                             "(default: 3). Peak HTTP in-flight ≈ max_concurrent × "
                             "max_geo_concurrent; each parsed response can be tens of MB, "
                             "so this is the main memory knob. Lower it if RAM is tight.")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="Read timeout in seconds — how long to wait for a "
                             "response body once connected. Overrides "
                             "census_api.timeout_seconds from config.")
    parser.add_argument("--connect-timeout", type=float, default=None,
                        help="Connect timeout in seconds — how long to wait for "
                             "the TCP/TLS handshake before giving up. Short by "
                             "design (default 5s): lets us fail fast when the "
                             "API is down while still giving slow-but-alive "
                             "endpoints the full --request-timeout window.")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max retries per request on transient errors (overrides "
                             "census_api.max_retries from config). Set to 1 for "
                             "fail-fast behavior; failed tables can be retried later "
                             "via a follow-up run (skip-if-complete keeps successes).")
    parser.add_argument("--no-skip-complete", action="store_true",
                        help="Re-fetch even if already marked success")
    parser.add_argument("--skip-failed", action="store_true",
                        help="On restart, also skip any (table, geo) that "
                             "previously hit a terminal FAILED_* state. "
                             "Use this when a prior run has exhausted its "
                             "fail-fast retries and you don't want another "
                             "pass to burn API calls on them.")
    parser.add_argument("--preflight", action="store_true",
                        help="Process first N tables only (for testing)")
    parser.add_argument("--preflight-count", type=int, default=5)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    config = load_pipeline_config(repo_root)

    with (repo_root / "config" / "pipeline.yaml").open() as f:
        pass  # already loaded above

    paths = PipelinePaths.from_config(repo_root, config)
    paths.ensure_dirs()

    run_id = make_run_id()
    init_logging(config)

    logger.info("=" * 60)
    logger.info("STAGE 8 — Data Ingestion")
    logger.info("=" * 60)

    # Size the thread pool to match concurrency needs.
    # Peak concurrent HTTP calls = max_concurrent_tables × max_geo_concurrent.
    # Add headroom for unpivot/file-write work happening off the event loop.
    import concurrent.futures
    import os
    cpu_count = os.cpu_count() or 4
    _max_concurrent = args.max_concurrent or config.get(
        "batching", {}
    ).get("max_concurrent_tables_s8", 50)
    _max_geo_concurrent = args.max_geo_concurrent
    peak_http = _max_concurrent * _max_geo_concurrent
    thread_pool_size = max(peak_http * 2, cpu_count * 4, 64)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_pool_size,
        thread_name_prefix="stage8_fetch",
    )
    asyncio.get_event_loop().set_default_executor(executor)
    logger.info(
        "Thread pool: %d workers (%d CPUs, %d tables × %d geo = %d peak HTTP)",
        thread_pool_size, cpu_count,
        _max_concurrent, _max_geo_concurrent, peak_http,
    )

    # Load Census API config. Size the urllib3 connection pool to match
    # peak in-flight HTTP calls so connections get reused instead of
    # discarded (which would show up as "Connection pool is full" warnings).
    census_cfg = CensusAPIConfig.from_config(config)
    if args.request_timeout is not None:
        census_cfg.timeout_seconds = args.request_timeout
    if args.connect_timeout is not None:
        census_cfg.connect_timeout_seconds = args.connect_timeout
    if args.max_retries is not None:
        census_cfg.max_retries = args.max_retries
    logger.info(
        "Census API: connect=%.0fs, read=%.0fs, max_retries=%d, "
        "max_connection_retries=%d (worst-case slow fetch ≈ %.0fs)",
        census_cfg.connect_timeout_seconds,
        census_cfg.timeout_seconds,
        census_cfg.max_retries,
        census_cfg.max_connection_retries,
        census_cfg.timeout_seconds * (census_cfg.max_retries + 1),
    )
    client = CensusAPIClient(census_cfg, pool_maxsize=max(peak_http + 16, 64))

    state = StateTracker(paths.status_db_path)

    # Load Georgia ZCTA filter set from raw data
    ga_zcta_set = load_georgia_zctas(paths.data_root)
    logger.info("Georgia ZCTA filter: %d ZCTAs", len(ga_zcta_set))

    # Resolve peer metros
    peer_cbsa_codes: Optional[list[str]] = None
    if args.peer_metros:
        if args.peer_metros == ["default"]:
            peer_cbsa_codes = list(DEFAULT_PEER_METROS.keys())
            logger.info(
                "Peer metros: all %d defaults", len(peer_cbsa_codes)
            )
        else:
            # Validate provided CBSA codes
            valid = []
            for code in args.peer_metros:
                if code in DEFAULT_PEER_METROS:
                    valid.append(code)
                else:
                    logger.warning(
                        "Unknown peer metro CBSA code: %s — skipping", code
                    )
            peer_cbsa_codes = valid
            logger.info(
                "Peer metros: %d specified (%s)",
                len(peer_cbsa_codes),
                ", ".join(
                    DEFAULT_PEER_METROS[c][0].split("-")[0]
                    for c in peer_cbsa_codes
                ),
            )

    # Resolve MSA CBSA filter — top-N by population + all in-scope state CBSAs.
    msa_cbsa_ids: Optional[list[str]] = None
    if args.msa_top_n > 0:
        from scripts.geography.schema import open_spatialite
        from scripts.ingestion.stage8_values import select_top_cbsas
        gazetteer_db_path = (
            Path(args.repo_root) if args.repo_root else _repo_root()
        ) / "data" / "geo" / "gazetteer.db"
        if gazetteer_db_path.exists():
            try:
                gdb = open_spatialite(gazetteer_db_path)
                msa_cbsa_ids = select_top_cbsas(
                    gdb,
                    top_n=args.msa_top_n,
                    include_state_fips=args.msa_extra_states,
                )
                gdb.close()
                if msa_cbsa_ids:
                    logger.info(
                        "MSA filter: %d CBSAs (top-%d by pop + "
                        "state-FIPS %s)",
                        len(msa_cbsa_ids), args.msa_top_n,
                        ",".join(args.msa_extra_states),
                    )
                else:
                    logger.warning(
                        "MSA filter inactive: no CBSA population data in "
                        "gazetteer. Run `python -m scripts.geography."
                        "populate_cbsa_population` to enable. Falling "
                        "back to nationwide MSA fetches."
                    )
            except Exception as e:
                logger.warning("Could not build MSA filter: %s", e)
                msa_cbsa_ids = None
        else:
            logger.warning(
                "Gazetteer DB not found at %s — MSA filter disabled",
                gazetteer_db_path,
            )

    # Discover work
    exclude_set: set[str] = set()
    if args.no_zcta:
        exclude_set.add("zcta")
        logger.info("ZCTA fetches disabled (--no-zcta)")
    if args.no_msa:
        exclude_set.add("msa")
        logger.info("MSA fetches disabled (--no-msa)")
    exclude_geo_levels = exclude_set if exclude_set else None
    if args.skip_failed:
        logger.info(
            "Skipping previously-failed (table, geo) pairs (--skip-failed)"
        )
    work_items = _discover_work_items(
        paths=paths,
        state=state,
        geo_level_filter=args.geo_levels,
        dataset_filter=args.dataset,
        year_filter=args.year,
        skip_if_complete=not args.no_skip_complete,
        peer_cbsa_codes=peer_cbsa_codes,
        msa_cbsa_ids=msa_cbsa_ids,
        exclude_geo_levels=exclude_geo_levels,
        skip_failed=args.skip_failed,
    )

    if args.preflight:
        work_items = work_items[:args.preflight_count]
        logger.info("PREFLIGHT: processing %d tables", len(work_items))

    if not work_items:
        logger.warning("No work items — Stage 8 has nothing to do")
        return 0

    logger.info("Total work items: %d tables", len(work_items))
    total_geo_fetches = sum(len(w.geo_specs) for w in work_items)
    total_peer_fetches = sum(len(w.peer_specs) for w in work_items)
    logger.info("Total Georgia geo-level fetches: %d", total_geo_fetches)
    if total_peer_fetches:
        logger.info("Total peer metro fetches: %d", total_peer_fetches)
    logger.info(
        "Grand total fetches: %d", total_geo_fetches + total_peer_fetches
    )

    max_concurrent = _max_concurrent

    t0 = time.time()
    summary = await run_stage8(
        work_items=work_items,
        client=client,
        paths=paths,
        state=state,
        ga_zcta_set=ga_zcta_set,
        max_concurrent=max_concurrent,
        max_geo_concurrent=_max_geo_concurrent,
    )
    elapsed = time.time() - t0

    # Write run manifest
    run_manifest = make_run_manifest(
        run_id=run_id,
        stage=STAGE_NAME,
        pipeline_version=config.get("pipeline_version", "unknown"),
        config_snapshot={"census_api": config.get("census_api", {})},
    )
    run_manifest["summary"] = summary
    run_manifest["elapsed_seconds"] = elapsed
    with paths.run_manifest_path(run_id).open("w") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info("Stage 8 complete (%.1fs)", elapsed)
    logger.info("  Tables: %d ok, %d failed",
                summary["tables_ok"], summary["tables_failed"])
    logger.info("  Total rows: %d", summary["total_rows"])
    logger.info("  Rows by geo level:")
    for geo_level, rows in sorted(summary["rows_by_geo_level"].items()):
        logger.info("    %-30s %d", geo_level, rows)
    logger.info("=" * 60)

    # Coverage summary from state tracker
    conn_path = paths.status_db_path
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(conn_path)
        conn.row_factory = _sqlite3.Row
        coverage_rows = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN json_extract(metadata, '$.missing_var_count') = 0
                         THEN 1 ELSE 0 END) as complete,
                SUM(CASE WHEN json_extract(metadata, '$.missing_var_count') > 0
                         THEN 1 ELSE 0 END) as incomplete,
                AVG(json_extract(metadata, '$.coverage_pct')) as avg_coverage
            FROM stage_status
            WHERE stage = ? AND status = 'success'
        """, (STAGE_NAME,)).fetchone()
        if coverage_rows and coverage_rows["total"]:
            logger.info("Coverage summary:")
            logger.info("  Complete (100%% vars):  %d", coverage_rows["complete"])
            logger.info("  Incomplete (<100%%):    %d", coverage_rows["incomplete"])
            logger.info("  Avg coverage:          %.2f%%", coverage_rows["avg_coverage"] or 0)

            # Show worst offenders
            worst = conn.execute("""
                SELECT entity_id,
                       json_extract(metadata, '$.coverage_pct') as cov,
                       json_extract(metadata, '$.missing_var_count') as missing
                FROM stage_status
                WHERE stage = ? AND status = 'success'
                  AND json_extract(metadata, '$.missing_var_count') > 0
                ORDER BY CAST(json_extract(metadata, '$.missing_var_count') AS INT) DESC
                LIMIT 10
            """, (STAGE_NAME,)).fetchall()
            if worst:
                logger.info("  Tables with most missing vars:")
                for row in worst:
                    logger.info("    %-50s %.1f%% (%d missing)",
                               row["entity_id"], row["cov"] or 0, row["missing"] or 0)
        conn.close()
    except Exception as e:
        logger.debug("Could not compute coverage summary: %s", e)

    logger.info("=" * 60)

    return 0 if summary["tables_failed"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
