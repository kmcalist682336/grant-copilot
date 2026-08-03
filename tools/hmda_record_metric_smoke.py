"""Smoke-test HMDA record metrics against direct DuckDB calculations.

This is intentionally *not* under ``tests/`` because it is a live-data smoke
tool: it reads HMDA Parquet files from GCS and compares the app's deterministic
record path against an independent DuckDB query.

Default run:

    python -m tools.hmda_record_metric_smoke

Useful while iterating:

    python -m tools.hmda_record_metric_smoke --limit 5
    python -m tools.hmda_record_metric_smoke --direct-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb
from dotenv import load_dotenv

from scripts.chatbot.aggregator import aggregate_results
from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall, RecordFilter
from scripts.chatbot.models import (
    CompoundSpec,
    ExtractedAnalysis,
    ExtractedConcept,
    ExtractedFilter,
    ExtractedGeoRef,
    ExtractedIntent,
)
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.gazetteer_db import open_spatialite
from scripts.chatbot.record_connector import build_record_caller
from scripts.chatbot.record_planner import _record_geo_prefixes, plan_record_query


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOT = "gs://demo-grant-bucket/record_level"
DEFAULT_YEAR = 2024

ACTION = "906bb78b0f70"
INCOME = "c1aa5d4f3f72"
LOAN_AMOUNT = "c02eb39025e6"
RACE = "38ad9c360a98"
SEX = "6057363dc2e9"

APPROVED_ACTIONS = ["Loan originated", "Application approved but not accepted"]
DENIED_ACTIONS = ["Application denied"]
ORIGINATED_ACTIONS = ["Loan originated"]


@dataclass(frozen=True)
class FilterSpec:
    variable_id: str
    dimension: str
    value: Any
    operator: str = "equals"


@dataclass(frozen=True)
class SmokeCase:
    query: str
    metric: str
    geo: str
    direction: Optional[str] = None
    year: int = DEFAULT_YEAR
    filters: list[FilterSpec] = field(default_factory=list)


METRICS: dict[str, dict[str, Any]] = {
    "approval_rate": {
        "operation": "percentage",
        "measure_text": "approval rate",
        "measure_hint": "mortgage approval rate",
        "variable_id": ACTION,
        "numerator": [FilterSpec(ACTION, "application status", APPROVED_ACTIONS, "in")],
    },
    "denial_rate": {
        "operation": "percentage",
        "measure_text": "denial rate",
        "measure_hint": "mortgage denial rate",
        "variable_id": ACTION,
        "numerator": [FilterSpec(ACTION, "application status", DENIED_ACTIONS, "in")],
    },
    "origination_rate": {
        "operation": "percentage",
        "measure_text": "origination rate",
        "measure_hint": "mortgage origination rate",
        "variable_id": ACTION,
        "numerator": [FilterSpec(ACTION, "application status", ORIGINATED_ACTIONS, "in")],
    },
    "application_count": {
        "operation": "count",
        "measure_text": "mortgage applications",
        "measure_hint": "loan applications",
        "variable_id": ACTION,
    },
    "denied_count": {
        "operation": "count",
        "measure_text": "denied mortgage applications",
        "measure_hint": "denied mortgage applications",
        "variable_id": ACTION,
        "filters": [FilterSpec(ACTION, "application status", DENIED_ACTIONS, "in")],
    },
    "average_income": {
        "operation": "average",
        "measure_text": "income",
        "measure_hint": "applicant income",
        "variable_id": INCOME,
    },
    "median_income": {
        "operation": "median",
        "measure_text": "income",
        "measure_hint": "applicant income",
        "variable_id": INCOME,
    },
    "average_loan_amount": {
        "operation": "average",
        "measure_text": "loan amount",
        "measure_hint": "loan amount",
        "variable_id": LOAN_AMOUNT,
    },
    "median_loan_amount": {
        "operation": "median",
        "measure_text": "loan amount",
        "measure_hint": "loan amount",
        "variable_id": LOAN_AMOUNT,
    },
}


CASES: list[SmokeCase] = [
    SmokeCase("approval rate of mortgage applications in Decatur", "approval_rate", "Decatur"),
    SmokeCase("denial rate of mortgage applications in Decatur", "denial_rate", "Decatur"),
    SmokeCase("origination rate of mortgage applications in Decatur", "origination_rate", "Decatur"),
    SmokeCase("how many mortgage applications were denied in Decatur", "denied_count", "Decatur"),
    SmokeCase("average loan amount in Decatur", "average_loan_amount", "Decatur"),
    SmokeCase("median loan amount in Decatur", "median_loan_amount", "Decatur"),
    SmokeCase("approval rate of mortgage applications in Atlanta", "approval_rate", "Atlanta"),
    SmokeCase("denial rate of mortgage applications in Atlanta", "denial_rate", "Atlanta"),
    SmokeCase("average mortgage loan amount in Atlanta", "average_loan_amount", "Atlanta"),
    SmokeCase("median mortgage loan amount in Atlanta", "median_loan_amount", "Atlanta"),
    SmokeCase("average applicant income in Atlanta", "average_income", "Atlanta"),
    SmokeCase("median applicant income in Atlanta", "median_income", "Atlanta"),
    SmokeCase("approval rate of Black mortgage applicants in Atlanta", "approval_rate", "Atlanta",
              filters=[FilterSpec(RACE, "applicant race", "Black or African American")]),
    SmokeCase("denial rate of Black mortgage applicants in Atlanta", "denial_rate", "Atlanta",
              filters=[FilterSpec(RACE, "applicant race", "Black or African American")]),
    SmokeCase("approval rate of White mortgage applicants in Atlanta", "approval_rate", "Atlanta",
              filters=[FilterSpec(RACE, "applicant race", "White")]),
    SmokeCase("denial rate of White mortgage applicants in Atlanta", "denial_rate", "Atlanta",
              filters=[FilterSpec(RACE, "applicant race", "White")]),
    SmokeCase("approval rate of Black women in Atlanta", "approval_rate", "Atlanta",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("denial rate of Black women in Atlanta", "denial_rate", "Atlanta",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("average income of Black women in Atlanta", "average_income", "Atlanta",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("average loan amount for Asian men in Atlanta", "average_loan_amount", "Atlanta",
              filters=[
                  FilterSpec(RACE, "applicant race", "Asian"),
                  FilterSpec(SEX, "applicant sex", "Male"),
              ]),
    SmokeCase("approval rate of Black women in east Atlanta", "approval_rate", "Atlanta", direction="e",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("denial rate of Black women in east Atlanta", "denial_rate", "Atlanta", direction="e",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("approval rate of Black women in southwest Atlanta", "approval_rate", "Atlanta", direction="sw",
              filters=[
                  FilterSpec(RACE, "applicant race", "Black or African American"),
                  FilterSpec(SEX, "applicant sex", "Female"),
              ]),
    SmokeCase("average income of Black mortgage applicants in southwest Atlanta", "average_income", "Atlanta", direction="sw",
              filters=[FilterSpec(RACE, "applicant race", "Black or African American")]),
    SmokeCase("denial rate of Asian mortgage applicants in east Atlanta", "denial_rate", "Atlanta", direction="e",
              filters=[FilterSpec(RACE, "applicant race", "Asian")]),
    SmokeCase("application count for female applicants in Georgia", "application_count", "Georgia",
              filters=[FilterSpec(SEX, "applicant sex", "Female")]),
    SmokeCase("approval rate of female applicants in Georgia", "approval_rate", "Georgia",
              filters=[FilterSpec(SEX, "applicant sex", "Female")]),
    SmokeCase("average income of female applicants in Georgia", "average_income", "Georgia",
              filters=[FilterSpec(SEX, "applicant sex", "Female")]),
    SmokeCase("denial rate of Black applicants in Fulton County", "denial_rate", "Fulton County",
              filters=[FilterSpec(RACE, "applicant race", "Black or African American")]),
    SmokeCase("average loan amount for White applicants in Fulton County", "average_loan_amount", "Fulton County",
              filters=[FilterSpec(RACE, "applicant race", "White")]),
]


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def setup_duckdb(root: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    access_id = os.getenv("GCS_HMAC_ACCESS_ID")
    secret = os.getenv("GCS_HMAC_SECRET")
    if access_id and secret:
        con.execute(
            "CREATE OR REPLACE SECRET hmda_gcs ("
            f"TYPE gcs, KEY_ID {sql_string(access_id)}, "
            f"SECRET {sql_string(secret)}, "
            f"SCOPE {sql_string(root.rstrip('/') + '/')})"
        )
    return con


def filter_to_extracted(spec: FilterSpec) -> ExtractedFilter:
    normalized_value = (
        json.dumps(spec.value)
        if isinstance(spec.value, list)
        else str(spec.value)
    )
    return ExtractedFilter(
        dimension=ExtractedConcept(
            text=spec.dimension,
            canonical_hint=spec.dimension,
            dataset_hint="hmda",
        ),
        operator=spec.operator if spec.operator in {"equals", "not_equals"} else "equals",
        value_text=(
            spec.value[0] if isinstance(spec.value, list) else str(spec.value)
        ),
        normalized_value_hint=normalized_value,
    )


def case_intent(case: SmokeCase) -> ExtractedIntent:
    metric = METRICS[case.metric]
    metric_filters = [
        filter_to_extracted(spec)
        for spec in metric.get("filters", [])
    ]
    user_filters = [filter_to_extracted(spec) for spec in case.filters]
    geo_ref = ExtractedGeoRef(
        text=case.geo,
        ref_type="administrative",
        scope_hint="Georgia" if case.geo not in {"Georgia"} else None,
    )
    compounds = []
    if case.direction:
        compounds = [
            CompoundSpec(
                primary_ref_idx=0,
                filter_ref_idx=0,
                op="directional",
                direction=case.direction,
            )
        ]
    return ExtractedIntent(
        geo_refs=[geo_ref],
        compounds=compounds,
        concepts=[],
        analyses=[
            ExtractedAnalysis(
                operation=metric["operation"],
                measure=ExtractedConcept(
                    text=metric["measure_text"],
                    canonical_hint=metric["measure_hint"],
                    dataset_hint="hmda",
                ),
                filters=[*metric_filters, *user_filters],
                groupings=[],
                population_context="mortgage applications",
            )
        ],
        years=[case.year],
    )


def resolve_case(case: SmokeCase, geo_db: sqlite3.Connection):
    resolved = resolve_intent(case_intent(case), geo_db)
    if not resolved:
        raise RuntimeError(f"no geography resolved for {case.query!r}")
    if len(resolved) != 1:
        raise RuntimeError(
            f"expected one geography for {case.query!r}, got {len(resolved)}"
        )
    return resolved[0]


def direct_value(
    con: duckdb.DuckDBPyConnection,
    *,
    root: str,
    case: SmokeCase,
    geo_prefixes: list[str],
) -> tuple[Optional[float], int]:
    metric = METRICS[case.metric]
    metric_filters = list(metric.get("filters", []))
    filters = [*metric_filters, *case.filters]
    numerator_filters = list(metric.get("numerator", []))
    selected_vars = sorted({
        metric["variable_id"],
        *(flt.variable_id for flt in filters),
        *(flt.variable_id for flt in numerator_filters),
    })
    aliases = {var: f"v{i}" for i, var in enumerate(selected_vars)}
    case_columns = ", ".join(
        f"MAX(CASE WHEN variable = ? THEN CAST(value AS VARCHAR) END) AS {alias}"
        for var, alias in aliases.items()
    )
    path = (
        f"{root.rstrip('/')}/table_id=hmda/year={case.year}/"
        "variable=*/hmda_*.parquet"
    )
    where_vars = ", ".join("?" for _ in selected_vars)
    sql = (
        "WITH pivoted AS ("
        f"SELECT record_id, MAX(CAST(census_tract AS VARCHAR)) AS census_tract, "
        f"{case_columns} "
        "FROM read_parquet(?, hive_partitioning=true) "
        f"WHERE variable IN ({where_vars}) "
        "GROUP BY record_id"
        ") "
    )
    params: list[Any] = [*selected_vars, path, *selected_vars]

    conditions: list[str] = []
    where_params: list[Any] = []
    if geo_prefixes:
        geo_parts = []
        for prefix in geo_prefixes:
            geo_parts.append("LEFT(census_tract, ?) = ?")
            where_params.extend([len(prefix), prefix])
        conditions.append("(" + " OR ".join(geo_parts) + ")")
    for flt in filters:
        alias = aliases[flt.variable_id]
        if flt.operator in {"in", "not_in"} or isinstance(flt.value, list):
            values = list(flt.value if isinstance(flt.value, list) else [flt.value])
            placeholders = ", ".join("?" for _ in values)
            op = "NOT IN" if flt.operator == "not_in" else "IN"
            conditions.append(f"{alias} {op} ({placeholders})")
            where_params.extend(values)
        elif flt.operator == "not_equals":
            conditions.append(f"{alias} <> ?")
            where_params.append(flt.value)
        else:
            conditions.append(f"{alias} = ?")
            where_params.append(flt.value)

    where_clause = "" if not conditions else " WHERE " + " AND ".join(conditions)
    measure_alias = aliases[metric["variable_id"]]
    operation = metric["operation"]
    if operation == "percentage":
        numerator_conditions = []
        numerator_params: list[Any] = []
        for flt in numerator_filters:
            alias = aliases[flt.variable_id]
            values = list(flt.value if isinstance(flt.value, list) else [flt.value])
            placeholders = ", ".join("?" for _ in values)
            numerator_conditions.append(f"{alias} IN ({placeholders})")
            numerator_params.extend(values)
        numerator_clause = " AND ".join(numerator_conditions)
        sql += (
            "SELECT "
            f"SUM(CASE WHEN {numerator_clause} THEN 1 ELSE 0 END)::DOUBLE "
            "/ NULLIF(COUNT(*), 0) AS metric_value, "
            "COUNT(*) AS rows_used "
            f"FROM pivoted{where_clause}"
        )
        params.extend(numerator_params)
    elif operation == "count":
        sql += f"SELECT COUNT(*)::DOUBLE AS metric_value, COUNT(*) AS rows_used FROM pivoted{where_clause}"
    elif operation == "average":
        sql += (
            f"SELECT AVG(TRY_CAST({measure_alias} AS DOUBLE)) AS metric_value, "
            f"COUNT(TRY_CAST({measure_alias} AS DOUBLE)) AS rows_used "
            f"FROM pivoted{where_clause}"
        )
    elif operation == "median":
        sql += (
            f"SELECT median(TRY_CAST({measure_alias} AS DOUBLE)) AS metric_value, "
            f"COUNT(TRY_CAST({measure_alias} AS DOUBLE)) AS rows_used "
            f"FROM pivoted{where_clause}"
        )
    else:
        raise ValueError(f"unsupported metric operation: {operation}")
    params.extend(where_params)
    value, rows_used = con.execute(sql, params).fetchone()
    return value, int(rows_used or 0)


async def planner_value(
    *,
    case: SmokeCase,
    resolved_geo: Any,
    geo_db: sqlite3.Connection,
    metadata_db: sqlite3.Connection,
    record_caller: Any,
) -> tuple[Optional[float], int, list[str]]:
    intent = case_intent(case)
    plan = plan_record_query(
        intent,
        [resolved_geo],
        semantic_router=None,
        geo_db=geo_db,
        metadata_db=metadata_db,
    )
    fetch_results = await record_caller.fetch_all(
        [planned.api_call for planned in plan.calls]
    )
    aggregated = aggregate_results(plan, fetch_results)
    if aggregated.fetch_failures:
        return None, 0, aggregated.fetch_failures
    if not aggregated.values:
        return None, 0, ["no aggregated values"]
    value = aggregated.values[0]
    metric_value = value.ratio if value.ratio is not None else value.value
    return metric_value, value.rows_aggregated, list(value.notes)


def close_quietly(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if close:
        close()


def almost_equal(a: Optional[float], b: Optional[float], *, tolerance: float) -> bool:
    if a is None or b is None:
        return a is b
    if math.isnan(float(a)) or math.isnan(float(b)):
        return math.isnan(float(a)) and math.isnan(float(b))
    return abs(float(a) - float(b)) <= tolerance


async def run(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    root = args.hmda_root
    cases = CASES[: args.limit] if args.limit else CASES

    gazetteer_path = REPO_ROOT / "data" / "geo" / "gazetteer.db"
    copied_gazetteer: Optional[Path] = None
    if args.copy_gazetteer_to_temp:
        temp_dir = Path(tempfile.gettempdir()) / "grant-copilot-smoke"
        temp_dir.mkdir(parents=True, exist_ok=True)
        copied_gazetteer = temp_dir / "gazetteer.db"
        if (
            not copied_gazetteer.exists()
            or copied_gazetteer.stat().st_size != gazetteer_path.stat().st_size
            or copied_gazetteer.stat().st_mtime < gazetteer_path.stat().st_mtime
        ):
            shutil.copy2(gazetteer_path, copied_gazetteer)
        gazetteer_path = copied_gazetteer

    try:
        geo_db = open_spatialite(gazetteer_path)
        spatialite_available = True
    except sqlite3.OperationalError as exc:
        if any(case.direction for case in cases):
            raise RuntimeError(
                "Directional smoke cases require SpatiaLite. Run this script "
                "inside WSL/Docker, or use --limit 20 / --direct-only with "
                "non-directional cases on Windows Python."
            ) from exc
        geo_db = sqlite3.connect(gazetteer_path)
        geo_db.row_factory = sqlite3.Row
        spatialite_available = False
    metadata_db = sqlite3.connect(REPO_ROOT / "data" / "metadata" / "search_index.db")
    metadata_db.row_factory = sqlite3.Row
    direct_con = setup_duckdb(root)
    record_caller = None
    record_con = None
    if not args.direct_only:
        cache_path = Path(tempfile.gettempdir()) / "hmda-record-metric-smoke-cache.sqlite"
        record_caller, record_con = build_record_caller(
            root,
            APICache(cache_path),
            record_id_column="record_id",
            layout="variable_tree",
            file_glob="hmda_*.parquet",
            geography_partition=None,
        )

    failures = 0
    try:
        print(f"Running {len(cases)} HMDA record metric smoke case(s)")
        print(f"Root: {root}")
        if copied_gazetteer is not None:
            print(f"Gazetteer copy: {copied_gazetteer}")
        if not spatialite_available:
            print("SpatiaLite unavailable; running non-directional cases only.")
        print()
        for idx, case in enumerate(cases, start=1):
            resolved_geo = resolve_case(case, geo_db)
            geo_prefixes = _record_geo_prefixes(resolved_geo, geo_db)
            direct, direct_rows = direct_value(
                direct_con,
                root=root,
                case=case,
                geo_prefixes=geo_prefixes,
            )
            status = "OK"
            details = (
                f"direct={direct!r} rows={direct_rows} "
                f"geo={resolved_geo.display_name} tracts/prefixes={len(geo_prefixes)}"
            )
            if not args.direct_only and record_caller is not None:
                planned, planned_rows, notes = await planner_value(
                    case=case,
                    resolved_geo=resolved_geo,
                    geo_db=geo_db,
                    metadata_db=metadata_db,
                    record_caller=record_caller,
                )
                if not almost_equal(planned, direct, tolerance=args.tolerance):
                    status = "FAIL"
                    failures += 1
                details += (
                    f" | planner={planned!r} rows={planned_rows}"
                    + (f" notes={notes}" if notes else "")
                )
            print(f"{idx:02d}. {status} {case.query}")
            print(f"    {details}")
    finally:
        close_quietly(record_con)
        close_quietly(direct_con)
        close_quietly(metadata_db)
        close_quietly(geo_db)

    print()
    if failures:
        print(f"FAILED: {failures} case(s) differed from direct DuckDB")
        return 1
    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmda-root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Only compute direct DuckDB truth; skip planner/caller comparison.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Allowed absolute difference between direct and planner values.",
    )
    parser.add_argument(
        "--no-copy-gazetteer-to-temp",
        dest="copy_gazetteer_to_temp",
        action="store_false",
        help=(
            "Open data/geo/gazetteer.db in place. By default the smoke test "
            "copies it to the OS temp directory to avoid SQLite lock/FTS "
            "issues on Windows-mounted WSL paths."
        ),
    )
    parser.set_defaults(copy_gazetteer_to_temp=True)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
