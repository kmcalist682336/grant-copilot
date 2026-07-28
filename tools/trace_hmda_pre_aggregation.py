"""Trace shared router/planner output through HMDA fetch, stopping pre-aggregation.

This intentionally adds no HMDA-specific router or planner. It uses the
existing extraction, geography resolution, SemanticRouter, and plan_query,
then chooses a connector solely from APIPlanCall.dataset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

import duckdb
import yaml
from dotenv import load_dotenv

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.auth_check import check_auth
from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.duckdb_caller import DuckDBCaller
from scripts.chatbot.extractor import extract_intent
from scripts.chatbot.gazetteer_db import open_spatialite
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.llm_client import GeminiVertexClient
from scripts.chatbot.planner import plan_query
from scripts.chatbot.semantic_router import (
    RouterConfig,
    SemanticRouter,
    VertexEmbedder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _print_json(label: str, value: object) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(value, indent=2, default=str))


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    access_id = os.getenv("GCS_HMAC_ACCESS_ID")
    secret = os.getenv("GCS_HMAC_SECRET")
    if not access_id or not secret:
        raise RuntimeError(
            "Set GCS_HMAC_ACCESS_ID and GCS_HMAC_SECRET in .env"
        )
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute(
        "CREATE OR REPLACE SECRET hmda_gcs ("
        f"TYPE gcs, KEY_ID {_sql_string(access_id)}, "
        f"SECRET {_sql_string(secret)}, "
        "SCOPE 'gs://demo-grant-bucket/')"
    )
    return con


async def run(query: str) -> int:
    load_dotenv(REPO_ROOT / ".env")
    with (REPO_ROOT / "config" / "chatbot.yaml").open() as handle:
        config = yaml.safe_load(handle)

    auth = check_auth(verbose=False)
    project = auth["project_id"]
    location = auth.get("location", "us-central1")
    vertex = config.get("vertex_ai", {})
    llm = GeminiVertexClient(
        project_id=project,
        location=location,
        model=vertex.get("model", "gemini-2.5-flash"),
        temperature=vertex.get("temperature", 0.1),
        timeout_seconds=vertex.get("timeout_seconds", 20.0),
        enable_cache=vertex.get("enable_cache", True),
        cache_ttl_seconds=vertex.get("cache_ttl_seconds", 3600),
    )

    metadata_path = REPO_ROOT / config["paths"]["metadata_db"]
    gazetteer_path = REPO_ROOT / config["paths"]["gazetteer_db"]
    metadata_db = sqlite3.connect(metadata_path)
    metadata_db.row_factory = sqlite3.Row
    gazetteer_db = open_spatialite(gazetteer_path)
    gazetteer_db.row_factory = sqlite3.Row
    router = SemanticRouter(
        REPO_ROOT / "data" / "metadata" / "embeddings.faiss",
        metadata_path,
        VertexEmbedder(project=project, location=location),
        config=RouterConfig(),
    )
    concept_map = ConceptMap.from_yaml(REPO_ROOT / config["paths"]["concept_map"])

    try:
        intent = extract_intent(query, llm)
        _print_json("1. ExtractedIntent (shared)", intent.model_dump())

        resolved_geos = resolve_intent(intent, gazetteer_db)
        _print_json(
            "2. ResolvedGeography list (shared)",
            [
                geo.model_dump() if hasattr(geo, "model_dump") else str(geo)
                for geo in resolved_geos
            ],
        )

        # No route_dataset and no HMDA-specific resolution. plan_query invokes
        # the same SemanticRouter path used for every dataset.
        plan = plan_query(
            intent,
            resolved_geos,
            concept_map,
            metadata_db,
            semantic_router=router,
        )
        _print_json(
            "3. Router resolutions (shared)",
            [
                {
                    "concept": item.concept.model_dump(),
                    "tier": item.tier,
                    "notes": item.notes,
                    "top_tables": [
                        {
                            "table_id": candidate.target_table_id,
                            "dataset": candidate.target_dataset,
                            "score": round(candidate.aggregate_score, 4),
                        }
                        for candidate in (
                            item.routed_result.top_tables[:10]
                            if item.routed_result else []
                        )
                    ],
                    "top_variables": [
                        {
                            "table_id": candidate.target_table_id,
                            "dataset": candidate.target_dataset,
                            "variable_id": candidate.target_variable_id,
                            "score": round(candidate.aggregate_score, 4),
                        }
                        for candidate in (
                            item.routed_result.top_variables[:10]
                            if item.routed_result else []
                        )
                    ],
                }
                for item in plan.concept_resolutions
            ],
        )
        _print_json(
            "4. APIPlanCalls from shared planner",
            [
                {
                    "role": call.role,
                    "geo_idx": call.geo_idx,
                    "concept_idx": call.concept_idx,
                    "api_call": asdict(call.api_call),
                }
                for call in plan.calls
            ],
        )

        hmda_calls = [
            call.api_call for call in plan.calls
            if call.api_call.dataset == "hmda"
        ]
        non_hmda = [
            asdict(call.api_call) for call in plan.calls
            if call.api_call.dataset != "hmda"
        ]
        _print_json(
            "5. Connector dispatch",
            {
                "rule": "dataset == 'hmda' -> DuckDBCaller; otherwise CensusCaller",
                "hmda_call_count": len(hmda_calls),
                "non_hmda_calls_not_executed_in_this_trace": non_hmda,
            },
        )
        if not hmda_calls:
            print(
                "\nSTOP: the shared router/planner produced no HMDA calls. "
                "No dataset-specific routing was added to force one."
            )
            return 0

        con = _duckdb_connection()
        try:
            caller = DuckDBCaller(
                con,
                APICache(REPO_ROOT / "data" / ".cache" / "hmda_trace_cache.db"),
                bucket_uri="gs://demo-grant-bucket/variable_tree",
                geo_id_column="record_id",
                layout="variable_tree",
                max_concurrent=1,
            )
            results = await caller.fetch_all(hmda_calls)
        finally:
            con.close()

        _print_json(
            "6. FetchResults from DuckDBCaller",
            [asdict(result) for result in results],
        )
        print("\nSTOP: aggregation and synthesis were intentionally not called.")
        return 0
    finally:
        router.metadata_db.close()
        metadata_db.close()
        gazetteer_db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "What was the average income of female mortgage applicants "
            "in Fulton County, Georgia in 2024?"
        ),
    )
    args = parser.parse_args()
    return asyncio.run(run(args.query))


if __name__ == "__main__":
    raise SystemExit(main())
