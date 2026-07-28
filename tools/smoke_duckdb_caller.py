"""Live smoke test for DuckDBCaller against one HMDA record in GCS."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall
from scripts.chatbot.duckdb_caller import DuckDBCaller

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIABLE_ID = "c1aa5d4f3f72"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    access_id = os.getenv("GCS_HMAC_ACCESS_ID")
    secret = os.getenv("GCS_HMAC_SECRET")
    if not access_id or not secret:
        raise RuntimeError("Set GCS_HMAC_ACCESS_ID and GCS_HMAC_SECRET in .env")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        con.execute(
            "CREATE OR REPLACE SECRET hmda_gcs ("
            f"TYPE gcs, KEY_ID {sql_string(access_id)}, "
            f"SECRET {sql_string(secret)}, "
            "SCOPE 'gs://demo-grant-bucket/')"
        )
        record_path = (
            "gs://demo-grant-bucket/variable_tree/table_id=hmda/year=2024/"
            f"variable={VARIABLE_ID}/part-*.parquet"
        )
        record = con.execute(
            "SELECT record_id FROM read_parquet(?) "
            "WHERE value IS NOT NULL LIMIT 1",
            [record_path],
        ).fetchone()
        if record is None:
            raise RuntimeError("No populated HMDA record found for smoke test")
        record_id = record[0]
        cache = APICache(REPO_ROOT / "data" / ".cache" / "duckdb_smoke.db")
        caller = DuckDBCaller(
            con,
            cache,
            bucket_uri="gs://demo-grant-bucket/variable_tree",
            geo_id_column="record_id",
            layout="variable_tree",
            max_concurrent=1,
        )
        plan = APIPlanCall(
            url=f"duckdb://hmda/2024/{VARIABLE_ID}/{record_id}",
            table_id="hmda",
            variables=[VARIABLE_ID],
            geo_level="record",
            geo_filter_ids=[record_id],
            year=2024,
            dataset="hmda",
            ttl_seconds=300,
        )
        first = await caller.fetch(plan)
        if first.error:
            raise RuntimeError(first.error)
        second = await caller.fetch(plan)
        if not first.rows:
            raise RuntimeError("DuckDBCaller returned no rows")
        print("wrapper_fetch_ok:", first.rows)
        print("cache_hit_on_second_fetch:", second.cache_hit)
    finally:
        con.close()


if __name__ == "__main__":
    asyncio.run(main())
