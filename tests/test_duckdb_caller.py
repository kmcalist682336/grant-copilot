from __future__ import annotations

import asyncio

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall
from scripts.chatbot.duckdb_caller import DuckDBCaller


class FakeCursor:
    description = [("geoid",), ("value",)]

    def fetchall(self):
        return [("13121", 42)]


class FakeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        return FakeCursor()


def _plan(url: str = "duckdb://stable-key") -> APIPlanCall:
    return APIPlanCall(
        url=url,
        table_id="loans",
        variables=["value"],
        geo_level="county",
        geo_filter_ids=["13121"],
        year=2024,
        dataset="gcs/v1",
        ttl_seconds=3600,
    )


def test_fetch_matches_census_contract_and_uses_cache(tmp_path):
    con = FakeConnection()
    caller = DuckDBCaller(
        con,
        APICache(tmp_path / "cache.db"),
        bucket_uri="gs://example-bucket",
    )

    first = asyncio.run(caller.fetch(_plan()))
    second = asyncio.run(caller.fetch(_plan()))

    assert first.rows == [{"geoid": "13121", "value": 42}]
    assert first.error is None
    assert first.cache_hit is False
    assert second.rows == first.rows
    assert second.cache_hit is True
    assert len(con.calls) == 1
    sql, params = con.calls[0]
    assert '"geoid" IN (?)' in sql
    assert params == [
        "gs://example-bucket/gcs/v1/table_id=loans/year=2024/"
        "geo_level=county/*.parquet",
        "13121",
    ]


def test_fetch_returns_query_errors(tmp_path):
    class BrokenConnection:
        def execute(self, sql, params):
            raise RuntimeError("not available")

    caller = DuckDBCaller(
        BrokenConnection(),
        APICache(tmp_path / "cache.db"),
        bucket_uri="gs://example-bucket",
    )

    result = asyncio.run(caller.fetch(_plan()))

    assert result.rows == []
    assert result.error == "duckdb_error: not available"


def test_variable_tree_layout_pivots_variable_files(tmp_path):
    con = FakeConnection()
    caller = DuckDBCaller(
        con,
        APICache(tmp_path / "cache.db"),
        bucket_uri="gs://example-bucket/variable_tree",
        geo_id_column="hmda_record_id",
        layout="variable_tree",
    )

    result = asyncio.run(caller.fetch(_plan()))

    assert result.error is None
    sql, params = con.calls[0]
    assert "MAX(CASE WHEN variable = ? THEN value END)" in sql
    assert "GROUP BY \"hmda_record_id\"" in sql
    assert params == [
        "value",
        "gs://example-bucket/variable_tree/table_id=loans/year=2024/"
        "variable=*/part-*.parquet",
        "value",
        "13121",
    ]
