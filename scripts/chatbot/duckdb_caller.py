"""Async fetch wrapper for Hive-partitioned Parquet queried by DuckDB.

``DuckDBCaller`` intentionally matches the ``CensusCaller`` fetch contract so
the existing aggregation and synthesis pipeline can consume either source.
The supplied DuckDB connection must already be configured for GCS access.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Iterable

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall, FetchResult

logger = logging.getLogger(__name__)

_SAFE_PARTITION_VALUE = re.compile(r"^[A-Za-z0-9._/-]+$")


class DuckDBCaller:
    """Drop-in fetcher backed by Hive-partitioned Parquet files."""

    def __init__(
        self,
        con: Any,
        cache: APICache,
        *,
        bucket_uri: str,
        geo_id_column: str = "geoid",
        layout: str = "wide_hive",
        max_concurrent: int = 8,
    ) -> None:
        if layout not in {"wide_hive", "variable_tree"}:
            raise ValueError(f"Unsupported DuckDB layout: {layout}")
        self._con = con
        self.cache = cache
        self.bucket_uri = bucket_uri.rstrip("/")
        self.geo_id_column = geo_id_column
        self.layout = layout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._query_lock = asyncio.Lock()

    async def __aenter__(self) -> "DuckDBCaller":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def fetch(self, plan: APIPlanCall) -> FetchResult:
        """Return one cache-first result; query failures are never raised."""
        started = time.time()
        cached = self.cache.get(plan.url)
        if cached is not None:
            return FetchResult(
                plan=plan,
                rows=self._to_rows(cached),
                cache_hit=True,
                elapsed_s=time.time() - started,
            )

        try:
            async with self._semaphore:
                # DuckDB connections are not safe for overlapping queries.
                async with self._query_lock:
                    data = await asyncio.to_thread(self._query, plan)
        except Exception as exc:
            return FetchResult(
                plan=plan,
                rows=[],
                cache_hit=False,
                elapsed_s=time.time() - started,
                error=f"duckdb_error: {exc}",
            )

        try:
            self.cache.set(plan.url, data, ttl_seconds=plan.ttl_seconds)
        except Exception as exc:
            logger.warning("Cache set failed for %s: %s", plan.cache_key, exc)

        return FetchResult(
            plan=plan,
            rows=self._to_rows(data),
            cache_hit=False,
            elapsed_s=time.time() - started,
        )

    async def fetch_all(self, plans: Iterable[APIPlanCall]) -> list[FetchResult]:
        """Fetch calls concurrently and preserve their input order."""
        return await asyncio.gather(*(self.fetch(plan) for plan in plans))

    def _query(self, plan: APIPlanCall) -> list[dict]:
        self._validate_partition_value("dataset", plan.dataset)
        self._validate_partition_value("table_id", plan.table_id)
        self._validate_partition_value("geo_level", plan.geo_level)

        if self.layout == "variable_tree":
            return self._query_variable_tree(plan)
        return self._query_wide_hive(plan)

    def _query_wide_hive(self, plan: APIPlanCall) -> list[dict]:
        columns = [self.geo_id_column, *plan.variables]
        select_list = ", ".join(self._quote_identifier(col) for col in columns)
        path = (
            f"{self.bucket_uri}/{plan.dataset}/table_id={plan.table_id}/"
            f"year={plan.year}/geo_level={plan.geo_level}/*.parquet"
        )

        sql = f"SELECT {select_list} FROM read_parquet(?)"
        params: list[Any] = [path]
        if plan.geo_filter_ids:
            placeholders = ", ".join("?" for _ in plan.geo_filter_ids)
            sql += (
                f" WHERE {self._quote_identifier(self.geo_id_column)} "
                f"IN ({placeholders})"
            )
            params.extend(plan.geo_filter_ids)

        cursor = self._con.execute(sql, params)
        description = cursor.description or []
        headers = [column[0] for column in description]
        return [dict(zip(headers, row)) for row in cursor.fetchall()]

    def _query_variable_tree(self, plan: APIPlanCall) -> list[dict]:
        """Pivot one-file-per-variable rows into the Census-style contract."""
        if not plan.variables:
            return []
        for variable in plan.variables:
            self._validate_partition_value("variable_id", variable)

        record_col = self._quote_identifier(self.geo_id_column)
        value_columns = ", ".join(
            f"MAX(CASE WHEN variable = ? THEN value END) "
            f"AS {self._quote_identifier(variable)}"
            for variable in plan.variables
        )
        path = (
            f"{self.bucket_uri}/table_id={plan.table_id}/year={plan.year}/"
            "variable=*/part-*.parquet"
        )
        variable_placeholders = ", ".join("?" for _ in plan.variables)
        sql = (
            f"SELECT {record_col}, {value_columns} "
            "FROM read_parquet(?, hive_partitioning=true) "
            f"WHERE variable IN ({variable_placeholders})"
        )
        params: list[Any] = [*plan.variables, path, *plan.variables]
        if plan.geo_filter_ids:
            record_placeholders = ", ".join("?" for _ in plan.geo_filter_ids)
            sql += f" AND {record_col} IN ({record_placeholders})"
            params.extend(plan.geo_filter_ids)
        sql += f" GROUP BY {record_col} ORDER BY {record_col}"

        cursor = self._con.execute(sql, params)
        headers = [column[0] for column in (cursor.description or [])]
        return [dict(zip(headers, row)) for row in cursor.fetchall()]

    @staticmethod
    def _to_rows(data: Any) -> list[dict]:
        return list(data) if data else []

    @staticmethod
    def _quote_identifier(value: str) -> str:
        if not value or "\x00" in value:
            raise ValueError("DuckDB column identifiers must be non-empty")
        return f'"{value.replace(chr(34), chr(34) * 2)}"'

    @staticmethod
    def _validate_partition_value(name: str, value: str) -> None:
        if not value or not _SAFE_PARTITION_VALUE.fullmatch(value):
            raise ValueError(f"Unsafe {name} partition value: {value!r}")
