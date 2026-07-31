"""Async fetch wrapper for Hive-partitioned Parquet queried by DuckDB.

``DuckDBCaller`` intentionally matches the ``CensusCaller`` fetch contract so
the existing aggregation and synthesis pipeline can consume either source.
The supplied DuckDB connection must already be configured for GCS access.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Iterable, Optional

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import APIPlanCall, FetchResult, RecordFilter

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
        file_glob: str = "*.parquet",
        geography_partition: Optional[str] = None,
        max_concurrent: int = 8,
    ) -> None:
        if layout not in {"wide_hive", "variable_tree"}:
            raise ValueError(f"Unsupported DuckDB layout: {layout}")
        self._con = con
        self.cache = cache
        self.bucket_uri = bucket_uri.rstrip("/")
        self.geo_id_column = geo_id_column
        self.layout = layout
        if not file_glob or "/" in file_glob or "\\" in file_glob:
            raise ValueError("file_glob must be a single filename pattern")
        self.file_glob = file_glob
        if geography_partition is not None:
            self._validate_partition_value(
                "geography_partition", geography_partition,
            )
        self.geography_partition = geography_partition
        self._semaphore = threading.Semaphore(max_concurrent)
        self._query_lock = threading.Lock()

    async def __aenter__(self) -> "DuckDBCaller":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def fetch(self, plan: APIPlanCall) -> FetchResult:
        """Return one cache-first result; query failures are never raised."""
        started = time.time()
        # The URL predates record predicates and therefore cannot distinguish
        # two otherwise identical calls with different filters. Use the
        # stable structured key whenever filters are present.
        cache_key = (
            plan.cache_key
            if (plan.record_filters or plan.record_numerator_filters or plan.geo_prefixes)
            else plan.url
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return FetchResult(
                plan=plan,
                rows=self._to_rows(cached),
                cache_hit=True,
                elapsed_s=time.time() - started,
            )

        try:
            data = await asyncio.to_thread(self._query_threadsafe, plan)
        except Exception as exc:
            return FetchResult(
                plan=plan,
                rows=[],
                cache_hit=False,
                elapsed_s=time.time() - started,
                error=f"duckdb_error: {exc}",
            )

        try:
            self.cache.set(cache_key, data, ttl_seconds=plan.ttl_seconds)
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

    def _query_threadsafe(self, plan: APIPlanCall) -> list[dict]:
        """Run one DuckDB query without binding locks to an event loop."""
        with self._semaphore:
            # DuckDB connections are not safe for overlapping queries.
            with self._query_lock:
                return self._query(plan)

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
        predicates = self._record_filter_predicates(plan.record_filters)
        if predicates:
            joiner = " AND " if plan.geo_filter_ids else " WHERE "
            sql += joiner + " AND ".join(
                predicate for predicate, _ in predicates
            )
            for _, values in predicates:
                params.extend(values)

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

        # Preserve the original compact query when no record predicates are
        # present. This keeps the Census-compatible fetch shape and existing
        # cache/query behavior unchanged for all current callers.
        if not plan.record_filters and not plan.record_numerator_filters and not plan.geo_prefixes:
            return self._query_variable_tree_unfiltered(plan, record_col)

        if not plan.record_filters and not plan.record_numerator_filters and plan.geo_prefixes:
            return self._query_variable_tree_geo_only(plan, record_col)

        filter_variables = [flt.variable_id for flt in plan.record_filters]
        numerator_variables = [
            flt.variable_id for flt in plan.record_numerator_filters
        ]
        all_variables = list(dict.fromkeys([
            *plan.variables, *filter_variables, *numerator_variables,
        ]))
        for variable in all_variables:
            self._validate_partition_value("variable_id", variable)

        value_columns = ", ".join(
            f"MAX(CASE WHEN variable = ? THEN value END) "
            f"AS {self._quote_identifier(variable)}"
            for variable in all_variables
        )
        path = (
            f"{self.bucket_uri}/table_id={plan.table_id}/year={plan.year}/"
            f"{self._variable_path_prefix()}variable=*/{self.file_glob}"
        )
        variable_placeholders = ", ".join("?" for _ in all_variables)
        geography_select = ""
        if plan.geo_prefixes:
            geography_select = (
                ', MAX(CAST("census_tract" AS VARCHAR)) AS "__geo_filter"'
            )
        numerator_conditions = []
        numerator_params: list[Any] = []
        if plan.record_numerator_filters:
            numerator_conditions = self._record_filter_predicates(
                plan.record_numerator_filters,
            )
            for _, condition_params in numerator_conditions:
                numerator_params.extend(condition_params)

        sql = (
            "WITH pivoted AS ("
            f" SELECT {record_col}, {value_columns}{geography_select} "
            "FROM read_parquet(?, hive_partitioning=true) "
            f"WHERE variable IN ({variable_placeholders})"
        )
        params: list[Any] = [*all_variables, *numerator_params, path, *all_variables]
        if plan.geo_filter_ids:
            record_placeholders = ", ".join("?" for _ in plan.geo_filter_ids)
            filter_column = self._geo_filter_column(record_col)
            sql += f" AND {filter_column} IN ({record_placeholders})"
            params.extend(plan.geo_filter_ids)
        sql += f" GROUP BY {record_col}) SELECT {record_col}"
        if plan.variables:
            sql += ", " + ", ".join(
                self._quote_identifier(variable) for variable in plan.variables
            )
        if plan.record_numerator_filters:
            numerator_clause = " AND ".join(
                condition for condition, _ in numerator_conditions
            ) or "FALSE"
            sql += (
                f', CASE WHEN {numerator_clause} '
                'THEN 1 ELSE 0 END AS "__record_numerator__", '
                '1 AS "__record_denominator__"'
            )
        sql += " FROM pivoted"
        conditions = self._record_filter_predicates(plan.record_filters)
        if conditions:
            sql += " WHERE " + " AND ".join(condition for condition, _ in conditions)
            for _, condition_params in conditions:
                params.extend(condition_params)
        if plan.geo_prefixes:
            geo_conditions = [
                'LEFT("__geo_filter", ?) = ?'
                for _ in plan.geo_prefixes
            ]
            geo_params: list[Any] = []
            for prefix in plan.geo_prefixes:
                geo_params.extend([len(prefix), prefix])
            # A geography can expand to many census tracts.  A record belongs
            # to one tract, so those alternatives must be ORed together;
            # joining them with AND makes every multi-tract geography empty.
            geo_clause = "(" + " OR ".join(geo_conditions) + ")"
            sql += (" AND " if conditions else " WHERE ") + geo_clause
            params.extend(geo_params)
        sql += f" ORDER BY {record_col}"

        cursor = self._con.execute(sql, params)
        headers = [column[0] for column in (cursor.description or [])]
        return [dict(zip(headers, row)) for row in cursor.fetchall()]

    def _query_variable_tree_unfiltered(
        self, plan: APIPlanCall, record_col: str,
    ) -> list[dict]:
        """Original variable-tree query, kept stable for no-filter calls."""
        value_columns = ", ".join(
            f"MAX(CASE WHEN variable = ? THEN value END) "
            f"AS {self._quote_identifier(variable)}"
            for variable in plan.variables
        )
        path = (
            f"{self.bucket_uri}/table_id={plan.table_id}/year={plan.year}/"
            f"{self._variable_path_prefix()}variable=*/{self.file_glob}"
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
            filter_column = self._geo_filter_column(record_col)
            sql += f" AND {filter_column} IN ({record_placeholders})"
            params.extend(plan.geo_filter_ids)
        sql += f" GROUP BY {record_col} ORDER BY {record_col}"

        cursor = self._con.execute(sql, params)
        headers = [column[0] for column in (cursor.description or [])]
        return [dict(zip(headers, row)) for row in cursor.fetchall()]

    def _query_variable_tree_geo_only(
        self, plan: APIPlanCall, record_col: str,
    ) -> list[dict]:
        """Pivot selected variables and filter by census-tract prefixes."""
        value_columns = ", ".join(
            f"MAX(CASE WHEN variable = ? THEN value END) "
            f"AS {self._quote_identifier(variable)}"
            for variable in plan.variables
        )
        path = (
            f"{self.bucket_uri}/table_id={plan.table_id}/year={plan.year}/"
            f"{self._variable_path_prefix()}variable=*/{self.file_glob}"
        )
        variable_placeholders = ", ".join("?" for _ in plan.variables)
        geo_conditions = [
            'LEFT("__geo_filter", ?) = ?'
            for _ in plan.geo_prefixes
        ]
        sql = (
            "WITH pivoted AS ("
            f" SELECT {record_col}, {value_columns}, "
            'MAX(CAST("census_tract" AS VARCHAR)) AS "__geo_filter" '
            "FROM read_parquet(?, hive_partitioning=true) "
            f"WHERE variable IN ({variable_placeholders}) "
            f"GROUP BY {record_col}) "
            f"SELECT {record_col}, "
            + ", ".join(self._quote_identifier(variable) for variable in plan.variables)
            # The tract list is an inclusive set, not a set of simultaneous
            # predicates.  Parenthesized OR keeps the rest of the query safe.
            + ' FROM pivoted WHERE (' + " OR ".join(geo_conditions) + ')'
            + f" ORDER BY {record_col}"
        )
        params: list[Any] = [*plan.variables, path, *plan.variables]
        for prefix in plan.geo_prefixes:
            params.extend([len(prefix), prefix])
        cursor = self._con.execute(sql, params)
        headers = [column[0] for column in (cursor.description or [])]
        return [dict(zip(headers, row)) for row in cursor.fetchall()]

    def _record_filter_predicates(
        self, filters: list[RecordFilter],
    ) -> list[tuple[str, list[Any]]]:
        """Convert data-only filters into safe SQL + bound parameters."""
        predicates: list[tuple[str, list[Any]]] = []
        for flt in filters:
            self._validate_partition_value("filter_variable_id", flt.variable_id)
            column = self._quote_identifier(flt.variable_id)
            operator = flt.operator
            if operator == "equals":
                predicates.append((
                    f"LOWER(CAST({column} AS VARCHAR)) = "
                    f"LOWER(CAST(? AS VARCHAR))", [flt.value],
                ))
            elif operator == "not_equals":
                predicates.append((
                    f"LOWER(CAST({column} AS VARCHAR)) <> "
                    f"LOWER(CAST(? AS VARCHAR))", [flt.value],
                ))
            elif operator in {"in", "not_in"}:
                if isinstance(flt.value, (str, bytes)) or flt.value is None:
                    raise ValueError(f"{operator} requires a non-string iterable")
                values = list(flt.value)
                if not values:
                    raise ValueError(f"{operator} requires at least one value")
                placeholders = ", ".join(
                    "LOWER(CAST(? AS VARCHAR))" for _ in values
                )
                comparator = "IN" if operator == "in" else "NOT IN"
                predicates.append((
                    f"LOWER(CAST({column} AS VARCHAR)) {comparator} ({placeholders})",
                    values,
                ))
            elif operator == "is_null":
                predicates.append((f"{column} IS NULL", []))
            elif operator == "is_not_null":
                predicates.append((f"{column} IS NOT NULL", []))
            else:  # defensive: the Literal is not a runtime trust boundary
                raise ValueError(f"Unsupported record filter operator: {operator!r}")
        return predicates

    def _variable_path_prefix(self) -> str:
        """Return the optional Hive geography partition segment."""
        if self.geography_partition is None:
            return ""
        return f"{self.geography_partition}=*/"

    def _geo_filter_column(self, record_col: str) -> str:
        """Choose the column represented by ``geo_filter_ids``.

        The Census-compatible caller uses ``geo_filter_ids`` for the output
        geography/tract IDs. In a record-level variable tree, those IDs may
        be represented by a Hive partition (usually ``census_tract``), while
        ``record_col`` remains the join key.
        """
        if self.geography_partition is None:
            return record_col
        return self._quote_identifier(self.geography_partition)

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
