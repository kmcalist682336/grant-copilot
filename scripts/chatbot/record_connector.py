"""Construction helpers for the optional record-level DuckDB connector.

The orchestrator accepts a connector object rather than knowing how GCS
credentials or DuckDB extensions are configured. This keeps the Census path
unchanged and makes the same connector usable by the REPL, the web app, and
small smoke tools.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.duckdb_caller import DuckDBCaller

logger = logging.getLogger(__name__)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_record_caller(
    root: str,
    cache: APICache,
    *,
    record_id_column: str = "record_id",
    layout: str = "variable_tree",
    file_glob: str = "*.parquet",
    geography_partition: Optional[str] = None,
    access_id: Optional[str] = None,
    secret: Optional[str] = None,
) -> tuple[DuckDBCaller, object]:
    """Create a configured DuckDB caller and return ``(caller, connection)``.

    Local Parquet roots do not need credentials. GCS roots use HMAC
    credentials when supplied; if both are omitted, DuckDB attempts
    anonymous access for a genuinely public bucket.
    """
    import duckdb

    root = root.rstrip("/")
    con = duckdb.connect()
    if root.startswith("gs://"):
        access_id = access_id or os.getenv("GCS_HMAC_ACCESS_ID")
        secret = secret or os.getenv("GCS_HMAC_SECRET")
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        if access_id and secret:
            con.execute(
                "CREATE OR REPLACE SECRET record_gcs ("
                f"TYPE gcs, KEY_ID {_sql_string(access_id)}, "
                f"SECRET {_sql_string(secret)}, "
                f"SCOPE {_sql_string(root + '/')})"
            )
        elif access_id or secret:
            con.close()
            raise RuntimeError(
                "Set both GCS_HMAC_ACCESS_ID and GCS_HMAC_SECRET, or "
                "omit both for a public GCS bucket"
            )
        else:
            logger.warning(
                "No GCS HMAC credentials supplied; attempting anonymous "
                "DuckDB access for a public bucket"
            )

    caller = DuckDBCaller(
        con,
        cache,
        bucket_uri=root,
        geo_id_column=record_id_column,
        layout=layout,
        file_glob=file_glob,
        geography_partition=geography_partition,
        max_concurrent=1,
    )
    return caller, con


def close_record_connection(connection: object) -> None:
    """Close a DuckDB connection returned by ``build_record_caller``."""
    close = getattr(connection, "close", None)
    if close is not None:
        close()
