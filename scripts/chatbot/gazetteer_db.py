"""Open the gazetteer SpatiaLite database.

Tiny helper. Lives in this project so we don't have to import anything
from the v1 batch pipeline. The DB itself was copied wholesale from v1
during bootstrap; only the connection-open code is here.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# SpatiaLite extension name; works on Ubuntu after `apt install libsqlite3-mod-spatialite`.
SPATIALITE_EXT = "mod_spatialite"


def _try_pragma(conn: sqlite3.Connection, statement: str) -> None:
    """Apply an optional SQLite tuning pragma when the filesystem permits it."""
    try:
        conn.execute(statement)
    except sqlite3.OperationalError:
        # Windows/WSL/Docker mounts can reject WAL sidecar creation even when
        # the database itself is readable.  The gazetteer is read-only in the
        # app, so failing to apply these performance pragmas must not prevent
        # startup.
        pass


def open_spatialite(db_path: Path) -> sqlite3.Connection:
    """Open a read/write SpatiaLite DB connection.

    For chatbot reads only, but we don't restrict — leaves room for future
    in-process precompute or alias enrichment without re-plumbing.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(SPATIALITE_EXT)
    _try_pragma(conn, "PRAGMA journal_mode=WAL")
    _try_pragma(conn, "PRAGMA synchronous=NORMAL")
    _try_pragma(conn, "PRAGMA cache_size=-64000")   # 64 MB cache
    _try_pragma(conn, "PRAGMA temp_store=MEMORY")
    conn.row_factory = sqlite3.Row
    return conn
