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


def open_spatialite(db_path: Path) -> sqlite3.Connection:
    """Open a read/write SpatiaLite DB connection.

    For chatbot reads only, but we don't restrict — leaves room for future
    in-process precompute or alias enrichment without re-plumbing.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(SPATIALITE_EXT)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64 MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.row_factory = sqlite3.Row
    return conn
