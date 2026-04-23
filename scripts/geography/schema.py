"""Gazetteer database schema and initialization.

Creates a SpatiaLite database with:
  - Administrative boundaries (states, counties, tracts, places)
  - Named places (neighborhoods, regions, corridors) — curated
  - Road corridors (interstates, US routes, state routes)
  - Alias/FTS table for fuzzy name matching
  - Pre-computed tract mappings for named places

All geometries stored in EPSG:4326 (WGS84).
Distance calculations use ST_Transform to EPSG:26916 (UTM Zone 16N, meters).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# SpatiaLite extension path — works on Ubuntu after apt-get install
SPATIALITE_EXT = "mod_spatialite"

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS admin_geographies (
    geoid           TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    geo_type        TEXT NOT NULL,
    state_fips      TEXT NOT NULL,
    county_fips     TEXT,
    place_fips      TEXT,
    lsad            TEXT,
    aland           REAL,
    population      INTEGER,
    api_for_clause  TEXT,
    api_in_clause   TEXT
);
CREATE INDEX IF NOT EXISTS idx_admin_geo_type
    ON admin_geographies (geo_type);
CREATE INDEX IF NOT EXISTS idx_admin_state_fips
    ON admin_geographies (state_fips, geo_type);
CREATE INDEX IF NOT EXISTS idx_admin_county_fips
    ON admin_geographies (state_fips, county_fips, geo_type);

CREATE TABLE IF NOT EXISTS named_places (
    place_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    place_type      TEXT NOT NULL,
    metro           TEXT,
    state_fips      TEXT NOT NULL DEFAULT '13',
    description     TEXT,
    source          TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_named_type ON named_places (place_type);
CREATE INDEX IF NOT EXISTS idx_named_metro ON named_places (metro);

CREATE TABLE IF NOT EXISTS roads (
    road_id              TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    full_name            TEXT,
    road_type            TEXT NOT NULL,
    state_fips           TEXT NOT NULL DEFAULT '13',
    default_buffer_miles REAL DEFAULT 2.0
);

CREATE TABLE IF NOT EXISTS place_tract_map (
    place_id        TEXT NOT NULL,
    tract_geoid     TEXT NOT NULL,
    county_geoid    TEXT NOT NULL,
    overlap_pct     REAL,
    overlap_type    TEXT,
    PRIMARY KEY (place_id, tract_geoid)
);
CREATE INDEX IF NOT EXISTS idx_ptm_place ON place_tract_map (place_id);
CREATE INDEX IF NOT EXISTS idx_ptm_tract ON place_tract_map (tract_geoid);

CREATE TABLE IF NOT EXISTS road_tract_map (
    road_id         TEXT NOT NULL,
    buffer_miles    REAL NOT NULL,
    tract_geoid     TEXT NOT NULL,
    county_geoid    TEXT NOT NULL,
    dist_meters     REAL,
    PRIMARY KEY (road_id, buffer_miles, tract_geoid)
);
CREATE INDEX IF NOT EXISTS idx_rtm_road ON road_tract_map (road_id, buffer_miles);

CREATE TABLE IF NOT EXISTS near_tract_map (
    anchor_type     TEXT NOT NULL,
    anchor_id       TEXT NOT NULL,
    buffer_miles    REAL NOT NULL,
    tract_geoid     TEXT NOT NULL,
    dist_meters     REAL NOT NULL,
    PRIMARY KEY (anchor_type, anchor_id, buffer_miles, tract_geoid)
);
CREATE INDEX IF NOT EXISTS idx_ntm_anchor
    ON near_tract_map (anchor_type, anchor_id, buffer_miles);

-- admin_place_tract_map: for each GA admin place (city), the tracts it
-- contains (area-overlap). Enables directional compound ops on admin
-- refs — "southwest Atlanta" = Atlanta's tracts filtered to SW quadrant.
CREATE TABLE IF NOT EXISTS admin_place_tract_map (
    admin_geoid     TEXT NOT NULL,
    tract_geoid     TEXT NOT NULL,
    county_geoid    TEXT NOT NULL,
    overlap_pct     REAL,
    overlap_type    TEXT,
    PRIMARY KEY (admin_geoid, tract_geoid)
);
CREATE INDEX IF NOT EXISTS idx_aptm_admin
    ON admin_place_tract_map (admin_geoid);

CREATE TABLE IF NOT EXISTS county_adjacency (
    county_geoid    TEXT NOT NULL,
    neighbor_geoid  TEXT NOT NULL,
    PRIMARY KEY (county_geoid, neighbor_geoid)
);
CREATE INDEX IF NOT EXISTS idx_county_adj_neighbor
    ON county_adjacency (neighbor_geoid);

CREATE TABLE IF NOT EXISTS geo_aliases (
    alias_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    alias           TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    target_type     TEXT NOT NULL,
    target_name     TEXT NOT NULL,
    alias_type      TEXT,
    state_fips      TEXT,
    confidence      REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_alias_target ON geo_aliases (target_id);
CREATE INDEX IF NOT EXISTS idx_alias_state  ON geo_aliases (state_fips);

CREATE VIRTUAL TABLE IF NOT EXISTS geo_fts USING fts5(
    alias,
    target_id,
    target_type,
    target_name,
    state_fips,
    content='geo_aliases',
    content_rowid='alias_id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS build_metadata (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Geometry column additions — run after tables exist
GEOMETRY_SQL = [
    ("SELECT AddGeometryColumn('admin_geographies', 'geom', 4326, 'MULTIPOLYGON', 'XY')"),
    ("SELECT CreateSpatialIndex('admin_geographies', 'geom')"),
    ("SELECT AddGeometryColumn('named_places', 'geom', 4326, 'MULTIPOLYGON', 'XY')"),
    ("SELECT CreateSpatialIndex('named_places', 'geom')"),
    ("SELECT AddGeometryColumn('roads', 'geom', 4326, 'MULTILINESTRING', 'XY')"),
    ("SELECT CreateSpatialIndex('roads', 'geom')"),
]

# FTS sync triggers
TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS alias_ai
    AFTER INSERT ON geo_aliases BEGIN
    INSERT INTO geo_fts(rowid, alias, target_id, target_type, target_name, state_fips)
    VALUES (new.alias_id, new.alias, new.target_id, new.target_type,
            new.target_name, new.state_fips);
END;

CREATE TRIGGER IF NOT EXISTS alias_ad
    AFTER DELETE ON geo_aliases BEGIN
    INSERT INTO geo_fts(geo_fts, rowid, alias, target_id, target_type,
                        target_name, state_fips)
    VALUES ('delete', old.alias_id, old.alias, old.target_id, old.target_type,
            old.target_name, old.state_fips);
END;
"""


def open_spatialite(db_path: Path) -> sqlite3.Connection:
    """Open a SpatiaLite database connection."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(SPATIALITE_EXT)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path, overwrite: bool = False) -> sqlite3.Connection:
    """Initialize a fresh SpatiaLite gazetteer database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        logger.warning("Removing existing database at %s", db_path)
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            extra = db_path.with_suffix(db_path.suffix + suffix)
            if extra.exists():
                extra.unlink()

    conn = open_spatialite(db_path)

    # Initialize SpatiaLite geometry tables
    if not _table_exists(conn, "spatial_ref_sys"):
        logger.info("Initializing SpatiaLite metadata tables")
        conn.execute("SELECT InitSpatialMetaData(1)")
        conn.commit()

    # Step 1: Create regular tables and indexes
    logger.info("Creating tables")
    for stmt in _split_schema(SCHEMA_DDL):
        conn.execute(stmt)
    conn.commit()

    # Step 2: Add geometry columns (errors if already exist — that's OK)
    logger.info("Adding geometry columns")
    for stmt in GEOMETRY_SQL:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                continue
            raise
    conn.commit()

    # Step 3: Create FTS triggers using executescript (handles
    # semicolons inside BEGIN...END blocks correctly)
    logger.info("Creating FTS triggers")
    try:
        conn.executescript(TRIGGER_SQL)
    except sqlite3.OperationalError as e:
        if "already exists" not in str(e).lower():
            raise
    conn.commit()

    logger.info("Database initialized at %s", db_path)
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def _split_schema(sql: str) -> list[str]:
    """Split schema SQL into individual statements for execution.

    Handles multi-line statements and inline comments correctly.
    Splits on semicolons that appear at end of a line (not inside strings).
    """
    import re
    # Split on semicolon followed by optional whitespace and a newline,
    # or semicolon at end of string
    parts = re.split(r';\s*(?:\n|$)', sql)
    result = []
    for part in parts:
        # Strip whitespace
        stmt = part.strip()
        if not stmt:
            continue
        # Skip if all non-empty lines are comments
        non_comment_lines = [
            line for line in stmt.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if non_comment_lines:
            result.append(stmt)
    return result


def set_build_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    from datetime import datetime, timezone
    conn.execute(
        "INSERT OR REPLACE INTO build_metadata (key, value, updated_at) "
        "VALUES (?, ?, ?)",
        (key, value, datetime.now(timezone.utc).isoformat()),
    )


def get_build_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    cur = conn.execute(
        "SELECT value FROM build_metadata WHERE key = ?", (key,)
    )
    row = cur.fetchone()
    return row[0] if row else None
