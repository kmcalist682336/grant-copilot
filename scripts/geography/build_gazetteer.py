"""Build the gazetteer database from scratch.

Downloads Census TIGER/Line shapefiles, loads road geometries from OSM,
builds alias tables, and pre-computes spatial join mappings.

This is a one-time build script. Re-run when:
  - Census releases new TIGER/Line data (every year or decennial)
  - You add new neighborhoods or curated named places
  - You expand to new states

Usage:
    # Full build from scratch
    python -m scripts.geography.build_gazetteer

    # Skip downloads (use cached files)
    python -m scripts.geography.build_gazetteer --no-download

    # Skip road fetch from Overpass (use local GeoJSON fallback)
    python -m scripts.geography.build_gazetteer --no-roads

    # Skip expensive spatial pre-computation (for quick testing)
    python -m scripts.geography.build_gazetteer --no-precompute

    # Overwrite existing database
    python -m scripts.geography.build_gazetteer --overwrite
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.geography.schema import (
    init_database, set_build_metadata, open_spatialite,
)
from scripts.geography.tiger_loader import (
    download_tiger_files,
    load_states, load_counties, load_tracts, load_places,
)
from scripts.geography.alias_builder import (
    build_admin_aliases,
    load_curated_county_aliases,
    load_curated_place_aliases,
    load_road_aliases,
    load_named_place_aliases,
    rebuild_fts_index,
)
from scripts.geography.road_loader import (
    load_roads_from_overpass,
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the gazetteer SpatiaLite database"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip downloading TIGER files (use cached copies)",
    )
    parser.add_argument(
        "--no-roads", action="store_true",
        help="Skip fetching road geometries from OpenStreetMap Overpass API",
    )
    parser.add_argument(
        "--no-precompute", action="store_true",
        help="Skip spatial pre-computation of place and road tract mappings",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing database (default: update in place)",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Path to output database (default: data/geo/gazetteer.db)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory for downloaded files (default: data/geo/raw)",
    )
    parser.add_argument(
        "--roads", nargs="*", default=None,
        help="Specific road IDs to fetch (default: all). E.g. --roads I-85 I-20",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_root = _repo_root()
    db_path = Path(args.db_path) if args.db_path else (
        repo_root / "data" / "geo" / "gazetteer.db"
    )
    data_dir = Path(args.data_dir) if args.data_dir else (
        repo_root / "data" / "geo" / "raw"
    )

    logger.info("=" * 60)
    logger.info("GAZETTEER BUILD")
    logger.info("  Database: %s", db_path)
    logger.info("  Data dir: %s", data_dir)
    logger.info("=" * 60)

    start = time.time()

    # ── Step 1: Initialize database ─────────────────────────────────────
    logger.info("[1/7] Initializing database")
    conn = init_database(db_path, overwrite=args.overwrite)

    # ── Step 2: Download TIGER/Line files ────────────────────────────────
    if not args.no_download:
        logger.info("[2/7] Downloading TIGER/Line shapefiles")
        tiger_files = download_tiger_files(data_dir, force=False)
    else:
        logger.info("[2/7] Skipping download — using cached files")
        tiger_dir = data_dir / "tiger"
        tiger_files = {p.name: p for p in tiger_dir.glob("*.zip")}

    # ── Step 3: Load administrative boundaries ───────────────────────────
    logger.info("[3/7] Loading administrative boundaries")

    if "tl_2020_us_state.zip" in tiger_files:
        n = load_states(conn, tiger_files["tl_2020_us_state.zip"])
        logger.info("  States: %d", n)

    if "tl_2020_us_county.zip" in tiger_files:
        n = load_counties(conn, tiger_files["tl_2020_us_county.zip"])
        logger.info("  Counties: %d", n)

    if "tl_2020_13_tract.zip" in tiger_files:
        n = load_tracts(conn, tiger_files["tl_2020_13_tract.zip"])
        logger.info("  Georgia tracts: %d", n)

    if "tl_2020_13_place.zip" in tiger_files:
        n = load_places(conn, tiger_files["tl_2020_13_place.zip"])
        logger.info("  Georgia places: %d", n)

    conn.commit()

    # ── Step 4: Load road geometries ──────────────────────────────────────
    if not args.no_roads:
        logger.info("[4/7] Fetching road geometries from OpenStreetMap Overpass API")
        logger.info("  (This takes 2-3 minutes — being polite to Overpass API)")
        road_ids = args.roads  # None = all roads
        results = load_roads_from_overpass(conn, road_ids=road_ids)
        ok = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        logger.info("  Roads loaded: %d OK, %d failed", ok, failed)
        if failed > 0:
            failed_roads = [k for k, v in results.items() if not v]
            logger.warning("  Failed roads: %s", failed_roads)
    else:
        logger.info("[4/7] Skipping road geometry fetch")

    # ── Step 5: Build aliases ──────────────────────────────────────────────
    logger.info("[5/7] Building alias tables")

    n = build_admin_aliases(conn)
    logger.info("  Generated admin aliases: %d", n)

    n = load_curated_county_aliases(conn)
    logger.info("  Curated county aliases: %d", n)

    n = load_curated_place_aliases(conn)
    logger.info("  Curated place aliases: %d", n)

    n = load_road_aliases(conn)
    logger.info("  Road aliases: %d", n)

    n = load_named_place_aliases(conn)
    logger.info("  Named place aliases: %d", n)

    rebuild_fts_index(conn)

    # ── Step 6: Pre-compute spatial mappings ──────────────────────────────
    if not args.no_precompute:
        logger.info("[6/7] Pre-computing spatial join mappings")
        logger.info("  (This may take 10-30 minutes for Georgia tracts)")

        from scripts.geography.spatial_precompute import (
            compute_place_tract_mappings,
            compute_road_tract_mappings,
        )

        n = compute_place_tract_mappings(conn)
        logger.info("  Place→tract mappings: %d", n)

        n = compute_road_tract_mappings(conn)
        logger.info("  Road→tract mappings: %d", n)
    else:
        logger.info("[6/7] Skipping spatial pre-computation")

    # ── Step 7: Record build metadata ────────────────────────────────────
    logger.info("[7/7] Recording build metadata")
    set_build_metadata(conn, "build_timestamp",
                       datetime.now(timezone.utc).isoformat())
    set_build_metadata(conn, "tiger_year", "2020")
    set_build_metadata(conn, "states_loaded", "US")
    set_build_metadata(conn, "tracts_loaded", "GA")
    set_build_metadata(conn, "roads_loaded", "GA")
    conn.commit()

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start

    # Print stats
    stats = {}
    for table in ["admin_geographies", "named_places", "roads",
                  "geo_aliases", "place_tract_map", "road_tract_map"]:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = n
        except Exception:
            stats[table] = 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("GAZETTEER BUILD COMPLETE (%.1f seconds)", elapsed)
    logger.info("")
    logger.info("  Database: %s", db_path)
    logger.info("  Size:     %.1f MB", db_path.stat().st_size / 1e6
                if db_path.exists() else 0)
    logger.info("")
    logger.info("  Rows loaded:")
    for table, count in stats.items():
        logger.info("    %-30s %d", table, count)
    logger.info("=" * 60)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
