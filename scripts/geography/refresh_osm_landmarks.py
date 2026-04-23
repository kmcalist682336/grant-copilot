"""Regenerate georgia_landmarks.geojson from the full OSM pbf.

Why this exists:
  The existing georgia_landmarks.geojson is a points-only export that
  excludes landmarks tagged as ways / relations (polygons). That leaves
  gaps like Piedmont Park, Centennial Olympic Park, Mercedes-Benz
  Stadium, Hartsfield-Jackson, and Grady Memorial Hospital — all of
  which exist as polygons in OSM.

Pipeline:
  1. ``osmium tags-filter``  — extract nodes/ways/relations matching
     landmark-relevant tag combinations into a small pbf.
  2. ``osmium export``       — materialize way/relation geometries and
     emit GeoJSON with both Point and (Multi)Polygon features.

Usage:
  python -m scripts.geography.refresh_osm_landmarks

Writes to ``data/geo/raw/osm/georgia_landmarks_v2.geojson``. The loader
in ``load_all_sources.load_osm_landmarks`` will prefer this file if it
exists, falling back to the original ``georgia_landmarks.geojson``.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Tag filter expressions used by ``osmium tags-filter``.
# Syntax: "n/KEY=VALUE[,VALUE2]" = node,way,relation scope prefix + key/values
# We include nodes/ways/relations for most categories since a landmark might
# be represented as any of the three in OSM.
TAG_FILTERS = [
    # Parks, green spaces
    "nwr/leisure=park,garden,nature_reserve",
    # Transit
    "nwr/aeroway=aerodrome,terminal",
    # Healthcare
    "nwr/amenity=hospital,clinic",
    # Education
    "nwr/amenity=university,college,school",
    # Culture / recreation
    "nwr/amenity=arts_centre,theatre,library,community_centre",
    "nwr/tourism=museum,attraction,zoo,aquarium",
    # Stadiums
    "nwr/leisure=stadium",
    "nwr/building=stadium",
    "nwr/sport=american_football,basketball,baseball,soccer",
    # Civic
    "nwr/amenity=courthouse,townhall,prison,police,fire_station",
    # Religious (larger landmarks only)
    "nwr/amenity=place_of_worship",
    # Historic / tourism
    "nwr/historic=district,monument,memorial,castle,fort,church",
    # Neighbourhoods named as places
    "nwr/place=neighbourhood,suburb,quarter,city_block",
    # Shopping landmarks
    "nwr/shop=mall",
    # Military (rare but needed for Fort X queries)
    "nwr/landuse=military",
]


def _require_osmium() -> str:
    path = shutil.which("osmium")
    if not path:
        raise SystemExit(
            "osmium not found on PATH. Install with `sudo apt install osmium-tool`."
        )
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def refresh_landmarks(
    pbf_path: Path,
    output_path: Path,
    scratch_dir: Path | None = None,
    keep_scratch: bool = False,
) -> dict:
    """Run the two-step osmium pipeline and write final GeoJSON.

    Returns a summary dict with counts by geometry type.
    """
    _require_osmium()
    if not pbf_path.exists():
        raise FileNotFoundError(f"OSM pbf not found at {pbf_path}")

    scratch_dir = scratch_dir or (output_path.parent / "_osm_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    filtered_pbf = scratch_dir / "landmarks_filtered.osm.pbf"

    logger.info("Filtering landmark tags (%d filters) …", len(TAG_FILTERS))
    cmd_filter = [
        "osmium", "tags-filter",
        str(pbf_path),
        *TAG_FILTERS,
        "-o", str(filtered_pbf),
        "--overwrite",
    ]
    subprocess.run(cmd_filter, check=True)
    logger.info(
        "Filtered pbf: %.1f MB",
        filtered_pbf.stat().st_size / (1024 * 1024),
    )

    logger.info("Exporting to GeoJSON (materializing polygon geometries) …")
    cmd_export = [
        "osmium", "export",
        str(filtered_pbf),
        "-f", "geojson",
        "-o", str(output_path),
        "--overwrite",
    ]
    subprocess.run(cmd_export, check=True)
    logger.info(
        "Wrote %s (%.1f MB)",
        output_path, output_path.stat().st_size / (1024 * 1024),
    )

    # Summarize
    with output_path.open() as f:
        data = json.load(f)
    features = data["features"]
    by_geom: dict[str, int] = {}
    for feat in features:
        gtype = (feat.get("geometry") or {}).get("type", "none")
        by_geom[gtype] = by_geom.get(gtype, 0) + 1

    if not keep_scratch:
        try:
            filtered_pbf.unlink()
            scratch_dir.rmdir()
        except OSError:
            pass

    return {"total": len(features), "by_geometry": by_geom}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pbf",
        type=Path,
        default=_repo_root() / "data" / "geo" / "raw" / "osm" / "georgia-latest.osm.pbf",
        help="Source OSM pbf file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "data" / "geo" / "raw" / "osm" / "georgia_landmarks_v2.geojson",
        help="Destination GeoJSON path",
    )
    parser.add_argument(
        "--keep-scratch", action="store_true",
        help="Keep intermediate filtered pbf (for debugging)",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    summary = refresh_landmarks(
        args.pbf, args.output,
        keep_scratch=args.keep_scratch,
    )
    logger.info("=" * 60)
    logger.info("OSM landmarks refresh complete")
    logger.info("  total features: %d", summary["total"])
    for gtype, n in sorted(
        summary["by_geometry"].items(), key=lambda x: -x[1]
    ):
        logger.info("    %-20s %d", gtype, n)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
