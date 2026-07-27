"""Resolve HMDA variable IDs to their column-separated Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = PROJECT_ROOT / "data" / "processed" / "variable_tree"
REGISTRY_PATH = ROOT / "dataset=hmda" / "variable_registry.json"
DEFAULT_REQUEST_PATH = (
    Path(__file__).resolve().parent / "config" / "hmda" / "request.json"
)


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, str]:
    """Load and validate the stable HMDA variable-ID registry."""
    with path.open(encoding="utf-8") as registry_file:
        document = json.load(registry_file)

    if document.get("dataset") != "hmda":
        raise ValueError("The variable registry is not for the HMDA dataset")

    registry: dict[str, str] = {}
    for entry in document.get("variables", []):
        variable_id = entry["variable_id"]
        if variable_id in registry:
            raise ValueError(f"Duplicate variable ID in registry: {variable_id}")
        registry[variable_id] = entry["source_name"]
    return registry


def resolve_files(
    request: dict,
    registry: dict[str, str] | None = None,
    *,
    root: Path = ROOT,
) -> dict:
    """Return a manifest for requested HMDA variable files."""
    dataset = request.get("dataset")
    year = request.get("year")
    variable_ids = request.get("variable_ids", [])

    if dataset != "hmda":
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not isinstance(year, int):
        raise ValueError("year must be an integer")
    if not variable_ids:
        raise ValueError("At least one variable ID is required")

    registry = registry or load_registry()
    missing_ids = [item for item in variable_ids if item not in registry]
    if missing_ids:
        raise ValueError(
            f"Variable IDs do not exist in the registry: {missing_ids}"
        )

    manifest = {
        "dataset": dataset,
        "year": year,
        "join_key": "hmda_record_id",
        "files": [],
    }
    for variable_id in variable_ids:
        path = (
            root
            / f"dataset={dataset}"
            / f"year={year}"
            / f"variable={variable_id}"
            / "part-000.parquet"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"No file found for variable {variable_id}: {path}"
            )
        try:
            display_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = path.as_posix()
        manifest["files"].append({
            "variable_id": variable_id,
            "source_name": registry[variable_id],
            "path": display_path,
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve an HMDA JSON request to Parquet paths."
    )
    parser.add_argument(
        "--request", type=Path, default=DEFAULT_REQUEST_PATH,
        help="Path to the resolver request JSON.",
    )
    args = parser.parse_args()

    with args.request.open(encoding="utf-8") as request_file:
        request = json.load(request_file)
    print(json.dumps(resolve_files(request), indent=2))


if __name__ == "__main__":
    main()
