import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/variable_tree")
REGISTRY_PATH = ROOT / "dataset=hmda" / "variable_registry.csv"


with open("scripts/ingestion/universal_ingestion/config/hmda/request.json", "r", encoding="utf-8") as request_file:
    request = json.load(request_file)


dataset = request.get("dataset")
year = request.get("year")
geography = request.get("geography")
variable_ids = request.get("variable_ids", [])


if dataset != "hmda":
    raise ValueError(f"Unsupported dataset: {dataset}")

if year != 2024:
    raise ValueError("The demo currently supports only 2024")

if geography != "state-GA":
    raise ValueError("The demo currently supports only state-GA")

if not variable_ids:
    raise ValueError("At least one variable ID is required")


# Load the mapping between hashes and readable HMDA variable names.
registry_frame = pd.read_csv(
    REGISTRY_PATH,
    dtype=str,
)

if registry_frame["variable_id"].duplicated().any():
    raise ValueError("The variable registry contains duplicate IDs")

registry = dict(
    zip(
        registry_frame["variable_id"],
        registry_frame["source_name"],
    )
)


# Confirm that every requested variable ID exists.
missing_ids = [
    variable_id
    for variable_id in variable_ids
    if variable_id not in registry
]

if missing_ids:
    raise ValueError(
        f"Variable IDs do not exist in the registry: {missing_ids}"
    )


def resolve_files(request, registry):
    dataset = request["dataset"]
    year = request["year"]
    geography = request["geography"]
    variable_ids = request["variable_ids"]

    manifest = {
        "dataset": dataset,
        "year": year,
        "geography": geography,
        "join_key": "hmda_record_id",
        "files": [],
    }

    for variable_id in variable_ids:
        path = (
            ROOT
            / f"dataset={dataset}"
            / f"year={year}"
            / f"geography={geography}"
            / f"variable={variable_id}"
            / "part-000.parquet"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"No file found for variable {variable_id}: {path}"
            )

        manifest["files"].append({
            "variable_id": variable_id,
            "source_name": registry[variable_id],
            "path": path.as_posix(),
        })

    return manifest


result = resolve_files(request, registry)

print(json.dumps(result, indent=2))