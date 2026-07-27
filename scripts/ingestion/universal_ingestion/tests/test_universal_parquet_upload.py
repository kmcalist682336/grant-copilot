from pathlib import Path

import duckdb

from scripts.ingestion.universal_ingestion.preprocess_lar import (
    DEFAULT_CODEBOOK,
    DEFAULT_SCHEMA,
)
from scripts.ingestion.universal_ingestion.universal_parquet_upload import (
    default_part_name,
    merge_registry,
    normalize_dataset_name,
    prepare_source,
    safe_part_name,
    variable_id,
)


def test_hmda_variable_ids_remain_stable():
    assert variable_id("hmda", "income") == "c1aa5d4f3f72"


def test_registry_merge_preserves_existing_variables():
    result = merge_registry(
        {
            "dataset": "example",
            "variables": [
                {"variable_id": "old", "source_name": "existing"},
            ],
        },
        dataset="example",
        variables=["new_column"],
    )

    names = {entry["source_name"] for entry in result["variables"]}
    assert names == {"existing", "new_column"}


def test_dataset_and_part_names_are_path_safe(tmp_path):
    assert normalize_dataset_name("HMDA") == "hmda"
    assert safe_part_name(tmp_path / "Loans 2023 GA.csv", None) == (
        "part-loans-2023-ga.parquet"
    )
    assert default_part_name(
        "hmda", 2023, tmp_path / "hmda_lar_2023_GA.csv"
    ) == "part-hmda_lar_2023.parquet"


def test_raw_hmda_csv_is_decoded_before_splitting(tmp_path: Path):
    source = tmp_path / "hmda_lar_2023_GA.csv"
    source.write_text(
        "action_taken,county_code,income\n1,13121,75\n",
        encoding="utf-8",
    )

    with prepare_source(
        dataset="hmda",
        year=2023,
        source=source,
        codebook_path=DEFAULT_CODEBOOK,
        schema_path=DEFAULT_SCHEMA,
        chunk_size=10,
    ) as prepared:
        row = duckdb.sql(
            "SELECT action_taken, county_code, income "
            f"FROM read_parquet('{prepared.as_posix()}')"
        ).fetchone()

    assert row == ("Loan originated", "13121", 75.0)
