from pathlib import Path

from scripts.ingestion.universal_ingestion.demo_hmda_query import resolve_files


def test_resolver_uses_dataset_year_variable_layout(tmp_path: Path):
    variable_id = "example123456"
    parquet_path = (
        tmp_path
        / "dataset=hmda"
        / "year=2023"
        / f"variable={variable_id}"
        / "part-000.parquet"
    )
    parquet_path.parent.mkdir(parents=True)
    parquet_path.touch()

    result = resolve_files(
        {
            "dataset": "hmda",
            "year": 2023,
            "variable_ids": [variable_id],
        },
        {variable_id: "income"},
        root=tmp_path,
    )

    assert "geography" not in result
    assert result["files"][0]["path"].endswith(
        "dataset=hmda/year=2023/"
        f"variable={variable_id}/part-000.parquet"
    )
