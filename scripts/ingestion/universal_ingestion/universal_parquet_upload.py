"""Split a tabular source by variable and upload the Parquets to GCS.

The physical object layout is:

    <prefix>/table_id=<dataset>/year=<year>/
        census_tract=<11-digit-geoid>/
            variable=<stable-variable-id>/<part-name>.parquet

Every variable file contains a common record ID and one ``value`` column.
HMDA is partitioned by census tract by default. Blank tract values are retained
under ``census_tract=unknown`` rather than being discarded.
The command is a dry run unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import duckdb

from scripts.ingestion.universal_ingestion.preprocess_lar import (
    DEFAULT_CODEBOOK,
    DEFAULT_SCHEMA,
    decode_csv_to_parquet,
)


_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def sql_string(value: str) -> str:
    """Escape a value as a DuckDB string literal."""
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    """Quote a SQL identifier, including punctuation and spaces."""
    return '"' + value.replace('"', '""') + '"'


def normalize_dataset_name(value: str) -> str:
    """Validate the dataset identifier used in IDs and object paths."""
    normalized = value.strip().lower()
    if not _SAFE_NAME.fullmatch(normalized):
        raise ValueError(
            "dataset must contain only lowercase letters, numbers, "
            "underscores, or hyphens"
        )
    return normalized


def variable_id(dataset: str, source_name: str) -> str:
    """Return a stable opaque variable ID scoped to one dataset."""
    canonical = f"{dataset}:{source_name.strip().lower()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def source_scan_sql(source: Path) -> str:
    """Return the appropriate DuckDB scan expression for a source file."""
    path = sql_string(source.resolve().as_posix())
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return f"read_parquet({path}, hive_partitioning = false)"
    if suffix in {".csv", ".txt"}:
        return f"read_csv_auto({path}, all_varchar = true)"
    raise ValueError("source must be a CSV, TXT, or Parquet file")


@contextmanager
def prepare_source(
    *,
    dataset: str,
    year: int,
    source: Path,
    codebook_path: Path,
    schema_path: Path,
    chunk_size: int,
) -> Iterator[Path]:
    """Apply a registered dataset-specific transform before splitting.

    HMDA CSVs are decoded and normalized with the official codebook/schema.
    Other CSVs and existing Parquets pass through unchanged.
    """
    if dataset == "hmda" and source.suffix.lower() == ".csv":
        with tempfile.TemporaryDirectory(prefix="hmda-upload-decoded-") as temp_dir:
            decoded_path = Path(temp_dir) / f"{source.stem}.decoded.parquet"
            print("Preparing raw HMDA CSV with the HMDA decoder...")
            decode_csv_to_parquet(
                source,
                decoded_path,
                year=year,
                codebook_path=codebook_path,
                schema_path=schema_path,
                chunk_size=chunk_size,
            )
            yield decoded_path
        return
    yield source


def safe_part_name(source: Path, requested: str | None) -> str:
    """Return a collision-resistant Parquet part filename."""
    raw = requested or f"part-{source.stem}"
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(raw).stem).strip("-").lower()
    if not stem:
        raise ValueError("part name cannot be empty")
    return f"{stem}.parquet"


def default_part_name(dataset: str, year: int, source: Path) -> str | None:
    """Return a dataset-specific default without temporary/source partitions."""
    if dataset == "hmda":
        return f"part-hmda_lar_{year}.parquet"
    return None


def merge_registry(
    existing: dict[str, Any] | None,
    *,
    dataset: str,
    variables: list[str],
) -> dict[str, Any]:
    """Merge newly observed variables into a durable registry document."""
    entries: dict[str, str] = {}
    if existing:
        existing_dataset = existing.get("dataset")
        if existing_dataset not in {None, dataset}:
            raise ValueError(
                f"Remote registry is for {existing_dataset!r}, not {dataset!r}"
            )
        for entry in existing.get("variables", []):
            entries[entry["variable_id"]] = entry["source_name"]

    for source_name in variables:
        opaque_id = variable_id(dataset, source_name)
        previous = entries.get(opaque_id)
        if previous is not None and previous != source_name:
            raise ValueError(
                f"Variable ID collision: {opaque_id} maps to "
                f"{previous!r} and {source_name!r}"
            )
        entries[opaque_id] = source_name

    return {
        "dataset": dataset,
        "id_strategy": f"sha256({dataset}:<source_name>)[:12]",
        "variables": [
            {"variable_id": opaque_id, "source_name": source_name}
            for opaque_id, source_name in sorted(
                entries.items(), key=lambda item: item[1]
            )
        ],
    }


def split_and_upload(
    *,
    dataset: str,
    year: int,
    source: Path,
    bucket_name: str,
    prefix: str,
    record_id_column: str | None,
    census_tract_column: str | None,
    part_name: str | None,
    project: str | None,
    execute: bool,
) -> int:
    """Split one source into variable files and optionally upload them."""
    dataset = normalize_dataset_name(dataset)
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")
    if not bucket_name.strip():
        raise ValueError("bucket cannot be empty")

    prefix = prefix.strip("/")
    table_prefix = "/".join(
        part for part in (prefix, f"table_id={dataset}") if part
    )
    scan = source_scan_sql(source)
    parquet_name = safe_part_name(source, part_name)
    connection = duckdb.connect()

    try:
        described = connection.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
        source_columns = [row[0] for row in described]
        if record_id_column and record_id_column not in source_columns:
            raise ValueError(
                f"record ID column {record_id_column!r} is not in the source"
            )
        if (
            census_tract_column
            and census_tract_column not in source_columns
        ):
            raise ValueError(
                f"census tract column {census_tract_column!r} is not in the source"
            )
        variables = [
            column for column in source_columns if column != record_id_column
        ]
        if not variables:
            raise ValueError("Source contains no variables to upload")

        print(f"Source: {source}")
        print(f"Dataset: {dataset}")
        print(f"Year: {year}")
        print(f"Variables: {len(variables)}")
        print(f"GCS root: gs://{bucket_name}/{table_prefix}/year={year}/")
        if census_tract_column:
            tract = sql_identifier(census_tract_column)
            invalid_tracts = connection.execute(
                f"""
                SELECT DISTINCT CAST({tract} AS VARCHAR)
                FROM {scan}
                WHERE {tract} IS NOT NULL
                  AND trim(CAST({tract} AS VARCHAR)) <> ''
                  AND NOT regexp_full_match(
                      trim(CAST({tract} AS VARCHAR)), '[0-9]{{1,11}}'
                  )
                LIMIT 10
                """
            ).fetchall()
            if invalid_tracts:
                values = ", ".join(repr(row[0]) for row in invalid_tracts)
                raise ValueError(
                    "census tract values must contain 1-11 digits; "
                    f"invalid examples: {values}"
                )
            tract_count = connection.execute(
                f"""
                SELECT COUNT(DISTINCT CASE
                    WHEN {tract} IS NULL
                      OR trim(CAST({tract} AS VARCHAR)) = ''
                    THEN 'unknown'
                    ELSE lpad(trim(CAST({tract} AS VARCHAR)), 11, '0')
                END)
                FROM {scan}
                """
            ).fetchone()[0]
            print(
                f"Census tract partitions: {tract_count:,} "
                "(blank values use census_tract=unknown)"
            )
        for variable in variables[:10]:
            opaque_id = variable_id(dataset, variable)
            print(
                f"  {variable} ({opaque_id}) -> "
                f"census_tract=<geoid>/variable={opaque_id}/{parquet_name}"
            )
        if len(variables) > 10:
            print(f"  ... and {len(variables) - 10} more variables")
        if not execute:
            print("Dry run only; pass --execute to create and upload files.")
            return len(variables)

        try:
            from google.cloud import storage
        except ImportError as error:
            raise RuntimeError(
                "GCS upload requires google-cloud-storage: "
                "python -m pip install google-cloud-storage"
            ) from error

        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        registry_object = f"{table_prefix}/variable_registry.json"
        registry_blob = bucket.blob(registry_object)
        existing_registry = None
        if registry_blob.exists(client=client):
            existing_registry = json.loads(registry_blob.download_as_text())
        registry = merge_registry(
            existing_registry,
            dataset=dataset,
            variables=variables,
        )

        connection.execute(
            f"""
            CREATE TEMP TABLE source_rows AS
            SELECT row_number() OVER () - 1 AS _source_row_number, *
            FROM {scan}
            """
        )
        generated_id_prefix = f"{dataset}:{year}:{source.stem}:"

        with tempfile.TemporaryDirectory(prefix="variable-upload-") as temp_dir:
            temporary_root = Path(temp_dir)
            for index, variable in enumerate(variables, start=1):
                opaque_id = variable_id(dataset, variable)
                if record_id_column:
                    record_id_sql = (
                        f"CAST({sql_identifier(record_id_column)} AS VARCHAR)"
                    )
                else:
                    record_id_sql = (
                        "concat("
                        f"{sql_string(generated_id_prefix)}, "
                        "lpad(CAST(_source_row_number AS VARCHAR), 12, '0'))"
                    )
                output_record_id = record_id_column or f"{dataset}_record_id"
                if census_tract_column:
                    tract = sql_identifier(census_tract_column)
                    variable_root = temporary_root / opaque_id
                    connection.execute(
                        f"""
                        COPY (
                            SELECT
                                {record_id_sql} AS {sql_identifier(output_record_id)},
                                {sql_identifier(variable)} AS value,
                                CASE
                                    WHEN {tract} IS NULL
                                      OR trim(CAST({tract} AS VARCHAR)) = ''
                                    THEN 'unknown'
                                    ELSE lpad(
                                        trim(CAST({tract} AS VARCHAR)), 11, '0'
                                    )
                                END AS _census_tract_partition
                            FROM source_rows
                        )
                        TO {sql_string(variable_root.as_posix())}
                        (
                            FORMAT PARQUET,
                            COMPRESSION ZSTD,
                            PARTITION_BY (_census_tract_partition)
                        )
                        """
                    )
                    partition_files = sorted(variable_root.rglob("*.parquet"))
                    if not partition_files:
                        raise RuntimeError(
                            f"No tract partitions were created for {variable!r}"
                        )
                    for local_path in partition_files:
                        partition_folder = local_path.parent.name
                        _, separator, tract_value = partition_folder.partition("=")
                        if not separator or not tract_value:
                            raise RuntimeError(
                                "Unexpected DuckDB partition directory: "
                                f"{partition_folder}"
                            )
                        object_name = (
                            f"{table_prefix}/year={year}/"
                            f"census_tract={tract_value}/"
                            f"variable={opaque_id}/{parquet_name}"
                        )
                        bucket.blob(object_name).upload_from_filename(
                            local_path,
                            content_type="application/vnd.apache.parquet",
                        )
                else:
                    local_path = temporary_root / f"{opaque_id}.parquet"
                    connection.execute(
                        f"""
                        COPY (
                            SELECT
                                {record_id_sql} AS {sql_identifier(output_record_id)},
                                {sql_identifier(variable)} AS value
                            FROM source_rows
                        )
                        TO {sql_string(local_path.as_posix())}
                        (FORMAT PARQUET, COMPRESSION ZSTD)
                        """
                    )
                    object_name = (
                        f"{table_prefix}/year={year}/"
                        f"variable={opaque_id}/{parquet_name}"
                    )
                    bucket.blob(object_name).upload_from_filename(
                        local_path,
                        content_type="application/vnd.apache.parquet",
                    )
                print(
                    f"[{index}/{len(variables)}] Uploaded {variable} -> "
                    f"{tract_count if census_tract_column else 1} file(s)"
                )

        registry_blob.upload_from_string(
            json.dumps(registry, indent=2) + "\n",
            content_type="application/json",
        )
        print(f"Registry: gs://{bucket_name}/{registry_object}")
        return len(variables)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a CSV or Parquet by variable and upload it to GCS."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bucket", required=True, help="Bucket name without gs://")
    parser.add_argument(
        "--prefix", default="variable_tree",
        help="Optional object prefix before table_id=<name>.",
    )
    parser.add_argument(
        "--record-id-column",
        help="Existing unique record-ID column; otherwise one is generated.",
    )
    parser.add_argument(
        "--census-tract-column",
        help=(
            "Column used for census-tract object partitions. For HMDA, "
            "this defaults to census_tract."
        ),
    )
    parser.add_argument(
        "--part-name",
        help="Optional object filename; defaults to part-<source-stem>.parquet.",
    )
    parser.add_argument("--project", help="Optional Google Cloud project ID.")
    parser.add_argument(
        "--codebook", type=Path, default=DEFAULT_CODEBOOK,
        help="HMDA codebook used when the source is a raw HMDA CSV.",
    )
    parser.add_argument(
        "--schema", type=Path, default=DEFAULT_SCHEMA,
        help="HMDA numeric schema used when the source is a raw HMDA CSV.",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create and upload files; otherwise show a dry run.",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    part_name = args.part_name or default_part_name(
        dataset, args.year, args.source,
    )
    census_tract_column = args.census_tract_column
    if dataset == "hmda" and census_tract_column is None:
        census_tract_column = "census_tract"
    with prepare_source(
        dataset=dataset,
        year=args.year,
        source=args.source,
        codebook_path=args.codebook,
        schema_path=args.schema,
        chunk_size=args.chunk_size,
    ) as prepared_source:
        split_and_upload(
            dataset=dataset,
            year=args.year,
            source=prepared_source,
            bucket_name=args.bucket,
            prefix=args.prefix,
            record_id_column=args.record_id_column,
            census_tract_column=census_tract_column,
            part_name=part_name,
            project=args.project,
            execute=args.execute,
        )


if __name__ == "__main__":
    main()
