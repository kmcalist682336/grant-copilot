"""Split a tabular source by variable and upload the Parquets to GCS.

The default physical object layout is:

    <prefix>/table_id=<dataset>/year=<year>/
        variable=<stable-variable-id>/<part-name>.parquet

Every variable file contains a common record ID, one ``value`` column, and,
for HMDA, the normalized ``census_tract`` column. An optional tract Hive
partition can be enabled with ``--census-tract-column``.
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
DEFAULT_HMDA_UNITS = (
    Path(__file__).parent / "config" / "hmda" / "hmda_units.json"
)


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
        return f"hmda_{year}.parquet"
    return None


def load_units(path: Path | None, *, dataset: str, year: int) -> dict[str, dict[str, Any]]:
    """Load year-appropriate canonical-unit metadata for source variables.

    Units are intentionally separate from the HMDA codebook: the codebook
    decodes categorical values, while this registry describes numeric scale
    and presentation units.  Variables without an entry pass through
    unchanged.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Unit registry not found: {path}")
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    registry_dataset = document.get("dataset")
    if registry_dataset not in {None, dataset}:
        raise ValueError(
            f"Unit registry is for {registry_dataset!r}, not {dataset!r}"
        )

    output: dict[str, dict[str, Any]] = {}
    for source_name, raw_metadata in document.get("variables", {}).items():
        if not isinstance(raw_metadata, dict):
            raise ValueError(
                f"Unit metadata for {source_name!r} must be an object"
            )
        start = int(raw_metadata.get("year_start", 0))
        end = int(raw_metadata.get("year_end", 9999))
        if not start <= year <= end:
            continue
        scale = float(raw_metadata.get("scale_factor", 1))
        if scale <= 0:
            raise ValueError(
                f"scale_factor for {source_name!r} must be positive"
            )
        output[source_name] = {
            "source_unit": str(raw_metadata.get("source_unit", "")),
            "canonical_unit": str(raw_metadata.get("canonical_unit", "")),
            "scale_factor": scale,
            "year_start": start,
            "year_end": end,
        }
    return output


def merge_registry(
    existing: dict[str, Any] | None,
    *,
    dataset: str,
    variables: list[str],
    units: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge newly observed variables into a durable registry document."""
    entries: dict[str, dict[str, Any]] = {}
    units = units or {}
    if existing:
        existing_dataset = existing.get("dataset")
        if existing_dataset not in {None, dataset}:
            raise ValueError(
                f"Remote registry is for {existing_dataset!r}, not {dataset!r}"
            )
        for entry in existing.get("variables", []):
            entries[entry["variable_id"]] = dict(entry)

    for source_name in variables:
        opaque_id = variable_id(dataset, source_name)
        entry = entries.get(opaque_id)
        previous = entry.get("source_name") if entry else None
        if previous is not None and previous != source_name:
            raise ValueError(
                f"Variable ID collision: {opaque_id} maps to "
                f"{previous!r} and {source_name!r}"
            )
        entry = entries.setdefault(
            opaque_id,
            {"variable_id": opaque_id, "source_name": source_name},
        )
        metadata = units.get(source_name)
        if metadata:
            entry.update(metadata)

    return {
        "dataset": dataset,
        "id_strategy": f"sha256({dataset}:<source_name>)[:12]",
        "variables": [
            entry
            for _, entry in sorted(
                entries.items(), key=lambda item: item[1]["source_name"]
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
    geography_column: str | None,
    part_name: str | None,
    project: str | None,
    execute: bool,
    units: dict[str, dict[str, Any]] | None = None,
    apply_unit_conversion: bool = True,
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
        # ``record_id_column`` is also the desired output name when the
        # source has no existing ID column.  This is important for raw HMDA
        # CSVs: the decoder supplies row numbers, but does not add a source
        # column named ``record_id``.  If the named column exists, preserve
        # its values; otherwise generate stable IDs under the requested name.
        source_record_id_column = (
            record_id_column if record_id_column in source_columns else None
        )
        if (
            census_tract_column
            and census_tract_column not in source_columns
        ):
            raise ValueError(
                f"census tract column {census_tract_column!r} is not in the source"
            )
        if geography_column and geography_column not in source_columns:
            raise ValueError(
                f"geography column {geography_column!r} is not in the source"
            )
        variables = [
            column for column in source_columns
            if column != source_record_id_column
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
            object_suffix = (
                f"census_tract=<geoid>/variable={opaque_id}/{parquet_name}"
                if census_tract_column
                else f"variable={opaque_id}/{parquet_name}"
            )
            print(f"  {variable} ({opaque_id}) -> {object_suffix}")
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
            units=units,
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
                metadata = (
                    (units or {}).get(variable, {})
                    if apply_unit_conversion else {}
                )
                scale_factor = float(metadata.get("scale_factor", 1))
                value_expression = sql_identifier(variable)
                if scale_factor != 1:
                    # Unit conversion is deterministic and happens before
                    # upload. Raw CSVs and the decoder output remain intact.
                    value_expression = (
                        f"CAST({value_expression} AS DOUBLE) * "
                        f"{scale_factor:g}"
                    )
                geography_sql = None
                if geography_column:
                    geography_source = sql_identifier(geography_column)
                    geography_sql = f"CASE\n"
                    geography_sql += (
                        f" WHEN {geography_source} IS NULL OR "
                        f"trim(CAST({geography_source} AS VARCHAR)) = '' OR "
                        f"NOT regexp_full_match(trim(CAST({geography_source} AS VARCHAR)), '[0-9]{{1,11}}')\n"
                        " THEN NULL\n"
                        f" ELSE lpad(trim(CAST({geography_source} AS VARCHAR)), 11, '0')\n"
                        "END"
                    )
                if source_record_id_column:
                    record_id_sql = (
                        f"CAST({sql_identifier(source_record_id_column)} AS VARCHAR)"
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
                                {value_expression} AS value,
                                {geography_sql or 'NULL'} AS census_tract,
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
                    geography_select = (
                        f", {geography_sql} AS census_tract"
                        if geography_sql else ""
                    )
                    connection.execute(
                        f"""
                        COPY (
                            SELECT
                                {record_id_sql} AS {sql_identifier(output_record_id)},
                                {value_expression} AS value
                                {geography_select}
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
            "Optional column used for census-tract object partitions. "
            "Leave unset to keep census_tract as a regular Parquet column."
        ),
    )
    parser.add_argument(
        "--geography-column",
        help=(
            "Geography column to retain in each variable Parquet. For HMDA, "
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
    parser.add_argument(
        "--units-registry", type=Path,
        help=(
            "JSON registry of source/canonical units and scale factors. "
            "Defaults to the HMDA registry when --dataset hmda."
        ),
    )
    parser.add_argument(
        "--values-already-canonical",
        action="store_true",
        help=(
            "Do not apply scale factors because the input values are already "
            "in canonical units. Use this when re-uploading converted Parquet."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create and upload files; otherwise show a dry run.",
    )
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    units_path = args.units_registry
    if dataset == "hmda" and units_path is None:
        units_path = DEFAULT_HMDA_UNITS
    units = load_units(units_path, dataset=dataset, year=args.year)
    part_name = args.part_name or default_part_name(
        dataset, args.year, args.source,
    )
    census_tract_column = args.census_tract_column
    geography_column = args.geography_column
    if dataset == "hmda" and geography_column is None:
        geography_column = "census_tract"
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
            geography_column=geography_column,
            part_name=part_name,
            project=args.project,
            execute=args.execute,
            units=units,
            apply_unit_conversion=not args.values_already_canonical,
        )


if __name__ == "__main__":
    main()
