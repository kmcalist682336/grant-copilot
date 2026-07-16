"""Decode raw HMDA data and write a physical variable tree.

This is the single entry point for the demo ingestion flow. It applies the
same codebook, geography, and numeric cleaning rules as ``preprocess_lar`` and
then writes one two-column Parquet file per decoded variable. The decoded
whole-table Parquet is temporary unless ``--keep-processed`` is supplied.

The command defaults to a dry run. Nothing is written unless ``--execute`` is
provided.

Example dry run:
    python -m scripts.ingestion.universal_ingestion.split_hmda_columns \
        --year 2024 --state GA

Example write:
    python -m scripts.ingestion.universal_ingestion.split_hmda_columns \
        --year 2024 --state GA --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
from pathlib import Path

import duckdb

from scripts.ingestion.universal_ingestion.preprocess_lar import (
    DEFAULT_CODEBOOK,
    DEFAULT_RAW_DIR,
    DEFAULT_SCHEMA,
    decode_lar,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "processed" / "hmda"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "variable_tree"


def sql_string(value: str) -> str:
    """Escape a value as a DuckDB string literal."""
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    """Quote a column identifier, including names containing punctuation."""
    return '"' + value.replace('"', '""') + '"'


def variable_id(source_name: str) -> str:
    """Return a stable opaque ID; never use Python's process-randomized hash."""
    canonical = f"hmda:{source_name.strip().lower()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def update_registry(dataset_root: Path, variables: list[str]) -> Path:
    """Maintain the durable mapping between opaque IDs and source columns."""
    registry_path = dataset_root / "variable_registry.csv"
    entries: dict[str, str] = {}
    if registry_path.is_file():
        with registry_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                entries[row["variable_id"]] = row["source_name"]

    for source_name in variables:
        opaque_id = variable_id(source_name)
        existing = entries.get(opaque_id)
        if existing is not None and existing != source_name:
            raise ValueError(
                f"Variable ID collision: {opaque_id} maps to {existing!r} and {source_name!r}"
            )
        entries[opaque_id] = source_name

    dataset_root.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["variable_id", "source_name"])
        writer.writeheader()
        for opaque_id, source_name in sorted(entries.items(), key=lambda item: item[1]):
            writer.writerow({"variable_id": opaque_id, "source_name": source_name})
    return registry_path


def find_source(input_root: Path, year: int, state: str) -> Path:
    """Prefer processed Parquet, with processed CSV as a fallback."""
    partition = input_root / f"year={year}" / f"state={state}"
    candidates = [
        partition / f"hmda_lar_{year}_{state}.parquet",
        partition / f"hmda_lar_{year}_{state}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No processed HMDA CSV or Parquet found under {partition}")


def source_scan_sql(source: Path) -> str:
    path = sql_string(source.as_posix())
    if source.suffix.lower() == ".parquet":
        # The source path already contains year= and state= directories.
        # Disable Hive inference so those virtual partition fields do not get
        # mistaken for physical HMDA variables and split into redundant files.
        return f"read_parquet({path}, hive_partitioning = false)"
    return f"read_csv_auto({path}, all_varchar = true)"


def split_partition(
    *,
    year: int,
    state: str,
    input_root: Path,
    output_root: Path,
    execute: bool,
) -> int:
    """Create one two-column Parquet file per variable for one partition."""
    state = state.upper()
    source = find_source(input_root, year, state)
    scan = source_scan_sql(source)
    con = duckdb.connect()

    try:
        # DESCRIBE reads the source schema without loading every value.
        described = con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()
        variables = [row[0] for row in described]

        print(f"Source: {source}")
        print(f"Variables: {len(variables)}")
        print(f"Output root: {output_root / 'dataset=hmda'}")
        if not execute:
            print("Dry run only; pass --execute to write files.")
            for variable in variables[:10]:
                opaque_id = variable_id(variable)
                path = (
                    output_root
                    / "dataset=hmda"
                    / f"year={year}"
                    / f"geography=state-{state}"
                    / f"variable={opaque_id}"
                    / "part-000.parquet"
                )
                print(f"  {variable} ({opaque_id}) -> {path}")
            if len(variables) > 10:
                print(f"  ... and {len(variables) - 10} more variable files")
            return len(variables)

        # Materialize the source once. This fixes one row number for the whole
        # transformation instead of independently rescanning every variable.
        con.execute(
            f"""
            CREATE TEMP TABLE hmda_source AS
            SELECT
                row_number() OVER () - 1 AS _source_row_number,
                *
            FROM {scan}
            """
        )

        id_prefix = f"{year}:{state}:{source.stem}:"
        dataset_root = output_root / "dataset=hmda"
        registry_path = update_registry(dataset_root, variables)
        print(f"Variable registry: {registry_path}")
        for index, variable in enumerate(variables, start=1):
            opaque_id = variable_id(variable)
            destination = (
                dataset_root
                / f"year={year}"
                / f"geography=state-{state}"
                / f"variable={opaque_id}"
                / "part-000.parquet"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)

            # Every physical variable file includes the same stable record ID.
            # Reconstructing a multi-variable loan table still requires SQL
            # JOINs on hmda_record_id; globbing alone will not perform that join.
            con.execute(
                f"""
                COPY (
                    SELECT
                        concat(
                            {sql_string(id_prefix)},
                            lpad(CAST(_source_row_number AS VARCHAR), 12, '0')
                        ) AS hmda_record_id,
                        {sql_identifier(variable)} AS value
                    FROM hmda_source
                )
                TO {sql_string(destination.as_posix())}
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            print(f"[{index}/{len(variables)}] Wrote {variable} as {opaque_id}")

        return len(variables)
    finally:
        con.close()


def decode_and_split(
    *,
    year: int,
    state: str,
    raw_root: Path,
    output_root: Path,
    codebook_path: Path,
    schema_path: Path,
    chunk_size: int,
    execute: bool,
    keep_processed: bool,
    processed_root: Path,
) -> int:
    """Decode a raw partition and split its decoded variables into files."""
    state = state.upper()

    if not execute:
        raw_path = (
            raw_root
            / f"year={year}"
            / f"state={state}"
            / f"hmda_lar_{year}_{state}.csv"
        )
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw HMDA file not found: {raw_path}")
        print(f"Raw source: {raw_path}")
        print(f"Codebook: {codebook_path}")
        print(f"Schema: {schema_path}")
        print(f"Output root: {output_root / 'dataset=hmda'}")
        print("Dry run only; pass --execute to decode and write files.")
        return 0

    if keep_processed:
        decode_lar(
            year=year,
            state=state,
            raw_dir=raw_root,
            output_dir=processed_root,
            codebook_path=codebook_path,
            schema_path=schema_path,
            chunk_size=chunk_size,
        )
        return split_partition(
            year=year,
            state=state,
            input_root=processed_root,
            output_root=output_root,
            execute=True,
        )

    # The splitter needs a common decoded table so every variable receives the
    # same record IDs. Keep it only for the duration of this command.
    with tempfile.TemporaryDirectory(prefix="hmda-decoded-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        decode_lar(
            year=year,
            state=state,
            raw_dir=raw_root,
            output_dir=temporary_root,
            codebook_path=codebook_path,
            schema_path=schema_path,
            chunk_size=chunk_size,
        )
        return split_partition(
            year=year,
            state=state,
            input_root=temporary_root,
            output_root=output_root,
            execute=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode raw HMDA data and split it into a variable tree."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--state", default="GA")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="Also retain the decoded whole-table Parquet dataset.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Destination used only with --keep-processed.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the column-separated files; otherwise show a dry run.",
    )
    args = parser.parse_args()

    decode_and_split(
        year=args.year,
        state=args.state,
        raw_root=args.raw_root,
        output_root=args.output_root,
        codebook_path=args.codebook,
        schema_path=args.schema,
        chunk_size=args.chunk_size,
        execute=args.execute,
        keep_processed=args.keep_processed,
        processed_root=args.processed_root,
    )


if __name__ == "__main__":
    main()
