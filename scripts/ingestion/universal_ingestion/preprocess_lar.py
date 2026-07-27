"""Preprocess a raw HMDA LAR CSV into a decoded Parquet dataset.

The source CSV remains untouched in the raw data layer. Fields represented in
the HMDA codebook are replaced with human-readable values in the processed
Parquet file.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "hmda"
DEFAULT_CODEBOOK = Path(__file__).parent / "config" / "hmda" / "hmda_lar_codebook.csv"
DEFAULT_SCHEMA = Path(__file__).parent / "config" / "hmda" / "hmda_schema.yaml"


def clean_column_name(name: str) -> str:
    """Return a stable snake_case-like column name."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return cleaned.strip("_")


def load_codebook(codebook_path: Path, year: int) -> dict[str, dict[str, str]]:
    """Load mappings that apply to the requested activity year."""
    codebook = pd.read_csv(codebook_path, dtype=str, keep_default_na=False)
    required = {"field_name", "raw_value", "label", "year_start", "year_end"}
    missing = required.difference(codebook.columns)
    if missing:
        raise ValueError(f"Codebook is missing columns: {', '.join(sorted(missing))}")

    starts = pd.to_numeric(codebook["year_start"], errors="coerce").fillna(0)
    ends = pd.to_numeric(codebook["year_end"], errors="coerce").fillna(9999)
    active = codebook.loc[(starts <= year) & (year <= ends)].copy()
    active["field_name"] = active["field_name"].map(clean_column_name)

    duplicates = active.duplicated(["field_name", "raw_value"], keep=False)
    if duplicates.any():
        examples = active.loc[duplicates, ["field_name", "raw_value"]].head(5)
        raise ValueError(f"Codebook has duplicate active mappings:\n{examples.to_string(index=False)}")

    return {
        field: dict(zip(group["raw_value"], group["label"]))
        for field, group in active.groupby("field_name", sort=False)
    }


def load_schema(schema_path: Path) -> dict[str, str]:
    """Load configured analytical types keyed by cleaned column name."""
    with schema_path.open(encoding="utf-8") as file:
        document = yaml.safe_load(file) or {}
    numeric_fields = document.get("numeric_fields", {})
    allowed = {"float64", "int64"}
    invalid = {field: kind for field, kind in numeric_fields.items() if kind not in allowed}
    if invalid:
        raise ValueError(f"Unsupported numeric types in {schema_path}: {invalid}")
    return {clean_column_name(field): kind for field, kind in numeric_fields.items()}


def normalize_geography(chunk: pd.DataFrame) -> None:
    """Normalize geographic identifiers without converting them to numbers."""
    if "state_code" in chunk:
        chunk["state_code"] = chunk["state_code"].str.strip().str.upper()
    if "county_code" in chunk:
        present = chunk["county_code"].ne("")
        chunk.loc[present, "county_code"] = chunk.loc[present, "county_code"].str.zfill(5)
    if "census_tract" in chunk:
        present = chunk["census_tract"].ne("")
        chunk.loc[present, "census_tract"] = chunk.loc[present, "census_tract"].str.zfill(11)


def normalize_numeric_fields(
    chunk: pd.DataFrame,
    numeric_types: dict[str, str],
    invalid_numeric_values: dict[str, set[str]],
) -> None:
    """Convert measures to nullable numbers and retain special-value meaning."""
    known_statuses = {
        "NA": "Not applicable",
        "Exempt": "Exempt",
    }
    for field, numeric_type in numeric_types.items():
        if field not in chunk.columns:
            continue

        values = chunk[field].astype("string")
        nonempty = values.notna() & values.ne("")
        numeric = pd.to_numeric(values, errors="coerce")
        status_mask = nonempty & numeric.isna()
        statuses = values.where(status_mask).replace(known_statuses)

        unexpected = values.loc[status_mask & ~values.isin(known_statuses)].dropna().unique()
        if len(unexpected):
            invalid_numeric_values[field].update(str(value) for value in unexpected)

        if numeric_type == "int64":
            fractional = numeric.loc[numeric.notna() & numeric.mod(1).ne(0)].unique()
            if len(fractional):
                invalid_numeric_values[field].update(str(value) for value in fractional)
                numeric = numeric.mask(numeric.mod(1).ne(0))
            chunk[field] = numeric.astype("Int64")
        else:
            chunk[field] = numeric.astype("Float64")
        chunk[f"{field}_status"] = statuses.astype("string")


def decode_chunk(
    chunk: pd.DataFrame,
    mappings: dict[str, dict[str, str]],
    unknown_values: dict[str, set[str]],
    numeric_types: dict[str, str] | None = None,
    invalid_numeric_values: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    chunk.columns = [clean_column_name(column) for column in chunk.columns]
    normalize_geography(chunk)

    for field, mapping in mappings.items():
        if field not in chunk.columns:
            continue
        values = chunk[field]
        decoded = values.map(mapping)
        unknown = values.loc[values.ne("") & ~values.isin(mapping)].unique()
        if len(unknown):
            unknown_values[field].update(str(value) for value in unknown)
        # Preserve empty and unknown values rather than silently losing them.
        chunk[field] = decoded.where(decoded.notna(), values)

    if numeric_types:
        normalize_numeric_fields(
            chunk,
            numeric_types,
            invalid_numeric_values if invalid_numeric_values is not None else defaultdict(set),
        )

    return chunk


def decode_csv_to_parquet(
    input_path: Path,
    output_path: Path,
    *,
    year: int,
    codebook_path: Path = DEFAULT_CODEBOOK,
    schema_path: Path = DEFAULT_SCHEMA,
    chunk_size: int = 50_000,
) -> Path:
    """Decode an exact raw HMDA CSV path into an exact Parquet path."""
    temp_path = output_path.with_suffix(".parquet.tmp")

    if not input_path.is_file():
        raise FileNotFoundError(f"Raw HMDA file not found: {input_path}")
    if not codebook_path.is_file():
        raise FileNotFoundError(f"HMDA codebook not found: {codebook_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"HMDA schema not found: {schema_path}")

    # Keep the transformation helpers importable when only pandas validation is
    # needed, while giving the Parquet command a direct dependency error.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet output requires pyarrow: python -m pip install pyarrow") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.unlink(missing_ok=True)
    mappings = load_codebook(codebook_path, year)
    numeric_types = load_schema(schema_path)
    unknown_values: dict[str, set[str]] = defaultdict(set)
    invalid_numeric_values: dict[str, set[str]] = defaultdict(set)
    writer: Any = None
    rows_written = 0

    try:
        try:
            chunks = pd.read_csv(
                input_path,
                dtype=str,
                keep_default_na=False,
                chunksize=chunk_size,
                low_memory=False,
            )
            for chunk in chunks:
                decoded = decode_chunk(
                    chunk,
                    mappings,
                    unknown_values,
                    numeric_types,
                    invalid_numeric_values,
                )
                table = pa.Table.from_pandas(decoded, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema, compression="snappy")
                writer.write_table(table)
                rows_written += len(decoded)
                print(f"Processed {rows_written:,} rows", end="\r")
        finally:
            if writer is not None:
                writer.close()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if writer is None:
        raise ValueError(f"Raw HMDA file contains no records: {input_path}")

    temp_path.replace(output_path)
    print(f"\nWrote {rows_written:,} rows to {output_path}")
    if unknown_values:
        print("Unmapped codebook values (left intact in raw columns):")
        for field, values in sorted(unknown_values.items()):
            preview = ", ".join(sorted(values)[:10])
            suffix = " ..." if len(values) > 10 else ""
            print(f"  {field}: {preview}{suffix}")
    if invalid_numeric_values:
        print("Unexpected non-numeric measure values (recorded in *_status columns):")
        for field, values in sorted(invalid_numeric_values.items()):
            preview = ", ".join(sorted(values)[:10])
            suffix = " ..." if len(values) > 10 else ""
            print(f"  {field}: {preview}{suffix}")

    return output_path


def decode_lar(
    year: int,
    state: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    codebook_path: Path = DEFAULT_CODEBOOK,
    schema_path: Path = DEFAULT_SCHEMA,
    chunk_size: int = 50_000,
) -> Path:
    """Decode one state/year HMDA LAR file and write it as Parquet."""
    state = state.upper()
    input_path = (
        raw_dir
        / "dataset=hmda"
        / f"year={year}"
        / f"hmda_lar_{year}_{state}.csv"
    )
    if not input_path.is_file():
        legacy_path = (
            raw_dir
            / "hmda"
            / f"year={year}"
            / f"state={state}"
            / f"hmda_lar_{year}_{state}.csv"
        )
        if legacy_path.is_file():
            input_path = legacy_path
    output_path = (
        output_dir
        / f"year={year}"
        / f"state={state}"
        / f"hmda_lar_{year}_{state}.parquet"
    )
    return decode_csv_to_parquet(
        input_path,
        output_path,
        year=year,
        codebook_path=codebook_path,
        schema_path=schema_path,
        chunk_size=chunk_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode an HMDA LAR CSV to Parquet.")
    parser.add_argument("--year", type=int, nargs="+", required=True)
    parser.add_argument("--state", required=True, help="Two-letter state code, such as GA")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    for year in args.year:
        decode_lar(
            year=year,
            state=args.state,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            codebook_path=args.codebook,
            schema_path=args.schema,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
