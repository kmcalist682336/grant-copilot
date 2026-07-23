from __future__ import annotations

import argparse
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"


def download_csv(
    year: int,
    state: str,
    output_root: Path = DEFAULT_RAW_ROOT,
) -> Path:

    """
    Download HMDA LAR CSV data for one state and year.

    Example:
        python scripts/hmda/download_lar_csv.py --year 2024 --state GA
    """
    
    state = state.upper()
    link = f"https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states={state}&years={year}&sexes=Female,Male,Joint,Sex%20Not%20Available"
    output_path = (
        output_root
        / "dataset=hmda"
        / f"year={year}"
        / f"hmda_lar_{year}_{state}.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting download for {state}, {year}...")
    print(f"URL: {link}")
    print(f"Saving to: {output_path}")

    with requests.get(link, stream=True, timeout=120) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

    print(f"Download complete: {output_path}")
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HMDA LAR CSV data by year and state."
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="HMDA activity year, for example 2024.",
    )

    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="Two-letter state code, for example GA.",
    )

    parser.add_argument(
        "--output-root", "--output-dir",
        dest="output_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Raw data root. Default: data/raw",
    )

    args = parser.parse_args()

    download_csv(
        year=args.year,
        state=args.state,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
