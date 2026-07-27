"""tools/append_cards.py — the non-destructive "append a new dataset's
hand-authored questions" wrapper called for in DATASET_INTEGRATION.md
Section 4, Step 4: "Required code touch-point #2 ... a thin wrapper ...
no destructive steps, dataset-scoped insert only."

Does NOT run build_cards.py or swap_cards.py (both are destructive
archive-and-rebuild operations meant for the original v1->v2 migration).
Instead it inserts directly into the EXISTING live `cards` table using the
real v2 schema verified against the live DB:

    - target_year is always NULL
    - years_available / geo_levels_available are JSON arrays, computed
      from `coverage` at (table_id, dataset) granularity
    - card identity is (target_table_id/target_variable_id, target_dataset, text)
      -- a question is embedded once per target, not once per year
    - embedded_at is left NULL so build_embedding_index.py picks it up

Card types follow the same vocabulary as the existing Census cards (see
DATASET_INTEGRATION.md "Card types and their weights/polarity"), with one
addition: `var_question_negative` (weight 1.0, polarity negative) — the
existing pipeline only ever generates positive var_question cards via the
LLM path, but hand-authored negative variable questions are a legitimate
extension of the same pattern already used for table_question_negative.

    table_question_canonical   weight=1.5   positive
    table_question_negative    weight=1.0   negative
    var_question               weight=2.0   positive
    var_question_negative      weight=1.0   negative   (new, additive)

Usage:
    python -m tools.append_cards --dry-run \
        --dataset hmda \
        --table-csv data/staging/hmda/hmda_table_questions.csv \
        --variable-csv data/staging/hmda/hmda_variable_questions.csv

    python -m tools.append_cards \
        --metadata-db data/metadata/search_index.db \
        --dataset hmda \
        --table-csv data/staging/hmda/hmda_table_questions.csv \
        --variable-csv data/staging/hmda/hmda_variable_questions.csv \
        -v
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DB = REPO_ROOT / "data" / "metadata" / "search_index.db"

# Optional columns we'll populate if (and only if) they exist on the live
# `cards` table — introspected at runtime via PRAGMA table_info, since the
# exact column set wasn't fully confirmed against the live schema.
OPTIONAL_CARD_COLS = [
    "measure_type", "subject_area", "l1_domains", "depth", "is_total",
    "is_commonly_queried", "kind", "temporal_framing",
]


def _existing_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_coverage_maps(db: sqlite3.Connection, dataset: str) -> dict[str, tuple[list, list]]:
    """(table_id) -> (years_available, geo_levels_available), collapsed at
    (table_id, dataset) granularity from successful coverage rows only —
    mirrors swap_cards.py's _load_coverage_maps / _json_years_for."""
    rows = db.execute(
        "SELECT table_id, year, geo_level FROM coverage "
        "WHERE dataset = ? AND status = 'success'",
        (dataset,),
    ).fetchall()
    years: dict[str, set] = {}
    geos: dict[str, set] = {}
    for table_id, year, geo_level in rows:
        years.setdefault(table_id, set()).add(year)
        geos.setdefault(table_id, set()).add(geo_level)
    out = {}
    for table_id in set(years) | set(geos):
        out[table_id] = (
            sorted(years.get(table_id, set())),
            sorted(geos.get(table_id, set())),
        )
    return out


def _card_already_exists(
    db: sqlite3.Connection, *, target_table_id: str, target_dataset: str,
    target_variable_id: str | None, text: str,
) -> bool:
    if target_variable_id is None:
        row = db.execute(
            "SELECT 1 FROM cards WHERE target_table_id=? AND target_dataset=? "
            "AND target_variable_id IS NULL AND text=? LIMIT 1",
            (target_table_id, target_dataset, text),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT 1 FROM cards WHERE target_table_id=? AND target_dataset=? "
            "AND target_variable_id=? AND text=? LIMIT 1",
            (target_table_id, target_dataset, target_variable_id, text),
        ).fetchone()
    return row is not None


def append_table_question_cards(
    db: sqlite3.Connection, csv_path: Path, *, dataset: str,
    coverage_maps: dict, weight_positive: float = 1.5, weight_negative: float = 1.0,
) -> int:
    rows = [r for r in _read_csv(csv_path) if r.get("dataset", "").strip() == dataset]
    tables_meta = {
        r[0]: r for r in db.execute(
            "SELECT table_id, measure_type, subject_area, l1_domains_json "
            "FROM tables WHERE dataset = ?", (dataset,),
        ).fetchall()
    }
    opt_cols = [c for c in OPTIONAL_CARD_COLS if c in _existing_columns(db, "cards")]
    inserted = 0
    for r in rows:
        table_id = r["table_id"].strip()
        polarity = r["polarity"].strip().lower()
        text = r["text"].strip()
        if not text:
            continue
        if _card_already_exists(db, target_table_id=table_id, target_dataset=dataset,
                                 target_variable_id=None, text=text):
            continue

        card_type = "table_question_negative" if polarity == "negative" else "table_question_canonical"
        weight = weight_negative if polarity == "negative" else weight_positive
        years, geos = coverage_maps.get(table_id, ([], []))
        meta = tables_meta.get(table_id)

        cols = ["card_type", "text", "target_table_id", "target_dataset",
                "target_variable_id", "target_year", "polarity", "weight",
                "years_available", "geo_levels_available", "embedded_at"]
        vals = [card_type, text, table_id, dataset, None, None, polarity, weight,
                json.dumps(years), json.dumps(geos), None]

        if "measure_type" in opt_cols:
            cols.append("measure_type"); vals.append(meta[1] if meta else None)
        if "subject_area" in opt_cols:
            cols.append("subject_area"); vals.append(meta[2] if meta else None)
        if "l1_domains" in opt_cols:
            cols.append("l1_domains"); vals.append(meta[3] if meta else None)

        placeholders = ",".join("?" for _ in cols)
        db.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({placeholders})", vals)
        inserted += 1
    return inserted


def append_variable_question_cards(
    db: sqlite3.Connection, csv_path: Path, *, dataset: str,
    coverage_maps: dict, weight_positive: float = 2.0, weight_negative: float = 1.0,
) -> int:
    rows = [r for r in _read_csv(csv_path) if r.get("dataset", "").strip() == dataset]
    variables_meta = {
        (r[0], r[1]): r for r in db.execute(
            "SELECT table_id, variable_id, depth, is_total, is_commonly_queried, kind "
            "FROM variables WHERE dataset = ?", (dataset,),
        ).fetchall()
    }
    tables_meta = {
        r[0]: r for r in db.execute(
            "SELECT table_id, measure_type, subject_area, l1_domains_json "
            "FROM tables WHERE dataset = ?", (dataset,),
        ).fetchall()
    }
    opt_cols = [c for c in OPTIONAL_CARD_COLS if c in _existing_columns(db, "cards")]
    inserted = 0
    skipped_not_cq = 0
    for r in rows:
        table_id = r["table_id"].strip()
        variable_id = r["variable_id"].strip()
        polarity = r["polarity"].strip().lower()
        text = r["text"].strip()
        if not text:
            continue

        vmeta = variables_meta.get((table_id, variable_id))
        if vmeta is None or vmeta[4] != 1:  # is_commonly_queried
            skipped_not_cq += 1
            continue

        if _card_already_exists(db, target_table_id=table_id, target_dataset=dataset,
                                 target_variable_id=variable_id, text=text):
            continue

        card_type = "var_question_negative" if polarity == "negative" else "var_question"
        weight = weight_negative if polarity == "negative" else weight_positive
        years, geos = coverage_maps.get(table_id, ([], []))
        tmeta = tables_meta.get(table_id)

        cols = ["card_type", "text", "target_table_id", "target_dataset",
                "target_variable_id", "target_year", "polarity", "weight",
                "years_available", "geo_levels_available", "embedded_at"]
        vals = [card_type, text, table_id, dataset, variable_id, None, polarity, weight,
                json.dumps(years), json.dumps(geos), None]

        if "measure_type" in opt_cols:
            cols.append("measure_type"); vals.append(tmeta[1] if tmeta else None)
        if "subject_area" in opt_cols:
            cols.append("subject_area"); vals.append(tmeta[2] if tmeta else None)
        if "l1_domains" in opt_cols:
            cols.append("l1_domains"); vals.append(tmeta[3] if tmeta else None)
        if "depth" in opt_cols:
            cols.append("depth"); vals.append(vmeta[2])
        if "is_total" in opt_cols:
            cols.append("is_total"); vals.append(vmeta[3])
        if "is_commonly_queried" in opt_cols:
            cols.append("is_commonly_queried"); vals.append(vmeta[4])
        if "kind" in opt_cols:
            cols.append("kind"); vals.append(vmeta[5])

        placeholders = ",".join("?" for _ in cols)
        db.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({placeholders})", vals)
        inserted += 1

    if skipped_not_cq:
        logger.info("skipped %d variable question rows (not is_commonly_queried=1 "
                     "or variable not found in `variables`)", skipped_not_cq)
    return inserted


def run(
    metadata_db: Path, dataset: str, *,
    table_csv: Path | None = None,
    variable_csv: Path | None = None,
    dry_run: bool = False,
) -> dict:
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    db = sqlite3.connect(str(metadata_db))
    db.execute("PRAGMA journal_mode=WAL")
    try:
        if dry_run:
            stats = {}
            if table_csv:
                stats["table_questions_in_csv"] = len(
                    [r for r in _read_csv(table_csv) if r.get("dataset", "").strip() == dataset]
                )
            if variable_csv:
                stats["variable_questions_in_csv"] = len(
                    [r for r in _read_csv(variable_csv) if r.get("dataset", "").strip() == dataset]
                )
            logger.info("[dry-run] no rows inserted")
            return stats

        coverage_maps = _load_coverage_maps(db, dataset)

        stats: dict = {}
        if table_csv:
            n = append_table_question_cards(db, table_csv, dataset=dataset, coverage_maps=coverage_maps)
            stats["table_question_cards_inserted"] = n
            logger.info("table question cards inserted: %d", n)

        if variable_csv:
            n = append_variable_question_cards(db, variable_csv, dataset=dataset, coverage_maps=coverage_maps)
            stats["variable_question_cards_inserted"] = n
            logger.info("variable question cards inserted: %d", n)

        db.commit()

        stats["total_cards_for_dataset"] = db.execute(
            "SELECT COUNT(*) FROM cards WHERE target_dataset = ?", (dataset,)
        ).fetchone()[0]
        stats["pending_embedding"] = db.execute(
            "SELECT COUNT(*) FROM cards WHERE target_dataset = ? AND embedded_at IS NULL",
            (dataset,),
        ).fetchone()[0]
        return stats
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--metadata-db", default=str(DEFAULT_METADATA_DB))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--variable-csv", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.table_csv and not args.variable_csv:
        logger.error("Provide at least one of --table-csv / --variable-csv.")
        return 1

    stats = run(
        metadata_db=Path(args.metadata_db),
        dataset=args.dataset,
        table_csv=args.table_csv,
        variable_csv=args.variable_csv,
        dry_run=args.dry_run,
    )
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
