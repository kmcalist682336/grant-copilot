"""K.2 — build the embedding-card table from the imported v1 stages.

After K.1 (``import_v1_stages.py``) the metadata DB carries:
- ``tables`` with stage4 descriptions + use_when + avoid_when
- ``table_questions`` with stage6 canonical + varied + negative questions
- ``variables`` with stage7 enrichment (label_natural, is_commonly_queried, ...)
- ``variable_questions`` with stage7 per-variable questions

This tool selects a subset of those signals and materializes one row per
embedding "card" into a ``cards`` table. Each card is a small text snippet
plus structured metadata; downstream (K.3) embeds the text via Vertex and
writes the vector into a FAISS index keyed on ``cards.rowid``.

Card sources + weights + polarity (defaults; tunable by flag):

    table_description          weight=1.0  positive   (1 / table)
    table_avoid_when           weight=1.0  NEGATIVE   (1 / table when present)
    table_question_canonical   weight=1.5  positive   (~10 / table)
    table_question_varied      weight=1.3  positive   (~14 / table)
    table_question_negative    weight=1.0  NEGATIVE   (~10 / table)
    var_label_natural          weight=1.0  positive   (1 / cq variable)
    var_question               weight=2.0  positive   (top-N per cq variable)
    concept_map_alias          weight=1.8  positive   (~3-5 / curated entry)

Filter levers:
    --include-non-cq-vars     also embed label_natural for non-commonly-queried
                              variables (large)
    --var-questions-cap N     keep only first N questions per variable (default 3)
    --max-var-depth N         drop var_question cards for variables deeper than N
                              (default unlimited)
    --no-concept-map          skip the concept_map curation cards

Idempotent — re-running rebuilds the cards table fresh by default.
Pass ``--append`` to add to existing rows (used during eval-driven
iteration).

Usage:
    python -m tools.build_cards --dry-run
    python -m tools.build_cards
    python -m tools.build_cards --var-questions-cap 5
    python -m tools.build_cards --include-non-cq-vars   # heavy
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DB = REPO_ROOT / "data" / "metadata" / "search_index.db"
DEFAULT_CONCEPT_MAP = REPO_ROOT / "config" / "concept_map.yaml"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CARDS_DDL = """
CREATE TABLE IF NOT EXISTS cards (
    rowid              INTEGER PRIMARY KEY AUTOINCREMENT,
    card_type          TEXT    NOT NULL,
    text               TEXT    NOT NULL,

    target_table_id    TEXT    NOT NULL,
    target_dataset     TEXT    NOT NULL,
    target_year        INTEGER NOT NULL,
    target_variable_id TEXT,                       -- null for table cards

    weight             REAL    NOT NULL DEFAULT 1.0,
    polarity           TEXT    NOT NULL DEFAULT 'positive'
                              CHECK (polarity IN ('positive', 'negative')),

    -- structured filter columns (denormalized at build time so retrieval
    -- doesn't need to JOIN against tables/variables)
    measure_type       TEXT,
    subject_area       TEXT,
    l1_domains         TEXT,           -- JSON list as string
    temporal_framing   TEXT,           -- from stage6_varied attributes
    is_commonly_queried INTEGER,
    is_total           INTEGER,
    depth              INTEGER,
    kind               TEXT,           -- variable kind ('estimate', etc.)

    -- embed bookkeeping (filled by K.3)
    embedded_at        REAL,
    embedding_version  TEXT
);

CREATE INDEX IF NOT EXISTS idx_cards_target
    ON cards(target_table_id, target_dataset, target_year);
CREATE INDEX IF NOT EXISTS idx_cards_type
    ON cards(card_type);
CREATE INDEX IF NOT EXISTS idx_cards_polarity
    ON cards(polarity);
CREATE INDEX IF NOT EXISTS idx_cards_unembedded
    ON cards(embedded_at);
CREATE INDEX IF NOT EXISTS idx_cards_variable
    ON cards(target_variable_id, target_dataset, target_year);
"""


def ensure_schema(db: sqlite3.Connection) -> None:
    db.executescript(_CARDS_DDL)
    db.commit()


def truncate_cards(db: sqlite3.Connection) -> int:
    """Drop and recreate the cards table — used when we want a fresh build."""
    n_before = db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    db.execute("DELETE FROM cards")
    db.execute("DELETE FROM sqlite_sequence WHERE name='cards'")
    db.commit()
    return n_before


# ---------------------------------------------------------------------------
# Per-card-type builders — each returns rows inserted
# ---------------------------------------------------------------------------

def _insert_count(db: sqlite3.Connection) -> int:
    """Return rowid of the most recent insert (i.e., total rows so far)."""
    return db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]


def build_table_descriptions(
    db: sqlite3.Connection, *, weight: float = 1.0,
) -> int:
    """One card per table: 'title. description use_when' (truncated)."""
    sql = f"""
    INSERT INTO cards
      (card_type, text, target_table_id, target_dataset, target_year,
       weight, polarity, measure_type, subject_area, l1_domains)
    SELECT
      'table_description',
      TRIM(
        title || '. ' ||
        COALESCE(description, '') || ' ' ||
        COALESCE(use_when, '')
      ),
      table_id, dataset, year,
      {weight}, 'positive',
      measure_type, subject_area, l1_domains_json
    FROM tables
    WHERE description IS NOT NULL
       OR use_when IS NOT NULL
    """
    cur = db.execute(sql)
    return cur.rowcount


def build_table_avoid_when(
    db: sqlite3.Connection, *, weight: float = 1.0,
) -> int:
    """One NEGATIVE card per table where avoid_when is populated."""
    sql = f"""
    INSERT INTO cards
      (card_type, text, target_table_id, target_dataset, target_year,
       weight, polarity, measure_type, subject_area, l1_domains)
    SELECT
      'table_avoid_when', avoid_when,
      table_id, dataset, year,
      {weight}, 'negative',
      measure_type, subject_area, l1_domains_json
    FROM tables
    WHERE avoid_when IS NOT NULL AND avoid_when != ''
    """
    cur = db.execute(sql)
    return cur.rowcount


def build_table_questions(
    db: sqlite3.Connection, *,
    weight_canonical: float = 1.5,
    weight_varied: float = 1.3,
    weight_negative: float = 1.0,
) -> int:
    """One card per stage6 question. Kind drives weight + polarity."""
    total = 0
    for kind, weight, polarity in (
        ('canonical', weight_canonical, 'positive'),
        ('varied',    weight_varied,    'positive'),
        ('negative',  weight_negative,  'negative'),
    ):
        sql = f"""
        INSERT INTO cards
          (card_type, text, target_table_id, target_dataset, target_year,
           weight, polarity, measure_type, subject_area, l1_domains,
           temporal_framing)
        SELECT
          'table_question_{kind}', tq.question_text,
          tq.table_id, tq.dataset, tq.year,
          {weight}, '{polarity}',
          t.measure_type, t.subject_area, t.l1_domains_json,
          json_extract(tq.attributes_json, '$.temporal_framing')
        FROM table_questions tq
        JOIN tables t
          ON t.table_id = tq.table_id
         AND t.dataset  = tq.dataset
         AND t.year     = tq.year
        WHERE tq.kind = '{kind}'
        """
        cur = db.execute(sql)
        n = cur.rowcount
        logger.info("  table_question_%s: %d cards", kind, n)
        total += n
    return total


def build_var_label_natural(
    db: sqlite3.Connection, *, include_non_cq: bool = False,
    weight: float = 1.0,
) -> int:
    """One card per (commonly-queried) variable using its label_natural."""
    cq_filter = "" if include_non_cq else "AND v.is_commonly_queried = 1"
    sql = f"""
    INSERT INTO cards
      (card_type, text, target_table_id, target_dataset, target_year,
       target_variable_id, weight, polarity,
       measure_type, subject_area, l1_domains,
       is_commonly_queried, is_total, depth, kind)
    SELECT
      'var_label_natural', v.label_natural,
      v.table_id, v.dataset, v.year, v.variable_id,
      {weight}, 'positive',
      t.measure_type, t.subject_area, t.l1_domains_json,
      v.is_commonly_queried, v.is_total, v.depth, v.kind
    FROM variables v
    JOIN tables t
      ON t.table_id = v.table_id
     AND t.dataset  = v.dataset
     AND t.year     = v.year
    WHERE v.label_natural IS NOT NULL AND v.label_natural != ''
      AND v.kind = 'estimate'
      {cq_filter}
    """
    cur = db.execute(sql)
    return cur.rowcount


def build_var_questions(
    db: sqlite3.Connection, *,
    cap_per_var: int = 3,
    max_depth: Optional[int] = None,
    weight: float = 2.0,
) -> int:
    """One card per stage7 per-variable question, filtered to commonly-
    queried variables, capped at top-N questions per variable."""
    depth_filter = f"AND v.depth <= {max_depth}" if max_depth else ""
    sql = f"""
    INSERT INTO cards
      (card_type, text, target_table_id, target_dataset, target_year,
       target_variable_id, weight, polarity,
       measure_type, subject_area, l1_domains,
       is_commonly_queried, is_total, depth, kind)
    SELECT
      'var_question', vq.question_text,
      vq.table_id, vq.dataset, vq.year, vq.var_id,
      {weight}, 'positive',
      t.measure_type, t.subject_area, t.l1_domains_json,
      v.is_commonly_queried, v.is_total, v.depth, v.kind
    FROM variable_questions vq
    JOIN variables v
      ON v.table_id    = vq.table_id
     AND v.dataset     = vq.dataset
     AND v.year        = vq.year
     AND v.variable_id = vq.var_id
    JOIN tables t
      ON t.table_id = vq.table_id
     AND t.dataset  = vq.dataset
     AND t.year     = vq.year
    WHERE v.is_commonly_queried = 1
      AND v.kind = 'estimate'
      AND vq.position < {cap_per_var}
      {depth_filter}
    """
    cur = db.execute(sql)
    return cur.rowcount


def build_concept_map_cards(
    db: sqlite3.Connection,
    concept_map_path: Path,
    *, weight: float = 1.8,
) -> int:
    """One card per concept_map canonical + each alias.

    Each curated entry pins to a specific (table_id, dataset). For year
    we use the most recent year that the table+dataset has coverage
    (find_supported_years returns descending; we take the first).
    """
    if not concept_map_path.exists():
        logger.info("  concept_map.yaml missing; skipping curation cards")
        return 0
    with concept_map_path.open() as f:
        doc = yaml.safe_load(f) or {}
    entries = doc.get("concepts") or []

    rows: list[tuple] = []
    for entry in entries:
        canonical = entry.get("canonical")
        aliases = entry.get("aliases") or []
        table_id = entry.get("table_id")
        dataset = entry.get("dataset", "acs/acs5")
        if not (canonical and table_id):
            continue
        # Pick the most recent year with successful coverage.
        year_row = db.execute(
            "SELECT MAX(year) FROM coverage WHERE table_id=? AND dataset=? "
            "AND status='success'",
            (table_id, dataset),
        ).fetchone()
        year = year_row[0] if year_row and year_row[0] is not None else None
        if year is None:
            continue
        for text in (canonical, *aliases):
            rows.append((
                'concept_map_alias', text,
                table_id, dataset, year, None,
                weight, 'positive',
                None, None, None, None,  # measure_type/subject/l1/temp
                None, None, None, None,  # is_cq/is_total/depth/kind
            ))
    if not rows:
        return 0
    db.executemany(
        "INSERT INTO cards "
        "(card_type, text, target_table_id, target_dataset, target_year, "
        " target_variable_id, weight, polarity, "
        " measure_type, subject_area, l1_domains, temporal_framing, "
        " is_commonly_queried, is_total, depth, kind) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_build(
    metadata_db_path: Path,
    *,
    include_non_cq_vars: bool = False,
    var_questions_cap: int = 3,
    max_var_depth: Optional[int] = None,
    use_concept_map: bool = True,
    concept_map_path: Path = DEFAULT_CONCEPT_MAP,
    append: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    db = sqlite3.connect(metadata_db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")

    try:
        ensure_schema(db)
        if dry_run:
            logger.info("[dry-run] showing planned counts only")
            return _dry_run_counts(
                db,
                include_non_cq_vars=include_non_cq_vars,
                var_questions_cap=var_questions_cap,
                max_var_depth=max_var_depth,
                use_concept_map=use_concept_map,
                concept_map_path=concept_map_path,
            )

        if not append:
            n = truncate_cards(db)
            if n:
                logger.info("Truncated existing cards: %d rows", n)

        stats: dict[str, int] = {}

        t0 = time.time()
        n = build_table_descriptions(db)
        stats["table_description"] = n
        logger.info("table_description: %d cards", n)

        n = build_table_avoid_when(db)
        stats["table_avoid_when"] = n
        logger.info("table_avoid_when (negative): %d cards", n)

        n = build_table_questions(db)
        stats["table_questions_total"] = n
        db.commit()

        n = build_var_label_natural(db, include_non_cq=include_non_cq_vars)
        stats["var_label_natural"] = n
        logger.info("var_label_natural: %d cards", n)
        db.commit()

        n = build_var_questions(
            db, cap_per_var=var_questions_cap, max_depth=max_var_depth,
        )
        stats["var_question"] = n
        logger.info("var_question (cap=%d, max_depth=%s): %d cards",
                    var_questions_cap, max_var_depth, n)
        db.commit()

        if use_concept_map:
            n = build_concept_map_cards(db, concept_map_path)
            stats["concept_map_alias"] = n
            logger.info("concept_map_alias: %d cards", n)
        else:
            stats["concept_map_alias"] = 0

        db.commit()
        stats["total"] = db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        stats["elapsed_s"] = round(time.time() - t0, 1)
        return stats
    finally:
        db.close()


def _dry_run_counts(
    db: sqlite3.Connection, **opts,
) -> dict[str, int]:
    """Compute card counts via SELECT COUNTs without inserting."""
    out: dict[str, int] = {}
    out["table_description"] = db.execute(
        "SELECT COUNT(*) FROM tables "
        "WHERE description IS NOT NULL OR use_when IS NOT NULL"
    ).fetchone()[0]
    out["table_avoid_when"] = db.execute(
        "SELECT COUNT(*) FROM tables "
        "WHERE avoid_when IS NOT NULL AND avoid_when != ''"
    ).fetchone()[0]
    out["table_questions_total"] = db.execute(
        "SELECT COUNT(*) FROM table_questions"
    ).fetchone()[0]
    cq = "" if opts["include_non_cq_vars"] else "AND is_commonly_queried = 1"
    out["var_label_natural"] = db.execute(
        f"SELECT COUNT(*) FROM variables "
        f"WHERE label_natural IS NOT NULL AND label_natural != '' "
        f"AND kind = 'estimate' {cq}"
    ).fetchone()[0]
    depth_clause = (
        f"AND v.depth <= {opts['max_var_depth']}"
        if opts.get("max_var_depth") else ""
    )
    out["var_question"] = db.execute(
        f"SELECT COUNT(*) FROM variable_questions vq "
        f"JOIN variables v ON v.table_id=vq.table_id AND v.dataset=vq.dataset "
        f"AND v.year=vq.year AND v.variable_id=vq.var_id "
        f"WHERE v.is_commonly_queried=1 AND v.kind='estimate' "
        f"AND vq.position < {opts['var_questions_cap']} {depth_clause}"
    ).fetchone()[0]
    out["concept_map_alias"] = 0   # quick-and-dirty: skip count
    out["total"] = sum(out.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--metadata-db", default=str(DEFAULT_METADATA_DB),
        help=f"(default: {DEFAULT_METADATA_DB})",
    )
    ap.add_argument(
        "--include-non-cq-vars", action="store_true",
        help="Embed label_natural for all variables, not just commonly_queried.",
    )
    ap.add_argument(
        "--var-questions-cap", type=int, default=3,
        help="Keep only first N questions per variable (default 3).",
    )
    ap.add_argument(
        "--max-var-depth", type=int, default=None,
        help="Drop var_question cards for variables with depth > N.",
    )
    ap.add_argument(
        "--no-concept-map", action="store_true",
        help="Skip concept_map curation cards.",
    )
    ap.add_argument(
        "--append", action="store_true",
        help="Add to existing cards instead of truncating first.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Show planned counts only.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not Path(args.metadata_db).exists():
        logger.error("Metadata DB missing: %s", args.metadata_db)
        return 1

    stats = run_build(
        Path(args.metadata_db),
        include_non_cq_vars=args.include_non_cq_vars,
        var_questions_cap=args.var_questions_cap,
        max_var_depth=args.max_var_depth,
        use_concept_map=not args.no_concept_map,
        append=args.append,
        dry_run=args.dry_run,
    )

    print("\nCard build summary:")
    for k, v in stats.items():
        print(f"  {k:30s} {v:>12,}")

    if not args.dry_run:
        # Quick size estimate
        total = stats.get("total", 0)
        raw_gb_f32 = total * 768 * 4 / 1024**3
        raw_gb_f16 = total * 768 * 2 / 1024**3
        print(f"\n  Embedding size projection:")
        print(f"    float32: {raw_gb_f32:.2f} GB raw, "
              f"~{raw_gb_f32 * 1.3:.2f} GB FAISS HNSW total")
        print(f"    float16: {raw_gb_f16:.2f} GB raw, "
              f"~{raw_gb_f16 * 1.3:.2f} GB FAISS HNSW total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
