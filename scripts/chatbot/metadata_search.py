"""Tier 2: FTS5-backed search over Census table & variable metadata.

The bootstrap step (``tools/bootstrap_from_old.py``) builds
``data/metadata/search_index.db``: a SQLite database with three tables:

  - ``tables``       — one row per (table_id, dataset, year), with full
                       metadata (title, description, use_when, avoid_when,
                       universe, topic_tags, sample_questions,
                       geography_levels, variable_count).
  - ``variables``    — one row per (table_id, dataset, year, variable_id);
                       label/concept fields are sparse but the
                       variable_id list is complete.
  - ``table_search`` — FTS5 virtual table over the searchable text columns
                       of ``tables``; tokenized with unicode61 + diacritics
                       stripping.

Public API:
    open_search_index(path)         — sqlite3.Connection (read-only)
    search_tables(db, query, ...)   — ranked TableCandidate list
    get_variables(db, table_id, dataset, year)
                                    — VariableInfo list for a chosen table
    get_coverage(db, table_id, dataset, year, geo_level)
                                    — single-combo coverage status
    has_successful_coverage(db, table_id, dataset, year, geo_levels)
                                    — convenience: success at every level?

Coverage table (created by ``tools/import_v1_coverage.py`` and filled by
``tools/probe_msa_coverage.py``) records whether each (table, dataset,
year, geo_level) combination actually returns rows from the Census API.
``search_tables`` accepts ``require_coverage_at`` to filter to only
combinations known to work.

Design: this layer is rank-only beyond the coverage filter. Dataset
preferences (e.g., ACS5 over ACS1 for tract-level queries) belong in
the Phase H planner with full geographic + temporal context.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic outputs
# ---------------------------------------------------------------------------

class TableCandidate(BaseModel):
    """A single Census table that matched the FTS query."""

    model_config = ConfigDict(extra="ignore")

    table_id: str
    dataset: str           # e.g. "acs/acs5", "dec/dhc"
    year: int

    title: str
    description: Optional[str] = None
    use_when: Optional[str] = None
    avoid_when: Optional[str] = None
    universe: Optional[str] = None

    topic_tags: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Parsed topic_tags_json: [{tag, confidence, rationale}, ...]",
    )
    sample_questions: list[str] = Field(default_factory=list)
    geography_levels: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Levels Census exposes the table at, parsed from JSON.",
    )
    variable_count: int = 0

    bm25_score: float = Field(
        ...,
        description="FTS5 BM25 score. Negative; lower = better. Ties common.",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="FTS5 snippet showing the matched text in context.",
    )


class VariableInfo(BaseModel):
    """One Census variable within a table — for the Phase H planner."""

    model_config = ConfigDict(extra="ignore")

    variable_id: str       # e.g. "B19013_001E", "GEO_ID"
    table_id: str
    dataset: str
    year: int
    label: str = ""
    concept: str = ""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def open_search_index(db_path: Path | str) -> sqlite3.Connection:
    """Open the metadata search DB. Read-only is implied (we never write)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Tuned for read-heavy access, similar to gazetteer_db.
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_json_loads(s: Optional[str], default: Any) -> Any:
    if s is None or s == "":
        return default
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        logger.debug("metadata JSON parse failed: %r", s[:120] if s else s)
        return default


def _normalize_sample_questions(raw: Any) -> list[str]:
    """sample_questions_json is heterogeneous across the bootstrap: some
    tables store ``["What is X?", ...]`` while others store
    ``[{"text": "What is X?", "categories": "..."}, ...]``. Flatten to a
    plain list[str] so downstream code doesn't have to branch."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


def _build_match_expression(query: str) -> str:
    """Wrap user text into a permissive FTS5 MATCH expression.

    Strategy: drop punctuation FTS5 treats as syntax (quotes, parens,
    operators), tokenize, AND-join the remaining terms with prefix
    matching on the last token. Mirrors the geo-side tokenizer's
    'phrase-quote each token' approach so multi-word concepts like
    "median household income" rank tightly.

    A query that's all-punctuation or empty returns "" — caller treats
    that as no-results.
    """
    # Strip FTS5 special chars; keep alphanumerics, hyphens, apostrophes.
    cleaned = []
    for ch in query:
        if ch.isalnum() or ch in " -'":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    norm = " ".join("".join(cleaned).split())
    if not norm:
        return ""
    tokens = norm.split()
    if len(tokens) == 1:
        return f'"{tokens[0]}" OR "{tokens[0]}"*'
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " ".join(quoted)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_tables(
    db: sqlite3.Connection,
    query: str,
    *,
    dataset: Optional[str] = None,
    year: Optional[int] = None,
    top_k: int = 10,
    require_coverage_at: Optional[list[str]] = None,
) -> list[TableCandidate]:
    """Rank Census tables by FTS5 relevance to ``query``.

    Args:
        db: Open connection from ``open_search_index``.
        query: Free-text concept (e.g. "median household income",
            "vehicle availability", "race by age").
        dataset: Optional exact-match filter on the dataset column
            ("acs/acs5", "dec/dhc", "acs/acs1/subject", ...).
        year: Optional exact-match filter on the year column.
        top_k: Max results to return.
        require_coverage_at: Optional list of geo_levels (e.g.,
            ``["tract", "county"]``). Only tables with a successful
            ``coverage`` row at *every* listed geo_level are returned.
            Use this to prune candidates Census doesn't actually serve
            at the geographies the user is asking about.

    Returns:
        Best-first ``TableCandidate`` list. Empty when the query produces
        no FTS hits or contains no searchable tokens (or when the
        coverage filter rules everything out).
    """
    match_expr = _build_match_expression(query)
    if not match_expr:
        return []

    where_extras: list[str] = []
    params: list[Any] = [match_expr]
    if dataset is not None:
        where_extras.append("fs.dataset = ?")
        params.append(dataset)
    if year is not None:
        where_extras.append("fs.year = ?")
        params.append(year)
    if require_coverage_at:
        # For every requested geo_level, require a successful coverage
        # row — implemented as a correlated EXISTS per level so the
        # planner can ask for "tract AND county" and get nothing back
        # if either is missing.
        for lvl in require_coverage_at:
            where_extras.append(
                "EXISTS (SELECT 1 FROM coverage cov "
                "        WHERE cov.table_id = fs.table_id "
                "          AND cov.dataset  = fs.dataset "
                "          AND cov.year     = fs.year "
                "          AND cov.geo_level = ? "
                "          AND cov.status   = 'success')"
            )
            params.append(lvl)
    extra_clause = (" AND " + " AND ".join(where_extras)) if where_extras else ""
    params.append(top_k)

    sql = f"""
        SELECT
            fs.table_id                        AS table_id,
            fs.dataset                         AS dataset,
            fs.year                            AS year,
            t.title                            AS title,
            t.description                      AS description,
            t.use_when                         AS use_when,
            t.avoid_when                       AS avoid_when,
            t.universe                         AS universe,
            t.topic_tags_json                  AS topic_tags_json,
            t.sample_questions_json            AS sample_questions_json,
            t.geography_levels_json            AS geography_levels_json,
            t.variable_count                   AS variable_count,
            bm25(table_search)                 AS score,
            snippet(table_search, -1, '<<', '>>', '...', 12) AS snippet
        FROM table_search fs
        JOIN tables t
          ON t.table_id = fs.table_id
         AND t.dataset  = fs.dataset
         AND t.year     = fs.year
        WHERE table_search MATCH ?{extra_clause}
        ORDER BY bm25(table_search)
        LIMIT ?
    """

    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning("FTS5 query failed for %r: %s", query, e)
        return []

    out: list[TableCandidate] = []
    for r in rows:
        out.append(TableCandidate(
            table_id=r["table_id"],
            dataset=r["dataset"],
            year=int(r["year"]),
            title=r["title"] or "",
            description=r["description"],
            use_when=r["use_when"],
            avoid_when=r["avoid_when"],
            universe=r["universe"],
            topic_tags=_safe_json_loads(r["topic_tags_json"], []),
            sample_questions=_normalize_sample_questions(
                _safe_json_loads(r["sample_questions_json"], []),
            ),
            geography_levels=_safe_json_loads(
                r["geography_levels_json"], [],
            ),
            variable_count=int(r["variable_count"] or 0),
            bm25_score=float(r["score"]),
            snippet=r["snippet"],
        ))
    return out


class CoverageRow(BaseModel):
    """One coverage record — see tools/import_v1_coverage.py."""

    model_config = ConfigDict(extra="ignore")

    table_id: str
    dataset: str
    year: int
    geo_level: str
    status: str           # 'success' | 'no_data'
    row_count: Optional[int] = None
    source: str           # 'v1_stage8' | 'msa_probe'


def get_coverage(
    db: sqlite3.Connection,
    table_id: str,
    dataset: str,
    year: int,
    geo_level: str,
) -> Optional[CoverageRow]:
    """Look up the coverage row for one (table, dataset, year, geo_level)."""
    r = db.execute(
        "SELECT table_id, dataset, year, geo_level, status, row_count, source "
        "FROM coverage "
        "WHERE table_id = ? AND dataset = ? AND year = ? AND geo_level = ?",
        (table_id, dataset, year, geo_level),
    ).fetchone()
    if r is None:
        return None
    return CoverageRow(
        table_id=r["table_id"],
        dataset=r["dataset"],
        year=int(r["year"]),
        geo_level=r["geo_level"],
        status=r["status"],
        row_count=r["row_count"],
        source=r["source"],
    )


def find_supported_years(
    db: sqlite3.Connection,
    table_id: str,
    dataset: str,
    geo_levels: list[str],
) -> list[int]:
    """Return every year for which (table_id, dataset) has 'success'
    coverage at *every* listed geo_level. Sorted descending (most recent
    first). Empty list = the combo never has the requested levels at any
    vintage we know about.
    """
    if not geo_levels:
        # Without level constraints, any year with any coverage qualifies.
        rows = db.execute(
            "SELECT DISTINCT year FROM coverage "
            "WHERE table_id = ? AND dataset = ? AND status = 'success' "
            "ORDER BY year DESC",
            (table_id, dataset),
        ).fetchall()
        return [int(r["year"]) for r in rows]
    placeholders = ",".join("?" * len(geo_levels))
    expected = len(set(geo_levels))
    rows = db.execute(
        f"SELECT year FROM coverage "
        f"WHERE table_id = ? AND dataset = ? AND status = 'success' "
        f"  AND geo_level IN ({placeholders}) "
        f"GROUP BY year HAVING COUNT(DISTINCT geo_level) = ? "
        f"ORDER BY year DESC",
        (table_id, dataset, *geo_levels, expected),
    ).fetchall()
    return [int(r["year"]) for r in rows]


def has_successful_coverage(
    db: sqlite3.Connection,
    table_id: str,
    dataset: str,
    year: int,
    geo_levels: list[str],
) -> bool:
    """True iff every geo_level in ``geo_levels`` has a 'success' row.

    Use this from concept_map validation: "this concept's table must
    return rows at tract and county or it's not useful for us."
    """
    if not geo_levels:
        return True
    placeholders = ",".join("?" * len(geo_levels))
    r = db.execute(
        f"SELECT COUNT(DISTINCT geo_level) AS c FROM coverage "
        f"WHERE table_id = ? AND dataset = ? AND year = ? "
        f"  AND status = 'success' "
        f"  AND geo_level IN ({placeholders})",
        (table_id, dataset, year, *geo_levels),
    ).fetchone()
    return int(r["c"] or 0) == len(set(geo_levels))


def get_variables(
    db: sqlite3.Connection,
    table_id: str,
    dataset: str,
    year: int,
) -> list[VariableInfo]:
    """Return all variables for a given table at a given (dataset, year).

    Used by the planner once it picks a TableCandidate — the variable_id
    list is what the Census API needs in the ``get=`` parameter.
    """
    rows = db.execute(
        """
        SELECT variable_id, table_id, dataset, year, label, concept
        FROM variables
        WHERE table_id = ? AND dataset = ? AND year = ?
        ORDER BY variable_id
        """,
        (table_id, dataset, year),
    ).fetchall()
    return [
        VariableInfo(
            variable_id=r["variable_id"],
            table_id=r["table_id"],
            dataset=r["dataset"],
            year=int(r["year"]),
            label=r["label"] or "",
            concept=r["concept"] or "",
        )
        for r in rows
    ]
