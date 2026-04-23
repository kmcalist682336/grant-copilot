"""FTS5 search + candidate model for the resolver.

A ``ResolutionCandidate`` is an internal staging object — a row from
``geo_aliases`` augmented with rank information and the target's subtype
(place_type / geo_type / road_type depending on ``target_type``).
The resolver turns the top-ranked candidate per reference into a
``ResolvedGeography`` with appropriate tract lookups.

Design notes:
  - FTS5 queries are scoped by ``target_type`` so each strategy searches
    the right corpus.
  - BM25 scores are negative (lower = better). We rank by position and
    expose the raw BM25 for tie-breaking / calibration downstream.
  - The subtype is joined via LEFT JOIN + COALESCE so the caller always
    gets a populated value regardless of target_type.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolutionCandidate:
    target_id: str
    target_type: str       # 'admin' | 'named_place' | 'road'
    target_name: str       # canonical display name
    state_fips: str
    rank_position: int     # 0-indexed; 0 = top hit
    alias_matched: str     # the alias row that matched
    alias_type: str        # 'exact' | 'common' | 'abbreviation' | ...
    bm25: float            # negative; lower = better
    subtype: Optional[str] = None  # place_type / geo_type / road_type


# Characters we strip before handing text to FTS5. We keep hyphens
# (preserves "I-85", "Poncey-Highland") and apostrophes (for names like
# "O'Brien"). Other punctuation gets collapsed to whitespace.
_STRIP_RE = re.compile(r"[^\w\s\-']+", re.UNICODE)


def normalize_query_text(text: str) -> str:
    """Strip punctuation (except - and '), collapse whitespace."""
    text = _STRIP_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def _build_fts_match(text: str) -> str:
    """Build an FTS5 MATCH expression from user text.

    Strategy: quote each token so hyphens don't split terms ("I-85" stays
    together), and require all tokens (AND semantics — FTS5 default).
    Adds prefix wildcards to the last token for partial matches.
    """
    norm = normalize_query_text(text)
    if not norm:
        return ""
    tokens = norm.split()
    if len(tokens) == 1:
        # Single token — allow exact or prefix
        return f'"{tokens[0]}" OR "{tokens[0]}"*'
    # Multi-token — phrase-quote all but last, prefix-match last
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " ".join(quoted)


def fts_search(
    db: sqlite3.Connection,
    text: str,
    *,
    target_type_filter: Optional[str] = None,
    limit: int = 10,
) -> list[ResolutionCandidate]:
    """Search geo_fts for text, return ranked candidates.

    Args:
        db: Open SpatiaLite connection (Row row_factory recommended).
        text: User-provided span; punctuation is normalized.
        target_type_filter: restrict to 'admin' | 'named_place' | 'road'.
        limit: Max candidates returned.

    Returns:
        List of ResolutionCandidate, ordered best-first. Empty if no hits.
    """
    match_expr = _build_fts_match(text)
    if not match_expr:
        return []

    # Lowercased, whitespace-trimmed user text for exact-alias-text
    # comparison in the ORDER BY. Lets us promote rows whose alias is
    # *literally* what the user typed above rows where the alias merely
    # contains the user's text as a substring.
    text_lc = text.strip().lower()

    params: list = [match_expr]
    type_clause = ""
    if target_type_filter is not None:
        type_clause = " AND f.target_type = ?"
        params.append(target_type_filter)
    params.append(text_lc)   # bound to the LOWER(f.alias) = ? clause
    params.append(limit)

    # Ranking notes:
    #
    # - text-exact tier: rows whose alias is literally what the user
    #   typed (case-insensitive) come first. This is what makes "Houston"
    #   surface the Houston MSA's principal_city alias ("Houston") above
    #   "Houston County" — the latter contains the query as a substring
    #   but isn't what the user wrote. The user's text is passed in as
    #   the second bound parameter for this comparison.
    #
    # - alias_type tier (within text-exact and within non-text-exact):
    #   'exact' canonical names rank above 'principal_city' (the MSA-
    #   shortcut aliases added by tools/add_msa_principal_city_aliases.py),
    #   which beats generic 'common'/'abbreviation'. Keeps "Atlanta" →
    #   Atlanta city (exact alias) but lets "Houston" surface the Houston
    #   MSA's principal_city when no same-name place/county is exact-text.
    #
    # - BM25 column weights: only `alias` carries weight. FTS5's BM25
    #   otherwise normalizes by the full row's token length, penalizing
    #   admin entries with long names like the MSAs.
    #
    # - We exclude admin geo_types we never resolve (urbanized_area,
    #   urban_cluster). These are duplicated heavily in the gazetteer
    #   and would otherwise crowd useful candidates out of the limit.
    #
    # - state='13' breaks remaining ties (GA-leaning default);
    #   scope-aware re-ranking happens in scoring.compute_confidence.
    sql = f"""
        SELECT
            f.alias           AS alias,
            f.target_id       AS target_id,
            f.target_type     AS target_type,
            f.target_name     AS target_name,
            f.state_fips      AS state_fips,
            a.alias_type      AS alias_type,
            bm25(geo_fts, 1.0, 0.0, 0.0, 0.0, 0.0) AS bm25_score,
            COALESCE(np.place_type, ag.geo_type, r.road_type) AS subtype
        FROM geo_fts f
        JOIN geo_aliases a ON a.alias_id = f.rowid
        LEFT JOIN named_places np
            ON np.place_id = f.target_id AND f.target_type = 'named_place'
        LEFT JOIN admin_geographies ag
            ON ag.geoid = f.target_id AND f.target_type = 'admin'
        LEFT JOIN roads r
            ON r.road_id = f.target_id AND f.target_type = 'road'
        WHERE geo_fts MATCH ?{type_clause}
          AND COALESCE(ag.geo_type, '') NOT IN ('urbanized_area', 'urban_cluster')
        ORDER BY CASE WHEN LOWER(f.alias) = ? THEN 0 ELSE 1 END,
                 CASE a.alias_type
                     WHEN 'exact'          THEN 0
                     WHEN 'principal_city' THEN 1
                     ELSE 2
                 END,
                 bm25(geo_fts, 1.0, 0.0, 0.0, 0.0, 0.0),
                 CASE WHEN f.state_fips = '13' THEN 0 ELSE 1 END
        LIMIT ?
    """

    try:
        rows = db.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        # Malformed MATCH expression — log and return empty rather than raise
        logger.warning("FTS5 query failed for %r: %s", text, e)
        return []

    return [
        ResolutionCandidate(
            target_id=r["target_id"],
            target_type=r["target_type"],
            target_name=r["target_name"],
            state_fips=r["state_fips"] or "",
            rank_position=i,
            alias_matched=r["alias"],
            alias_type=r["alias_type"] or "exact",
            bm25=float(r["bm25_score"]),
            subtype=r["subtype"],
        )
        for i, r in enumerate(rows)
    ]
