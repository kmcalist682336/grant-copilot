"""Tier 0.5: SQLite cache for composite-concept decompositions.

Composite concepts ("gentrification", "fresh food access", "income
inequality") don't map to a single Census table. The Phase G/H planner
breaks them down into ≤ 5 measurable proxy concepts that DO map (e.g.,
gentrification → median home value + rent + racial composition + edu +
income). LLM decomposition is expensive and the same composite shows up
across many user queries, so we cache it persistently.

This module is the storage layer only. The LLM call that *creates* a
``Decomposition`` lives next to the planner so it can use planner
context. Phase J pre-seeds the cache from a curated YAML list.

Public API:
    Decomposition         — Pydantic record stored per concept
    DecompositionCache    — SQLite-backed get/set/list/purge
    normalize_concept(s)  — exposed for the seeder + tests
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


_DEFAULT_TTL_SECONDS = 90 * 24 * 60 * 60   # 90 days; aligns with chatbot.yaml
MAX_SUB_CONCEPTS = 5                       # see chatbot.yaml: max_sub_concepts


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decomposition_cache (
    concept_hash       TEXT PRIMARY KEY,
    concept            TEXT NOT NULL,
    sub_concepts_json  TEXT NOT NULL,
    rationale          TEXT,
    confidence         REAL NOT NULL,
    model_version      TEXT,
    created_at         REAL NOT NULL,
    ttl_seconds        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decomp_created_at
    ON decomposition_cache(created_at);
"""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_concept(text: str) -> str:
    """Canonicalize a composite-concept string for cache keying.

    Lowercase + collapse whitespace + strip leading/trailing punctuation.
    Keeps internal punctuation (so "gentrification (re)development" still
    hashes distinctly from "gentrification redevelopment") but normalizes
    the trivial casing/whitespace variants users will type.
    """
    s = text.lower().strip()
    s = _WS_RE.sub(" ", s)
    return s.strip(".,;:!?()[]{}\"'`")


def _hash_concept(text: str) -> str:
    return hashlib.sha256(normalize_concept(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pydantic record
# ---------------------------------------------------------------------------

class Decomposition(BaseModel):
    """One composite-concept → sub-concept decomposition."""

    model_config = ConfigDict(extra="ignore")

    concept: str = Field(
        ..., description="Original composite concept text (verbatim).",
    )
    sub_concepts: list[str] = Field(
        ...,
        description=(
            "Measurable proxy concepts that, taken together, approximate "
            f"the composite. Capped at {MAX_SUB_CONCEPTS}."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="LLM-reported confidence in the decomposition.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Brief LLM explanation of why these proxies were chosen.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="LLM identifier that produced this entry "
                    "(e.g. 'gemini-2.5-flash@2026-04-01').",
    )

    @field_validator("sub_concepts")
    @classmethod
    def _cap_and_dedupe(cls, v: list[str]) -> list[str]:
        # Strip + drop empties + dedupe (case-insensitive) preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            s = (item or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        if not out:
            raise ValueError("sub_concepts must contain at least one item")
        if len(out) > MAX_SUB_CONCEPTS:
            raise ValueError(
                f"sub_concepts has {len(out)} items; cap is {MAX_SUB_CONCEPTS}"
            )
        return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class DecompositionCache:
    """KV cache for composite-concept decompositions, keyed on
    normalized concept text. Mirrors APICache's connection pattern."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=10.0,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Get / Set / Delete
    # ------------------------------------------------------------------

    def get(
        self, concept: str, *, now: Optional[float] = None,
    ) -> Optional[Decomposition]:
        """Return the cached Decomposition if present and not expired.

        Cache miss returns None. Lazy-purges expired entries on read so
        the next caller sees a clean state.
        """
        now = now if now is not None else time.time()
        h = _hash_concept(concept)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT concept, sub_concepts_json, rationale, confidence, "
                "       model_version, created_at, ttl_seconds "
                "FROM decomposition_cache WHERE concept_hash = ?",
                (h,),
            ).fetchone()
        if not row:
            return None
        (concept_stored, sub_json, rationale, confidence,
         model_version, created_at, ttl_seconds) = row
        if now - created_at > ttl_seconds:
            self.delete(concept)
            return None
        try:
            sub_concepts = json.loads(sub_json)
        except json.JSONDecodeError as e:
            logger.warning(
                "Decomposition cache row for %r had bad JSON: %s",
                concept[:60], e,
            )
            self.delete(concept)
            return None
        try:
            return Decomposition(
                concept=concept_stored,
                sub_concepts=sub_concepts,
                confidence=float(confidence),
                rationale=rationale,
                model_version=model_version,
            )
        except Exception as e:  # pragma: no cover — safety net
            logger.warning(
                "Cached row for %r failed validation, evicting: %s",
                concept[:60], e,
            )
            self.delete(concept)
            return None

    def set(
        self, decomp: Decomposition, *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        now: Optional[float] = None,
    ) -> None:
        """Insert or replace a cached decomposition."""
        now = now if now is not None else time.time()
        h = _hash_concept(decomp.concept)
        body = json.dumps(decomp.sub_concepts, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decomposition_cache "
                "(concept_hash, concept, sub_concepts_json, rationale, "
                " confidence, model_version, created_at, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    h, decomp.concept, body, decomp.rationale,
                    float(decomp.confidence), decomp.model_version,
                    now, ttl_seconds,
                ),
            )
            conn.commit()

    def delete(self, concept: str) -> None:
        h = _hash_concept(concept)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM decomposition_cache WHERE concept_hash = ?",
                (h,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Inspection / Maintenance
    # ------------------------------------------------------------------

    def list_all(self) -> list[Decomposition]:
        """Dump every cached entry. Used by the Phase J seeder + manual
        inspection. Order is insertion order (created_at ASC)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT concept, sub_concepts_json, rationale, confidence, "
                "       model_version "
                "FROM decomposition_cache ORDER BY created_at ASC"
            ).fetchall()
        out: list[Decomposition] = []
        for r in rows:
            try:
                out.append(Decomposition(
                    concept=r[0],
                    sub_concepts=json.loads(r[1]),
                    confidence=float(r[3]),
                    rationale=r[2],
                    model_version=r[4],
                ))
            except Exception as e:
                logger.warning("Skipping malformed cache row %r: %s",
                               r[0][:60], e)
        return out

    def purge_expired(self, *, now: Optional[float] = None) -> int:
        """Delete every expired row. Returns count deleted."""
        now = now if now is not None else time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM decomposition_cache "
                "WHERE created_at + ttl_seconds < ?",
                (now,),
            )
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(AVG(confidence), 0.0) "
                "FROM decomposition_cache"
            ).fetchone()
        return {
            "entries": row[0],
            "avg_confidence": round(float(row[1]), 3),
            "path": str(self.path),
        }
