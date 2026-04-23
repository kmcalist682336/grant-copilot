"""SQLite-backed cache for Census API responses.

Schema is intentionally minimal: keyed on (url) with TTL stored per row.
Designed to survive across process restarts (POC) and easy to swap for
Redis later in production.

Thread-safe via SQLite's own locking; safe for asyncio because each call
is short and we use ``check_same_thread=False`` carefully (no shared
cursors across tasks).
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_cache (
    url_hash             TEXT PRIMARY KEY,
    url                  TEXT NOT NULL,
    response_json        TEXT NOT NULL,
    fetched_at           REAL NOT NULL,
    ttl_seconds          INTEGER NOT NULL,
    response_size_bytes  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_cache_fetched_at
    ON api_cache(fetched_at);
"""


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class APICache:
    """KV cache for Census API responses with per-key TTL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        # check_same_thread=False — we open a fresh connection per call
        # which is cheap; safer than sharing a long-lived one across
        # asyncio tasks. WAL mode lets reads happen during writes.
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
    # Get / Set
    # ------------------------------------------------------------------

    def get(self, url: str, *, now: Optional[float] = None) -> Optional[Any]:
        """Return the cached response if present and not expired."""
        now = now if now is not None else time.time()
        url_hash = _hash_url(url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json, fetched_at, ttl_seconds "
                "FROM api_cache WHERE url_hash = ?",
                (url_hash,),
            ).fetchone()
        if not row:
            return None
        response_json, fetched_at, ttl_seconds = row
        if now - fetched_at > ttl_seconds:
            # Lazy-purge: stale entries get deleted on next read
            self.delete(url)
            return None
        try:
            return json.loads(response_json)
        except json.JSONDecodeError as e:
            logger.warning("Cache row for %s had bad JSON: %s", url[:60], e)
            self.delete(url)
            return None

    def set(
        self, url: str, response: Any, *, ttl_seconds: int,
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.time()
        url_hash = _hash_url(url)
        body = json.dumps(response, separators=(",", ":"))
        size = len(body.encode("utf-8"))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache "
                "(url_hash, url, response_json, fetched_at, ttl_seconds, "
                " response_size_bytes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url_hash, url, body, now, ttl_seconds, size),
            )
            conn.commit()

    def delete(self, url: str) -> None:
        url_hash = _hash_url(url)
        with self._connect() as conn:
            conn.execute("DELETE FROM api_cache WHERE url_hash = ?", (url_hash,))
            conn.commit()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self, *, now: Optional[float] = None) -> int:
        """Delete every expired row. Returns count deleted."""
        now = now if now is not None else time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM api_cache "
                "WHERE fetched_at + ttl_seconds < ?",
                (now,),
            )
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(response_size_bytes), 0) "
                "FROM api_cache"
            ).fetchone()
        return {
            "entries": row[0],
            "total_bytes": row[1],
            "total_mb": round(row[1] / 1024 / 1024, 2),
            "path": str(self.path),
        }
