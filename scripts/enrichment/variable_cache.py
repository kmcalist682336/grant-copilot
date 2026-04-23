"""SQLite-backed cache for Stage 7 variable descriptions and questions.

Variables repeat across years — B17001_004E has the same label in 2020, 2021,
2022, etc. This cache stores LLM-generated descriptions and questions keyed
by a fingerprint derived from (prompt_version, dataset_path, table_id, var_id,
hash(label_raw)). Year is intentionally excluded from the fingerprint so that
unchanged variables share cache entries across years.

Bumping the prompt version constant invalidates all entries for that type,
forcing regeneration with the new prompt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Bump these when the prompt template or output schema changes.
DESCRIPTION_PROMPT_VERSION = "desc_v1"
QUESTION_PROMPT_VERSION = "quest_v1"


def compute_fingerprint(
    prompt_version: str,
    dataset_path: str,
    table_id: str,
    var_id: str,
    label_raw: str,
) -> str:
    """Stable fingerprint for a variable's prompt input."""
    label_hash = hashlib.sha256(label_raw.encode("utf-8")).hexdigest()[:16]
    return f"{prompt_version}:{dataset_path}:{table_id}:{var_id}:{label_hash}"


class VariableCache:
    """SQLite cache for variable descriptions and questions.

    Thread-safe; uses WAL mode so concurrent readers don't block writers.
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS variable_descriptions (
        fingerprint TEXT PRIMARY KEY,
        dataset_path TEXT NOT NULL,
        table_id TEXT NOT NULL,
        var_id TEXT NOT NULL,
        label_natural TEXT NOT NULL,
        description TEXT NOT NULL,
        column_name TEXT NOT NULL,
        model_id TEXT,
        generated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS variable_questions (
        fingerprint TEXT PRIMARY KEY,
        dataset_path TEXT NOT NULL,
        table_id TEXT NOT NULL,
        var_id TEXT NOT NULL,
        questions_json TEXT NOT NULL,
        model_id TEXT,
        generated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_desc_table
        ON variable_descriptions (dataset_path, table_id);
    CREATE INDEX IF NOT EXISTS idx_quest_table
        ON variable_questions (dataset_path, table_id);
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None,  # autocommit
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def _transaction(self):
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(self.SCHEMA_SQL)

    # ---- description entries ----

    def get_description(self, fingerprint: str) -> Optional[dict]:
        cur = self._get_conn().execute(
            "SELECT label_natural, description, column_name, model_id "
            "FROM variable_descriptions WHERE fingerprint = ?",
            (fingerprint,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "label_natural": row[0],
            "description": row[1],
            "column_name": row[2],
            "model_id": row[3],
        }

    def put_descriptions_batch(self, entries: list[dict]) -> None:
        """Bulk insert descriptions. Each entry has keys: fingerprint,
        dataset_path, table_id, var_id, label_natural, description,
        column_name, model_id."""
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                e["fingerprint"], e["dataset_path"], e["table_id"],
                e["var_id"], e["label_natural"], e["description"],
                e["column_name"], e.get("model_id"), now,
            )
            for e in entries
        ]
        with self._lock, self._transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO variable_descriptions "
                "(fingerprint, dataset_path, table_id, var_id, "
                " label_natural, description, column_name, model_id, generated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    # ---- question entries ----

    def get_questions(self, fingerprint: str) -> Optional[list[str]]:
        cur = self._get_conn().execute(
            "SELECT questions_json FROM variable_questions WHERE fingerprint = ?",
            (fingerprint,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_questions_batch(self, entries: list[dict]) -> None:
        """Bulk insert questions. Each entry has keys: fingerprint,
        dataset_path, table_id, var_id, questions (list), model_id."""
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                e["fingerprint"], e["dataset_path"], e["table_id"],
                e["var_id"],
                json.dumps(e["questions"], ensure_ascii=False),
                e.get("model_id"), now,
            )
            for e in entries
        ]
        with self._lock, self._transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO variable_questions "
                "(fingerprint, dataset_path, table_id, var_id, "
                " questions_json, model_id, generated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    # ---- stats ----

    def stats(self) -> dict:
        conn = self._get_conn()
        desc_count = conn.execute(
            "SELECT COUNT(*) FROM variable_descriptions"
        ).fetchone()[0]
        quest_count = conn.execute(
            "SELECT COUNT(*) FROM variable_questions"
        ).fetchone()[0]
        return {
            "description_entries": desc_count,
            "question_entries": quest_count,
        }
