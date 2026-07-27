"""Import manually written HMDA questions into search_index.db."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

METADATA_DB = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "search_index.db"
)

TABLE_QUESTIONS_CSV = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "hmda_table_questions.csv"
)

VARIABLE_QUESTIONS_CSV = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "hmda_variable_questions.csv"
)

DATASET = "hmda"


def clean_value(value):
    if pd.isna(value):
        return None

    return value


def create_question_tables(
    db: sqlite3.Connection,
) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS table_questions (
            question_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            table_id          TEXT NOT NULL,
            dataset           TEXT NOT NULL,
            question_text     TEXT NOT NULL,
            polarity          TEXT NOT NULL,
            question_type     TEXT,
            position          INTEGER,
            route_to_dataset  TEXT,
            notes             TEXT
        );

        CREATE TABLE IF NOT EXISTS variable_questions (
            question_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            table_id                TEXT NOT NULL,
            dataset                 TEXT NOT NULL,
            var_id                  TEXT NOT NULL,
            question_text           TEXT NOT NULL,
            polarity                TEXT NOT NULL,
            question_type           TEXT,
            position                INTEGER,
            required_filters_json   TEXT,
            route_to_variable_id    TEXT,
            notes                   TEXT
        );
        """
    )

    db.commit()


def import_table_questions(
    db: sqlite3.Connection,
) -> int:
    frame = pd.read_csv(TABLE_QUESTIONS_CSV)

    db.execute(
        "DELETE FROM table_questions WHERE dataset = ?",
        (DATASET,),
    )

    rows = [
        (
            row["table_id"],
            row["dataset"],
            row["question_text"],
            row["polarity"],
            clean_value(row["question_type"]),
            int(row["position"]),
            clean_value(row["route_to_dataset"]),
            clean_value(row["notes"]),
        )
        for _, row in frame.iterrows()
    ]

    db.executemany(
        """
        INSERT INTO table_questions (
            table_id,
            dataset,
            question_text,
            polarity,
            question_type,
            position,
            route_to_dataset,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def import_variable_questions(
    db: sqlite3.Connection,
) -> int:
    frame = pd.read_csv(VARIABLE_QUESTIONS_CSV)

    db.execute(
        "DELETE FROM variable_questions WHERE dataset = ?",
        (DATASET,),
    )

    rows = [
        (
            row["table_id"],
            row["dataset"],
            row["var_id"],
            row["question_text"],
            row["polarity"],
            clean_value(row["question_type"]),
            int(row["position"]),
            clean_value(row["required_filters_json"]),
            clean_value(row["route_to_variable_id"]),
            clean_value(row["notes"]),
        )
        for _, row in frame.iterrows()
    ]

    db.executemany(
        """
        INSERT INTO variable_questions (
            table_id,
            dataset,
            var_id,
            question_text,
            polarity,
            question_type,
            position,
            required_filters_json,
            route_to_variable_id,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def main() -> None:
    db = sqlite3.connect(METADATA_DB)

    try:
        create_question_tables(db)

        table_count = import_table_questions(db)
        variable_count = import_variable_questions(db)

        db.commit()

        print(f"Imported {table_count} table questions.")
        print(f"Imported {variable_count} variable questions.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()