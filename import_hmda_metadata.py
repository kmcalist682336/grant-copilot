from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

TABLES_CSV = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "hmda_tables.csv"
)

VARIABLES_CSV = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "hmda_variables.csv"
)

COVERAGE_CSV = (
    REPO_ROOT
    / "data"
    / "metadata"
    / "hmda_coverage.csv"
)

DATASET = "hmda"

def create_tables(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS tables (
            table_id          TEXT NOT NULL,
            dataset           TEXT NOT NULL,
            year              INTEGER NOT NULL,
            title             TEXT,
            description       TEXT,
            use_when          TEXT,
            avoid_when        TEXT,
            measure_type      TEXT,
            subject_area      TEXT,
            l1_domains_json   TEXT,

            PRIMARY KEY (table_id, dataset, year)
        );

        CREATE TABLE IF NOT EXISTS variables (
            table_id              TEXT NOT NULL,
            dataset               TEXT NOT NULL,
            year                  INTEGER NOT NULL,
            variable_id           TEXT NOT NULL,
            column_name           TEXT,
            storage_hash          TEXT,
            label_natural         TEXT,
            role                  TEXT,
            kind                  TEXT,
            is_terminal           INTEGER,
            is_commonly_queried   INTEGER,
            is_total              INTEGER,
            depth                 INTEGER,
            geo_id                TEXT,
            lowest_geo_level      TEXT,

            PRIMARY KEY (
                table_id,
                dataset,
                year,
                variable_id
            )
        );

        CREATE TABLE IF NOT EXISTS coverage (
            table_id      TEXT NOT NULL,
            dataset       TEXT NOT NULL,
            year          INTEGER NOT NULL,
            geo_level     TEXT NOT NULL,
            status        TEXT NOT NULL,

            PRIMARY KEY (
                table_id,
                dataset,
                year,
                geo_level
            )
        );
        """
    )

    db.commit()


def import_tables(
    db: sqlite3.Connection,
    csv_path: Path,
) -> int:
    frame = pd.read_csv(csv_path)

    frame = frame.where(
        pd.notna(frame),
        None,
    )

    db.execute(
        "DELETE FROM tables WHERE dataset = ?",
        (DATASET,),
    )

    rows = [
        (
            row["table_id"],
            row["dataset"],
            int(row["year"]),
            row.get("title"),
            row.get("description"),
            row.get("use_when"),
            row.get("avoid_when"),
            row.get("measure_type"),
            row.get("subject_area"),
            row.get("l1_domains_json"),
        )
        for _, row in frame.iterrows()
    ]

    db.executemany(
        """
        INSERT INTO tables (
            table_id,
            dataset,
            year,
            title,
            description,
            use_when,
            avoid_when,
            measure_type,
            subject_area,
            l1_domains_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def import_variables(
    db: sqlite3.Connection,
    csv_path: Path,
) -> int:
    frame = pd.read_csv(csv_path)

    frame = frame.where(
        pd.notna(frame),
        None,
    )

    db.execute(
        "DELETE FROM variables WHERE dataset = ?",
        (DATASET,),
    )

    rows = [
        (
            row["table_id"],
            row["dataset"],
            int(row["year"]),
            row["variable_id"],
            row.get("column_name"),
            row.get("storage_hash"),
            row.get("label_natural"),
            row.get("role"),
            row.get("kind"),
            int(row.get("is_terminal", 0)),
            int(row.get("is_commonly_queried", 1)),
            int(row.get("is_total", 0)),
            int(row.get("depth", 0)),
            row.get("geo_id"),
            row.get("lowest_geo_level"),
        )
        for _, row in frame.iterrows()
    ]

    db.executemany(
        """
        INSERT INTO variables (
            table_id,
            dataset,
            year,
            variable_id,
            column_name,
            storage_hash,
            label_natural,
            role,
            kind,
            is_terminal,
            is_commonly_queried,
            is_total,
            depth,
            geo_id,
            lowest_geo_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def import_coverage(
    db: sqlite3.Connection,
    csv_path: Path,
) -> int:
    frame = pd.read_csv(csv_path)

    db.execute(
        "DELETE FROM coverage WHERE dataset = ?",
        (DATASET,),
    )

    rows = [
        (
            row["table_id"],
            row["dataset"],
            int(row["year"]),
            row["geo_level"],
            row["status"],
        )
        for _, row in frame.iterrows()
    ]

    db.executemany(
        """
        INSERT INTO coverage (
            table_id,
            dataset,
            year,
            geo_level,
            status
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    return len(rows)


def main() -> None:
    db = sqlite3.connect(METADATA_DB)

    try:
        create_tables(db)

        table_count = import_tables(
            db,
            TABLES_CSV,
        )

        variable_count = import_variables(
            db,
            VARIABLES_CSV,
        )

        coverage_count = import_coverage(
            db,
            COVERAGE_CSV,
        )

        db.commit()

        print(f"Imported {table_count} table rows.")
        print(f"Imported {variable_count} variable rows.")
        print(f"Imported {coverage_count} coverage rows.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()