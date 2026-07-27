"""Build HMDA table-level routing cards"""

from __future__ import annotations

import sqlite3
import os

def build_table_description_cards(
    db: sqlite3.Connection,
    dataset: str,
) -> int:
    sql = """
    INSERT INTO cards (
        card_type,
        text,
        target_table_id,
        target_dataset,
        target_year,
        target_variable_id,
        weight,
        polarity,
        measure_type,
        subject_area,
        l1_domains
    )
    SELECT
        'table_description',
        TRIM(
            COALESCE(t.title, '') || '. ' ||
            COALESCE(t.description, '') || ' ' ||
            COALESCE(t.use_when, '')
        ),
        t.table_id,
        t.dataset,
        t.year,
        NULL,
        1.0,
        'positive',
        t.measure_type,
        t.subject_area,
        t.l1_domains_json
    FROM tables t
    JOIN (
        SELECT
            table_id,
            dataset,
            MAX(year) AS latest_year
        FROM tables
        WHERE dataset = ?
        GROUP BY table_id, dataset
    ) latest
      ON latest.table_id = t.table_id
     AND latest.dataset = t.dataset
     AND latest.latest_year = t.year
    """

    cur = db.execute(
        sql,
        (dataset,),
    )

    return cur.rowcount


def build_table_avoid_cards(
    db: sqlite3.Connection,
    dataset: str,
) -> int:
    sql = """
    INSERT INTO cards (
        card_type,
        text,
        target_table_id,
        target_dataset,
        target_year,
        target_variable_id,
        weight,
        polarity,
        measure_type,
        subject_area,
        l1_domains
    )
    SELECT
        'table_avoid_when',
        t.avoid_when,
        t.table_id,
        t.dataset,
        t.year,
        NULL,
        1.0,
        'negative',
        t.measure_type,
        t.subject_area,
        t.l1_domains_json
    FROM tables t
    JOIN (
        SELECT
            table_id,
            dataset,
            MAX(year) AS latest_year
        FROM tables
        WHERE dataset = ?
        GROUP BY table_id, dataset
    ) latest
      ON latest.table_id = t.table_id
     AND latest.dataset = t.dataset
     AND latest.latest_year = t.year
    WHERE t.avoid_when IS NOT NULL
      AND TRIM(t.avoid_when) != ''
    """

    cur = db.execute(
        sql,
        (dataset,),
    )

    return cur.rowcount


def build_table_question_cards(
    db: sqlite3.Connection,
    dataset: str,
) -> int:
    sql = """
    INSERT INTO cards (
        card_type,
        text,
        target_table_id,
        target_dataset,
        target_year,
        target_variable_id,
        weight,
        polarity,
        measure_type,
        subject_area,
        l1_domains
    )
    SELECT
        CASE
            WHEN tq.polarity = 'negative'
                THEN 'table_question_negative'
            WHEN tq.question_type = 'varied'
                THEN 'table_question_varied'
            ELSE 'table_question_canonical'
        END,
        tq.question_text,
        tq.table_id,
        tq.dataset,
        latest.latest_year,
        NULL,
        CASE
            WHEN tq.polarity = 'negative' THEN 1.0
            WHEN tq.question_type = 'varied' THEN 1.3
            ELSE 1.5
        END,
        tq.polarity,
        t.measure_type,
        t.subject_area,
        t.l1_domains_json
    FROM table_questions tq
    JOIN (
        SELECT
            table_id,
            dataset,
            MAX(year) AS latest_year
        FROM tables
        WHERE dataset = ?
        GROUP BY table_id, dataset
    ) latest
      ON latest.table_id = tq.table_id
     AND latest.dataset = tq.dataset
    JOIN tables t
      ON t.table_id = latest.table_id
     AND t.dataset = latest.dataset
     AND t.year = latest.latest_year
    WHERE tq.dataset = ?
    """

    cur = db.execute(
        sql,
        (dataset, dataset),
    )

    return cur.rowcount


def build_all_table_cards(
    db: sqlite3.Connection,
    dataset: str,
) -> dict[str, int]:
    stats = {
        "table_description": build_table_description_cards(
            db,
            dataset,
        ),
        "table_avoid_when": build_table_avoid_cards(
            db,
            dataset,
        ),
        "table_questions": build_table_question_cards(
            db,
            dataset,
        ),
    }

    db.commit()

    return stats