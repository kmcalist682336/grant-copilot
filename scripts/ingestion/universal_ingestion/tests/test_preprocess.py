"""Optional tests for the HMDA preprocessing environment.

Run explicitly after installing the ingestion dependencies:
    python -m pytest scripts/ingestion/universal_ingestion/tests -q
"""

from collections import defaultdict

import pandas as pd

from scripts.ingestion.universal_ingestion.preprocess_lar import (
    clean_column_name,
    decode_chunk,
    normalize_numeric_fields,
)


def test_clean_column_name_normalizes_hmda_repeated_fields():
    assert clean_column_name("co-applicant_age") == "co_applicant_age"
    assert clean_column_name("aus-3") == "aus_3"


def test_decode_replaces_codes_and_preserves_unknowns():
    chunk = pd.DataFrame({"action_taken": ["1", "999", ""]})
    unknown = defaultdict(set)

    result = decode_chunk(
        chunk,
        {"action_taken": {"1": "Loan originated"}},
        unknown,
    )

    assert result["action_taken"].tolist() == ["Loan originated", "999", ""]
    assert unknown == {"action_taken": {"999"}}


def test_numeric_normalization_separates_status_from_measure():
    chunk = pd.DataFrame({"interest_rate": ["6.25", "NA", "Exempt", ""]})
    invalid = defaultdict(set)

    normalize_numeric_fields(chunk, {"interest_rate": "float64"}, invalid)

    assert chunk["interest_rate"].tolist()[:1] == [6.25]
    assert chunk["interest_rate"].isna().tolist() == [False, True, True, True]
    assert chunk["interest_rate_status"].tolist()[:3] == [pd.NA, "Not applicable", "Exempt"]
    assert not invalid


def test_numeric_normalization_reports_unexpected_text():
    chunk = pd.DataFrame({"loan_amount": ["100000", "unknown"]})
    invalid = defaultdict(set)

    normalize_numeric_fields(chunk, {"loan_amount": "float64"}, invalid)

    assert invalid == {"loan_amount": {"unknown"}}
    assert chunk["loan_amount_status"].tolist() == [pd.NA, "unknown"]
