"""Tier-2 metadata search tests — exercise FTS5 over the real index."""
from __future__ import annotations

import pytest

from scripts.chatbot.metadata_search import (
    CoverageRow, TableCandidate, VariableInfo,
    find_supported_years, get_coverage, get_variables,
    has_successful_coverage, search_tables,
)


# ---------------------------------------------------------------------------
# search_tables — happy paths
# ---------------------------------------------------------------------------

def test_median_household_income_finds_b19013(metadata_db):
    out = search_tables(metadata_db, "median household income", top_k=10)
    assert out
    # B19013 is THE canonical median household income table; should be in
    # the top 10 across at least one (dataset, year) combination.
    table_ids = {c.table_id for c in out}
    assert "B19013" in table_ids, (
        f"expected B19013 among top-10; got {sorted(table_ids)}"
    )

    # Each candidate must carry parsed metadata, not raw JSON strings.
    for c in out:
        assert isinstance(c, TableCandidate)
        assert isinstance(c.topic_tags, list)
        assert isinstance(c.sample_questions, list)
        assert isinstance(c.geography_levels, list)
        assert c.bm25_score < 0  # FTS5 BM25 is negative


def test_results_ranked_by_bm25_ascending(metadata_db):
    out = search_tables(metadata_db, "vehicle availability", top_k=8)
    assert len(out) >= 2
    scores = [c.bm25_score for c in out]
    assert scores == sorted(scores), (
        f"expected ascending bm25 (lower = better); got {scores}"
    )


def test_top_k_caps_results(metadata_db):
    out = search_tables(metadata_db, "income", top_k=3)
    assert len(out) <= 3


def test_snippet_highlights_query_term(metadata_db):
    out = search_tables(metadata_db, "poverty", top_k=5)
    assert out
    # At least one snippet should carry the FTS marker around 'poverty'
    # (we asked for '<<' / '>>').
    assert any(
        c.snippet and ("<<" in c.snippet and ">>" in c.snippet)
        for c in out
    )


# ---------------------------------------------------------------------------
# search_tables — filters
# ---------------------------------------------------------------------------

def test_dataset_filter_excludes_other_datasets(metadata_db):
    out = search_tables(
        metadata_db, "median household income",
        dataset="acs/acs5", top_k=10,
    )
    assert out
    assert all(c.dataset == "acs/acs5" for c in out), (
        f"got datasets: {sorted({c.dataset for c in out})}"
    )


def test_year_filter_returns_only_matching_year(metadata_db):
    out = search_tables(
        metadata_db, "income",
        year=2022, top_k=10,
    )
    assert out
    assert all(c.year == 2022 for c in out)


def test_dataset_and_year_combined(metadata_db):
    out = search_tables(
        metadata_db, "income",
        dataset="acs/acs5", year=2022, top_k=10,
    )
    assert out
    assert all(c.dataset == "acs/acs5" and c.year == 2022 for c in out)


# ---------------------------------------------------------------------------
# search_tables — edge cases
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty(metadata_db):
    assert search_tables(metadata_db, "", top_k=5) == []


def test_punctuation_only_query_returns_empty(metadata_db):
    assert search_tables(metadata_db, "!!! ??? ...", top_k=5) == []


def test_gibberish_query_returns_empty(metadata_db):
    out = search_tables(metadata_db, "xyzzzqq notarealconcept", top_k=5)
    assert out == []


def test_quotes_in_query_dont_break_fts(metadata_db):
    """User-supplied quotes/parens shouldn't crash FTS5 syntax — they
    should be stripped and the query handled gracefully."""
    out = search_tables(
        metadata_db, '"median" (household) income',
        top_k=5,
    )
    assert out
    assert any("B19013" == c.table_id for c in out)


# ---------------------------------------------------------------------------
# get_variables
# ---------------------------------------------------------------------------

def test_get_variables_returns_known_table(metadata_db):
    """B19013 in ACS5/2022 should have at least the canonical estimate
    variable and the GEO_ID column."""
    out = get_variables(metadata_db, "B19013", "acs/acs5", 2022)
    assert out, "expected B19013/acs5/2022 to have variables"
    var_ids = {v.variable_id for v in out}
    assert "B19013_001E" in var_ids, (
        f"missing canonical estimate var; got {sorted(var_ids)[:10]}"
    )

    for v in out:
        assert isinstance(v, VariableInfo)
        assert v.table_id == "B19013"
        assert v.dataset == "acs/acs5"
        assert v.year == 2022


def test_get_variables_unknown_table_returns_empty(metadata_db):
    out = get_variables(metadata_db, "ZZZ99999", "acs/acs5", 2022)
    assert out == []


def test_get_variables_wrong_dataset_returns_empty(metadata_db):
    """B19013 exists in ACS but not in decennial — wrong dataset → []."""
    out = get_variables(metadata_db, "B19013", "dec/dhc", 2020)
    assert out == []


# ---------------------------------------------------------------------------
# coverage helpers (populated by tools/import_v1_coverage.py + probe)
# ---------------------------------------------------------------------------

def test_get_coverage_returns_row_for_known_combo(metadata_db):
    """B19013 / acs/acs5 / 2022 / county was successfully fetched in v1."""
    cov = get_coverage(metadata_db, "B19013", "acs/acs5", 2022, "county")
    assert cov is not None
    assert isinstance(cov, CoverageRow)
    assert cov.status == "success"
    assert cov.row_count is not None and cov.row_count > 100  # ~159 GA counties
    assert cov.source in ("v1_stage8", "msa_probe")


def test_get_coverage_returns_none_for_unknown_combo(metadata_db):
    cov = get_coverage(metadata_db, "ZZZ99999", "acs/acs5", 2022, "county")
    assert cov is None


def test_has_successful_coverage_all_levels_present(metadata_db):
    """B19013 acs/acs5 2022 has tract + county + state coverage in v1."""
    assert has_successful_coverage(
        metadata_db, "B19013", "acs/acs5", 2022,
        ["tract", "county", "state"],
    ) is True


def test_has_successful_coverage_missing_level_false(metadata_db):
    """ZCTA was GA-only and limited; B19013 doesn't have it."""
    assert has_successful_coverage(
        metadata_db, "B19013", "acs/acs5", 2022,
        ["tract", "county", "zcta"],
    ) is False


def test_has_successful_coverage_empty_list_is_true(metadata_db):
    assert has_successful_coverage(
        metadata_db, "B19013", "acs/acs5", 2022, [],
    ) is True


def test_find_supported_years_returns_descending(metadata_db):
    yrs = find_supported_years(
        metadata_db, "B19013", "acs/acs5",
        ["tract", "county", "msa", "state"],
    )
    assert yrs, "expected at least one supported year"
    assert yrs == sorted(yrs, reverse=True)
    assert all(2020 <= y <= 2030 for y in yrs)


def test_find_supported_years_no_match_returns_empty(metadata_db):
    yrs = find_supported_years(
        metadata_db, "ZZZ99999", "acs/acs5",
        ["tract", "county"],
    )
    assert yrs == []


# ---------------------------------------------------------------------------
# search_tables coverage filter
# ---------------------------------------------------------------------------

def test_require_coverage_at_excludes_uncovered_tables(metadata_db):
    """Decennial tables don't have tract+county+msa+state coverage so they
    should drop out when the filter is set. (The unfiltered top-10
    currently includes ``dec/dhcas/2020/PBG43`` and similar.)"""
    unfiltered = search_tables(
        metadata_db, "median household income", top_k=10,
    )
    filtered = search_tables(
        metadata_db, "median household income",
        require_coverage_at=["tract", "county", "msa", "state"],
        top_k=10,
    )
    assert unfiltered, "sanity"
    assert filtered, "sanity"

    # Every filtered row must actually pass the coverage test.
    for t in filtered:
        assert has_successful_coverage(
            metadata_db, t.table_id, t.dataset, t.year,
            ["tract", "county", "msa", "state"],
        ), (
            f"filtered row failed re-check: "
            f"{t.dataset}/{t.year}/{t.table_id}"
        )

    # The unfiltered top-10 includes non-tract decennial datasets; the
    # filtered top-10 must drop them.
    assert any("dec/" in t.dataset for t in unfiltered), (
        "test premise: unfiltered top-10 should contain decennial hits"
    )
    assert all("dec/" not in t.dataset for t in filtered), (
        f"expected ACS-only after filter; got datasets="
        f"{sorted({t.dataset for t in filtered})}"
    )


def test_require_coverage_at_with_impossible_level_returns_empty(metadata_db):
    """Asking for a geo level no row supports yields no results."""
    out = search_tables(
        metadata_db, "median household income",
        require_coverage_at=["impossible_level_zzzqq"], top_k=10,
    )
    assert out == []


def test_require_coverage_at_combines_with_dataset_filter(metadata_db):
    out = search_tables(
        metadata_db, "median household income",
        dataset="acs/acs5",
        require_coverage_at=["tract", "county"],
        top_k=10,
    )
    assert out
    assert all(t.dataset == "acs/acs5" for t in out)
