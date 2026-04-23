"""Tests for the derived-data dump — verifies CSV shape per geo_level,
variables.txt sidecar content, and graceful no-op on empty results."""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.census_caller import APIPlanCall, FetchResult
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.derived_data_dump import dump_derived_data
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.planner import PlannedCall, PlanResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _geo(geo_id, name, level="county", tracts=None):
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region",
    ) else level
    ref_type = "neighborhood" if level == "neighborhood" else "administrative"
    return ResolvedGeography(
        geo_id=geo_id, geo_level=level, geo_type=level,
        display_name=name, tract_geoids=tracts or [],
        county_geoid=None, api_for_clause=f"{level}:*",
        api_in_clause="", confidence=0.95, assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type=ref_type),
    )


def _api(table_id, dataset, geo_level, variables, year=2024):
    return APIPlanCall(
        url=f"https://example.com/{table_id}", table_id=table_id,
        variables=variables, geo_level=geo_level,
        geo_filter_ids=[], year=year, dataset=dataset, ttl_seconds=3600,
    )


def _fetch(api_call: APIPlanCall, rows: list[dict]) -> FetchResult:
    return FetchResult(
        plan=api_call, rows=rows, cache_hit=False, elapsed_s=0.1,
    )


def _metadata_db_with_labels(labels):
    """In-memory sqlite that mimics search_index.db.variables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE variables (
            table_id TEXT, dataset TEXT, year INTEGER,
            variable_id TEXT, label_natural TEXT,
            PRIMARY KEY (table_id, dataset, year, variable_id)
        )
    """)
    for (vid, dataset, year, table_id, label) in labels:
        conn.execute(
            "INSERT INTO variables VALUES (?,?,?,?,?)",
            (table_id, dataset, year, vid, label),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Happy path — county-only query, one row
# ---------------------------------------------------------------------------

def test_county_query_writes_relevant_variables_county(tmp_path: Path):
    dekalb = _geo("13089", "DeKalb County", level="county")
    concept = ExtractedConcept(text="median household income",
                                canonical_hint="median household income")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(value="B19013_001E")
    api = _api("B19013", "acs/acs5", "county", ["B19013_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[dekalb], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="primary", variables=cv,
        )],
    )
    aggregated = AggregatedResult(values=[
        AggregatedValue(geo=dekalb, concept=concept, year=2024,
                        role="primary", value=75000.0, rows_aggregated=1),
    ])
    fetch_results = [_fetch(api, [
        {"B19013_001E": "75000", "state": "13", "county": "089"},
    ])]
    md = _metadata_db_with_labels([
        ("B19013_001E", "acs/acs5", 2024, "B19013", "Median household income"),
    ])

    folder = tmp_path / f"data_20260422_200000"
    out = dump_derived_data(
        query="median income in DeKalb",
        plan=plan, aggregated=aggregated, fetch_results=fetch_results,
        metadata_db=md, folder=folder,
    )
    assert out == folder
    assert (folder / "relevant_variables_county.csv").exists()
    assert (folder / "variables.txt").exists()
    assert (folder / "query.txt").read_text().startswith("median income")
    with (folder / "relevant_variables_county.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["geo_id"] == "13089"
    assert r["display_name"] == "DeKalb County"
    assert r["geo_level"] == "county"
    assert r["parent_display_name"] == ""
    assert r["variable_id"] == "B19013_001E"
    assert r["variable_slot"] == "value"
    assert r["value"] == "75000"


# ---------------------------------------------------------------------------
# us → national filename remap
# ---------------------------------------------------------------------------

def test_us_level_writes_national_csv(tmp_path: Path):
    us = _geo("1", "United States", level="us")
    concept = ExtractedConcept(text="median household income",
                                canonical_hint="median household income")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(value="B19013_001E")
    api = _api("B19013", "acs/acs5", "us", ["B19013_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[us], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="comparator.us", variables=cv,
        )],
    )
    fetch_results = [_fetch(api, [{"B19013_001E": "80734", "us": "1"}])]

    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="national median", plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(geo=us, concept=concept, year=2024,
                            role="comparator.us", value=80734.0,
                            rows_aggregated=1),
        ]),
        fetch_results=fetch_results, metadata_db=None, folder=folder,
    )
    assert (folder / "relevant_variables_national.csv").exists()
    assert not (folder / "relevant_variables_us.csv").exists()


# ---------------------------------------------------------------------------
# Tract-level rows carry parent_display_name linking back to the composite
# ---------------------------------------------------------------------------

def test_tract_rows_link_to_composite_neighborhood(tmp_path: Path):
    tracts = [
        "13121000100", "13121000200", "13121000300",
    ]
    buckhead = _geo(
        "BUCKHEAD", "Buckhead", level="neighborhood", tracts=tracts,
    )
    concept = ExtractedConcept(text="median household income",
                                canonical_hint="median household income")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(value="B19013_001E")
    api = _api("B19013", "acs/acs5", "tract", ["B19013_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[buckhead], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="primary", variables=cv,
            tract_filter=tracts,
        )],
    )
    # Include an extra tract (should be filtered out by tract_filter).
    fetch_rows = [
        {"B19013_001E": "125000", "state": "13", "county": "121",
         "tract": "000100"},
        {"B19013_001E": "140000", "state": "13", "county": "121",
         "tract": "000200"},
        {"B19013_001E": "155000", "state": "13", "county": "121",
         "tract": "000300"},
        {"B19013_001E": "65000", "state": "13", "county": "121",
         "tract": "999999"},      # not in filter — must be dropped
    ]
    fetch_results = [_fetch(api, fetch_rows)]

    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="median income in Buckhead", plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(geo=buckhead, concept=concept, year=2024,
                            role="primary", value=140000.0,
                            rows_aggregated=3),
        ]),
        fetch_results=fetch_results, metadata_db=None, folder=folder,
    )
    path = folder / "relevant_variables_tract.csv"
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    # Only the 3 in the tract_filter should appear.
    assert len(rows) == 3
    for r in rows:
        assert r["parent_display_name"] == "Buckhead"
        assert r["parent_geo_level"] == "neighborhood"
        assert r["geo_level"] == "tract"
        assert r["variable_id"] == "B19013_001E"
    tract_ids = {r["geo_id"] for r in rows}
    assert tract_ids == set(tracts)


# ---------------------------------------------------------------------------
# Admin-level rows leave parent columns blank
# ---------------------------------------------------------------------------

def test_admin_level_rows_have_empty_parent_columns(tmp_path: Path):
    fulton = _geo("13121", "Fulton County", level="county")
    concept = ExtractedConcept(text="poverty rate",
                                canonical_hint="poverty rate")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(numerator=["B17001_002E"],
                          denominator="B17001_001E")
    api = _api("B17001", "acs/acs5", "county",
               ["B17001_002E", "B17001_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[fulton], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="primary", variables=cv,
        )],
    )
    fetch_results = [_fetch(api, [{
        "B17001_002E": "86800", "B17001_001E": "700000",
        "state": "13", "county": "121",
    }])]
    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="poverty rate in Fulton", plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(
                geo=fulton, concept=concept, year=2024, role="primary",
                ratio=0.124, sample_size=700000.0, rows_aggregated=1,
            ),
        ]),
        fetch_results=fetch_results, metadata_db=None, folder=folder,
    )
    with (folder / "relevant_variables_county.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    slots = {r["variable_slot"] for r in rows}
    assert slots == {"numerator", "denominator"}
    for r in rows:
        assert r["parent_display_name"] == ""
        assert r["parent_geo_level"] == ""
        assert r["geo_id"] == "13121"


# ---------------------------------------------------------------------------
# variables.txt sidecar content
# ---------------------------------------------------------------------------

def test_variables_txt_lists_every_used_variable(tmp_path: Path):
    fulton = _geo("13121", "Fulton County", level="county")
    concept = ExtractedConcept(text="poverty rate",
                                canonical_hint="poverty rate")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(numerator=["B17001_002E"],
                          denominator="B17001_001E")
    api = _api("B17001", "acs/acs5", "county",
               ["B17001_002E", "B17001_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[fulton], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="primary", variables=cv,
        )],
    )
    fetch_results = [_fetch(api, [{
        "B17001_002E": "86800", "B17001_001E": "700000",
        "state": "13", "county": "121",
    }])]
    md = _metadata_db_with_labels([
        ("B17001_001E", "acs/acs5", 2024, "B17001",
         "Population for whom poverty status is determined"),
        ("B17001_002E", "acs/acs5", 2024, "B17001",
         "Population whose income is below the poverty level"),
    ])
    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="q", plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(geo=fulton, concept=concept, year=2024,
                            role="primary", ratio=0.124,
                            sample_size=700000.0, rows_aggregated=1),
        ]),
        fetch_results=fetch_results, metadata_db=md, folder=folder,
    )
    txt = (folder / "variables.txt").read_text()
    assert "B17001_001E\tacs/acs5\tPopulation for whom poverty status is determined" in txt
    assert "B17001_002E\tacs/acs5\tPopulation whose income is below the poverty level" in txt


def test_variables_txt_still_emits_unknown_var_with_empty_label(tmp_path: Path):
    dekalb = _geo("13089", "DeKalb County", level="county")
    concept = ExtractedConcept(text="obscure metric",
                                canonical_hint="obscure metric")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(value="B99999_001E")
    api = _api("B99999", "acs/acs5", "county", ["B99999_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[dekalb], concept_resolutions=[],
        calls=[PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0,
            year=2024, role="primary", variables=cv,
        )],
    )
    fetch_results = [_fetch(api, [
        {"B99999_001E": "42", "state": "13", "county": "089"},
    ])]
    md = _metadata_db_with_labels([])      # no labels
    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="q", plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(geo=dekalb, concept=concept, year=2024,
                            role="primary", value=42.0, rows_aggregated=1),
        ]),
        fetch_results=fetch_results, metadata_db=md, folder=folder,
    )
    txt = (folder / "variables.txt").read_text()
    assert "B99999_001E\tacs/acs5\t" in txt


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------

def test_dump_returns_none_when_no_fetch_results(tmp_path: Path):
    intent = ExtractedIntent(geo_refs=[], concepts=[])
    plan = PlanResult(
        intent=intent, resolved_geos=[], concept_resolutions=[], calls=[],
    )
    folder = tmp_path / "data_20260422_200000"
    out = dump_derived_data(
        query="q", plan=plan,
        aggregated=AggregatedResult(values=[]),
        fetch_results=None, metadata_db=None, folder=folder,
    )
    assert out is None
    assert not folder.exists()


def test_dump_skips_failed_fetches(tmp_path: Path):
    dekalb = _geo("13089", "DeKalb County", level="county")
    concept = ExtractedConcept(text="x")
    intent = ExtractedIntent(geo_refs=[], concepts=[concept])
    cv = ConceptVariables(value="B19013_001E")
    api_ok = _api("B19013", "acs/acs5", "county", ["B19013_001E"])
    api_bad = _api("B99999", "acs/acs5", "county", ["B99999_001E"])
    plan = PlanResult(
        intent=intent, resolved_geos=[dekalb], concept_resolutions=[],
        calls=[
            PlannedCall(api_call=api_ok, geo_idx=0, concept_idx=0,
                        year=2024, role="primary", variables=cv),
            PlannedCall(api_call=api_bad, geo_idx=0, concept_idx=0,
                        year=2024, role="primary", variables=cv),
        ],
    )
    fetch_results = [
        _fetch(api_ok, [{"B19013_001E": "75000",
                        "state": "13", "county": "089"}]),
        FetchResult(plan=api_bad, rows=[], cache_hit=False,
                    elapsed_s=0.1, error="404"),
    ]
    folder = tmp_path / "data_20260422_200000"
    dump_derived_data(
        query="q", plan=plan,
        aggregated=AggregatedResult(values=[]),
        fetch_results=fetch_results, metadata_db=None, folder=folder,
    )
    with (folder / "relevant_variables_county.csv").open() as f:
        rows = list(csv.DictReader(f))
    var_ids = {r["variable_id"] for r in rows}
    assert var_ids == {"B19013_001E"}
