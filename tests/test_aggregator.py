"""Phase H aggregator tests — uses synthetic FetchResults so no network."""
from __future__ import annotations

import pytest

from scripts.chatbot.aggregator import (
    AggregatedValue, _filter_rows_to_tracts, _to_float,
    aggregate_results,
)
from scripts.chatbot.census_caller import APIPlanCall, FetchResult
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.planner import PlanResult, PlannedCall


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _api(table_id: str, dataset: str, geo_level: str,
         vars_: list[str], year: int = 2023) -> APIPlanCall:
    return APIPlanCall(
        url=f"https://example.com/{table_id}", table_id=table_id,
        variables=vars_, geo_level=geo_level,
        geo_filter_ids=[], year=year, dataset=dataset,
        ttl_seconds=3600,
    )


def _geo(name: str = "Test", level: str = "county",
         tract_geoids: list[str] = None) -> ResolvedGeography:
    from scripts.chatbot.models import ExtractedGeoRef
    ref_type = "neighborhood" if level == "neighborhood" else "administrative"
    # data_level_available enum doesn't include 'neighborhood'/'corridor'/etc.;
    # those geos resolve to underlying tract data.
    data_level = "tract" if level in (
        "neighborhood", "corridor", "region", "landmark_area",
    ) else level
    return ResolvedGeography(
        geo_id="TEST", geo_level=level, geo_type=level,
        display_name=name,
        tract_geoids=tract_geoids or [],
        county_geoid=None,
        api_for_clause=f"{level}:*",
        api_in_clause="",
        confidence=0.9,
        assumption_notes=[],
        data_level_available=data_level,
        source_ref=ExtractedGeoRef(text=name, ref_type=ref_type),
    )


def _intent(concepts: list[ExtractedConcept]) -> ExtractedIntent:
    return ExtractedIntent(geo_refs=[], concepts=concepts)


def _plan(calls: list[PlannedCall], geos: list[ResolvedGeography],
          concepts: list[ExtractedConcept]) -> PlanResult:
    return PlanResult(
        intent=_intent(concepts),
        resolved_geos=geos,
        concept_resolutions=[],   # aggregator doesn't read this
        calls=calls,
    )


def _fetch(rows: list[dict], call: APIPlanCall) -> FetchResult:
    return FetchResult(
        plan=call, rows=rows, cache_hit=False, elapsed_s=0.0,
    )


def _failed(call: APIPlanCall, error: str) -> FetchResult:
    return FetchResult(
        plan=call, rows=[], cache_hit=False, elapsed_s=0.0, error=error,
    )


# ---------------------------------------------------------------------------
# _to_float
# ---------------------------------------------------------------------------

def test_to_float_handles_none_and_sentinels():
    assert _to_float(None) is None
    assert _to_float("") is None
    assert _to_float("(X)") is None
    assert _to_float("**") is None
    assert _to_float("-666666666") is None


def test_to_float_normal_values():
    assert _to_float("42") == 42.0
    assert _to_float(" 12345.6 ") == 12345.6
    assert _to_float(0) == 0.0


# ---------------------------------------------------------------------------
# _filter_rows_to_tracts
# ---------------------------------------------------------------------------

def test_filter_rows_to_tracts_picks_only_wanted_geoids():
    rows = [
        {"state": "13", "county": "121", "tract": "009001", "VAL": "10"},
        {"state": "13", "county": "121", "tract": "009002", "VAL": "20"},
        {"state": "13", "county": "121", "tract": "009003", "VAL": "30"},
    ]
    out = _filter_rows_to_tracts(rows, ["13121009001", "13121009003"])
    assert len(out) == 2
    assert {r["VAL"] for r in out} == {"10", "30"}


def test_filter_rows_to_tracts_empty_filter_returns_all():
    rows = [{"state": "13", "county": "121", "tract": "009001"}]
    assert _filter_rows_to_tracts(rows, []) == rows


# ---------------------------------------------------------------------------
# value role — admin (single row)
# ---------------------------------------------------------------------------

def test_value_role_admin_single_row():
    geo = _geo("DeKalb County", "county")
    concept = ExtractedConcept(text="median income")
    api = _api("B19013", "acs/acs5", "county", ["B19013_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(value="B19013_001E"),
    )
    fetch = _fetch(
        [{"NAME": "DeKalb County, GA", "B19013_001E": "65432",
          "state": "13", "county": "089"}],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    assert len(out.values) == 1
    v = out.values[0]
    assert v.value == 65432.0
    assert v.ratio is None and v.components is None
    assert v.aggregation_caveat is None  # single row, no caveat
    assert v.rows_aggregated == 1


# ---------------------------------------------------------------------------
# value role — tract aggregation triggers caveat
# ---------------------------------------------------------------------------

def test_value_role_tract_aggregation_uses_mean_with_caveat():
    geo = _geo(
        "Buckhead", "neighborhood",
        tract_geoids=["13121009001", "13121009002"],
    )
    concept = ExtractedConcept(text="median income")
    api = _api("B19013", "acs/acs5", "tract", ["B19013_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(value="B19013_001E"),
        tract_filter=["13121009001", "13121009002"],
    )
    fetch = _fetch(
        [
            {"NAME": "T1", "B19013_001E": "100000",
             "state": "13", "county": "121", "tract": "009001"},
            {"NAME": "T2", "B19013_001E": "60000",
             "state": "13", "county": "121", "tract": "009002"},
        ],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    v = out.values[0]
    assert v.value == 80000.0   # mean of 100000 and 60000
    assert v.aggregation_caveat is not None
    assert "tracts" in v.aggregation_caveat
    assert v.rows_aggregated == 2


# ---------------------------------------------------------------------------
# numerator+denominator role — sum then divide
# ---------------------------------------------------------------------------

def test_ratio_role_sums_numerator_and_denominator():
    geo = _geo(
        "Buckhead", "neighborhood",
        tract_geoids=["13121009001", "13121009002"],
    )
    concept = ExtractedConcept(text="poverty rate")
    api = _api("B17001", "acs/acs5", "tract",
               ["B17001_002E", "B17001_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            numerator="B17001_002E", denominator="B17001_001E",
        ),
        tract_filter=["13121009001", "13121009002"],
    )
    # Tract 1: 100 of 1000 in poverty (10%); Tract 2: 200 of 1000 (20%)
    # Combined: 300 / 2000 = 15%
    fetch = _fetch(
        [
            {"B17001_002E": "100", "B17001_001E": "1000",
             "state": "13", "county": "121", "tract": "009001"},
            {"B17001_002E": "200", "B17001_001E": "1000",
             "state": "13", "county": "121", "tract": "009002"},
        ],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    v = out.values[0]
    assert v.value is None
    assert v.ratio == pytest.approx(0.15)
    assert v.sample_size == 2000.0
    assert v.rows_aggregated == 2


def test_ratio_role_handles_multi_var_numerator():
    """Bachelor's-or-higher: sum 4 numerator cells, divide by universe."""
    geo = _geo("Test", "county")
    concept = ExtractedConcept(text="college rate")
    api = _api("B15003", "acs/acs5", "county",
               ["B15003_022E", "B15003_023E", "B15003_024E",
                "B15003_025E", "B15003_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            numerator=["B15003_022E", "B15003_023E",
                       "B15003_024E", "B15003_025E"],
            denominator="B15003_001E",
        ),
    )
    fetch = _fetch(
        [{"B15003_022E": "100", "B15003_023E": "50",
          "B15003_024E": "10", "B15003_025E": "5",
          "B15003_001E": "1000"}],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    v = out.values[0]
    assert v.ratio == pytest.approx(0.165)


def test_ratio_role_zero_denominator_skipped():
    geo = _geo("Test", "tract", tract_geoids=["13121009001"])
    concept = ExtractedConcept(text="x")
    api = _api("B17001", "acs/acs5", "tract",
               ["B17001_002E", "B17001_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            numerator="B17001_002E", denominator="B17001_001E",
        ),
        tract_filter=["13121009001"],
    )
    fetch = _fetch(
        [{"B17001_002E": "0", "B17001_001E": "0",
          "state": "13", "county": "121", "tract": "009001"}],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    v = out.values[0]
    assert v.ratio is None
    assert v.sample_size is None


# ---------------------------------------------------------------------------
# components role
# ---------------------------------------------------------------------------

def test_components_role_sums_each_independently():
    geo = _geo("Test", "tract", tract_geoids=["13121009001",
                                              "13121009002"])
    concept = ExtractedConcept(text="race")
    api = _api("B02001", "acs/acs5", "tract",
               ["B02001_002E", "B02001_003E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            components=["B02001_002E", "B02001_003E"],
        ),
        tract_filter=["13121009001", "13121009002"],
    )
    fetch = _fetch(
        [
            {"B02001_002E": "500", "B02001_003E": "300",
             "state": "13", "county": "121", "tract": "009001"},
            {"B02001_002E": "200", "B02001_003E": "700",
             "state": "13", "county": "121", "tract": "009002"},
        ],
        api,
    )
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    v = out.values[0]
    assert v.components == {"B02001_002E": 700.0, "B02001_003E": 1000.0}
    assert v.rows_aggregated == 2


# ---------------------------------------------------------------------------
# multi-county tract aggregation — calls merge into one bucket
# ---------------------------------------------------------------------------

def test_multi_county_tract_calls_merge_into_one_aggregated_value():
    """Two PlannedCalls for the same (geo, concept, year, role) — one per
    county — should produce ONE AggregatedValue."""
    geo = _geo(
        "Cross-County Region", "neighborhood",
        tract_geoids=["13121009001", "13089020100"],
    )
    concept = ExtractedConcept(text="poverty rate")
    api1 = _api("B17001", "acs/acs5", "tract",
                ["B17001_002E", "B17001_001E"])
    api2 = _api("B17001", "acs/acs5", "tract",
                ["B17001_002E", "B17001_001E"])
    call1 = PlannedCall(
        api_call=api1, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            numerator="B17001_002E", denominator="B17001_001E",
        ),
        tract_filter=["13121009001"],
    )
    call2 = PlannedCall(
        api_call=api2, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(
            numerator="B17001_002E", denominator="B17001_001E",
        ),
        tract_filter=["13089020100"],
    )
    fetch1 = _fetch(
        [{"B17001_002E": "150", "B17001_001E": "1500",
          "state": "13", "county": "121", "tract": "009001"}],
        api1,
    )
    fetch2 = _fetch(
        [{"B17001_002E": "350", "B17001_001E": "2500",
          "state": "13", "county": "089", "tract": "020100"}],
        api2,
    )
    out = aggregate_results(
        _plan([call1, call2], [geo], [concept]),
        [fetch1, fetch2],
    )
    assert len(out.values) == 1
    v = out.values[0]
    # Combined: (150+350) / (1500+2500) = 500 / 4000 = 0.125
    assert v.ratio == pytest.approx(0.125)
    assert v.sample_size == 4000.0


# ---------------------------------------------------------------------------
# multi-year (change) — produces one value per year
# ---------------------------------------------------------------------------

def test_multi_year_produces_one_value_per_year():
    geo = _geo("Test", "county")
    concept = ExtractedConcept(text="median income")
    calls = []
    fetches = []
    for year, val in ((2020, 50000), (2023, 65000)):
        api = _api("B19013", "acs/acs5", "county",
                   ["B19013_001E"], year=year)
        calls.append(PlannedCall(
            api_call=api, geo_idx=0, concept_idx=0, year=year,
            variables=ConceptVariables(value="B19013_001E"),
        ))
        fetches.append(_fetch(
            [{"B19013_001E": str(val), "state": "13", "county": "089"}],
            api,
        ))
    out = aggregate_results(_plan(calls, [geo], [concept]), fetches)
    by_year = {v.year: v for v in out.values}
    assert by_year[2020].value == 50000.0
    assert by_year[2023].value == 65000.0


# ---------------------------------------------------------------------------
# fetch failures surface in fetch_failures
# ---------------------------------------------------------------------------

def test_fetch_failures_recorded_separately():
    geo = _geo("Test", "county")
    concept = ExtractedConcept(text="x")
    api = _api("B19013", "acs/acs5", "county", ["B19013_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(value="B19013_001E"),
    )
    fetch = _failed(api, "http_500")
    out = aggregate_results(_plan([call], [geo], [concept]), [fetch])
    assert len(out.fetch_failures) == 1
    assert "http_500" in out.fetch_failures[0]
    # Aggregated value still exists, but value is None
    assert len(out.values) == 1
    assert out.values[0].value is None


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_aggregate_results_mismatched_lengths_raises():
    geo = _geo("Test", "county")
    concept = ExtractedConcept(text="x")
    api = _api("B19013", "acs/acs5", "county", ["B19013_001E"])
    call = PlannedCall(
        api_call=api, geo_idx=0, concept_idx=0, year=2023,
        variables=ConceptVariables(value="B19013_001E"),
    )
    with pytest.raises(ValueError, match="fetch results"):
        aggregate_results(_plan([call], [geo], [concept]), [])
