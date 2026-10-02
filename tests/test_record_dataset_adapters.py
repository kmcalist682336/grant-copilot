"""Execute two unrelated datasets through the same planner and DuckDB path."""
import asyncio
from dataclasses import replace

import duckdb
import pytest

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.aggregator import aggregate_results
from scripts.chatbot.duckdb_caller import DuckDBCaller
from scripts.chatbot.models import ExtractedAnalysis, ExtractedConcept, ExtractedFilter, ExtractedIntent
from scripts.chatbot.record_dataset import RecordDatasetAdapter, get_record_dataset, register_record_dataset
from scripts.chatbot.record_metric_map import RecordMetricMap, RecordMetricRecipe
from scripts.chatbot.record_planner import plan_record_query
from tests.test_aggregator import _geo


@pytest.fixture
def pantry(monkeypatch):
    from scripts.chatbot import record_dataset
    get_record_dataset("hmda")
    monkeypatch.setattr(record_dataset, "_ADAPTERS", dict(record_dataset._ADAPTERS))
    from pathlib import Path
    from scripts.chatbot.record_dataset import load_record_datasets
    return load_record_datasets(Path(__file__).parent / "fixtures/record_datasets.yaml")[0]


def _intent(dataset, measure, operation="percentage", **kwargs):
    return ExtractedIntent(analyses=[ExtractedAnalysis(
        measure=ExtractedConcept(text=measure, dataset_hint=dataset),
        operation=operation, **kwargs)], temporal_intent="latest")


def _execute(tmp_path, adapter, plan, rows):
    """Write tiny local variable-tree Parquet files; no network or mocks."""
    con = duckdb.connect()
    con.execute(f'CREATE TABLE records (ticket VARCHAR, "{adapter.geography_column}" VARCHAR, value VARCHAR)')
    for year in {c.year for c in plan.calls}:
        for variable in rows[0][2]:
            folder = tmp_path / f"table_id={adapter.table_id}" / f"year={year}" / f"variable={variable}"
            folder.mkdir(parents=True)
            con.execute("DELETE FROM records")
            con.executemany("INSERT INTO records VALUES (?, ?, ?)",
                            [(key, geo, str(values[variable])) for key, geo, values in rows])
            con.execute("COPY records TO ? (FORMAT PARQUET)", [str(folder / "part.parquet")])
    cache = APICache(tmp_path / "cache.db")
    caller = DuckDBCaller(con, cache, bucket_uri=str(tmp_path), geo_id_column="ticket", layout="variable_tree")
    results = asyncio.run(caller.fetch_all([c.api_call for c in plan.calls]))
    assert all(r.succeeded for r in results), [r.error for r in results]
    aggregated = aggregate_results(plan, results)
    con.close()
    return aggregated


def _pantry_rows():
    return [
        ("1", "A1", {"decision": "Y", "segment": "F", "kg": 10}),
        ("2", "A1", {"decision": "N", "segment": "F", "kg": 20}),
        ("3", "A1", {"decision": "P", "segment": "F", "kg": 30}),
        ("4", "A10", {"decision": "Y", "segment": "F", "kg": 100}),
        ("5", "A1", {"decision": "Y", "segment": "S", "kg": 5}),
    ]


def test_second_dataset_rate_filters_years_and_exact_geography(pantry, tmp_path):
    intent = _intent("pantry", "fulfillment rate", filters=[ExtractedFilter(
        dimension=ExtractedConcept(text="household type", dataset_hint="pantry"), value_text="families")])
    geo = _geo().model_copy(update={"geo_id": "A1"})
    plan = plan_record_query(intent, [geo], dataset="pantry", semantic_router=None)
    assert [c.year for c in plan.calls] == [2022, 2025]
    assert all(c.api_call.record_geography_column == "service_area" for c in plan.calls)
    result = _execute(tmp_path, pantry, plan, _pantry_rows())
    assert [v.ratio for v in result.values] == [0.5, 0.5]
    assert [v.sample_size for v in result.values] == [2, 2]


@pytest.mark.parametrize("operation,expected", [("sum", 65), ("average", 16.25), ("median", 15), ("count", 4)])
def test_second_dataset_numeric_operations(pantry, tmp_path, operation, expected):
    plan = plan_record_query(_intent("pantry", "food weight", operation),
        [_geo().model_copy(update={"geo_id": "A1"})], dataset="pantry",
        semantic_router=None, trend_lookback_years=0)
    result = _execute(tmp_path, pantry, plan, _pantry_rows())
    assert result.values[0].value == expected


def test_selection_and_missing_coverage_fail_closed(pantry):
    intent = _intent("pantry", "food weight", "count")
    with pytest.raises(TypeError, match="dataset"):
        plan_record_query(intent, [], semantic_router=None)
    with pytest.raises(ValueError, match="No record adapter"):
        plan_record_query(intent, [], dataset="unregistered", semantic_router=None)
    assert plan_record_query(intent, [], dataset="pantry", semantic_router=None).calls == []
    intent.years = [1990]
    assert plan_record_query(intent, [_geo()], dataset="pantry", semantic_router=None).calls == []
    with pytest.raises(ValueError, match="does not support geography"):
        plan_record_query(_intent("pantry", "food weight", "count"), [_geo(level="state")],
                          dataset="pantry", semantic_router=None)


def test_invalid_category_and_cross_dataset_recipe(pantry):
    with pytest.raises(ValueError, match="Unsupported"):
        pantry.filter_value("segment", "businesses")
    from scripts.chatbot.record_planner import record_metric_recipe_for_analysis
    hmda_recipes = get_record_dataset("hmda").recipes()
    analysis = _intent("pantry", "mortgage denial rate").analyses[0]
    assert record_metric_recipe_for_analysis(analysis, dataset="pantry", record_metric_map=hmda_recipes) is None


@pytest.mark.parametrize("metric,expected,denominator", [
    # Preserve the shipped definitions: denial/approval use all matching
    # records, while origination explicitly uses completed applications.
    ("mortgage denial rate", 1/6, 6),
    ("mortgage approval rate", 2/6, 6),
    ("mortgage origination rate among completed applications", 1/3, 3),
    ("mortgage withdrawal rate", 1/5, 5),
    ("incomplete mortgage application rate", 1/5, 5),
])
def test_hmda_calculations_execute_with_original_universes(tmp_path, metric, expected, denominator):
    adapter = get_record_dataset("hmda")
    plan = plan_record_query(_intent("hmda", metric), [], dataset="hmda",
                             semantic_router=None, trend_lookback_years=0)
    statuses = ["Loan originated", "Application approved but not accepted", "Application denied",
                "Application withdrawn by applicant", "File closed for incompleteness", "Purchased loan"]
    rows = [(str(i), "13121000100", {"906bb78b0f70": s}) for i, s in enumerate(statuses)]
    rows.append(("outside", "06121000100", {"906bb78b0f70": "Application denied"}))
    result = _execute(tmp_path, adapter, plan, rows)
    assert result.values[0].ratio == pytest.approx(expected)
    assert result.values[0].sample_size == denominator


def test_geography_semantics_are_part_of_cache_key():
    from scripts.chatbot.census_caller import APIPlanCall
    call = APIPlanCall(url="record://x", table_id="visits", variables=["decision"],
        geo_level="record", geo_filter_ids=[], year=2025, dataset="pantry", ttl_seconds=60,
        geo_prefixes=["A1"], record_geography_column="service_area", record_geography_match="exact")
    assert call.cache_key != replace(call, record_geography_match="prefix").cache_key
    assert call.cache_key != replace(call, record_geography_column="another_area").cache_key


def test_second_dataset_grouping_uses_its_own_categories(pantry, tmp_path):
    intent = _intent("pantry", "food weight", "sum",
        groupings=[ExtractedConcept(text="household type", dataset_hint="pantry")],
        grouping_values={"household type": ["families", "individuals"]})
    plan = plan_record_query(intent, [_geo().model_copy(update={"geo_id": "A1"})],
                             dataset="pantry", semantic_router=None, trend_lookback_years=0)
    result = _execute(tmp_path, pantry, plan, _pantry_rows())
    assert {v.role: v.value for v in result.values} == {
        "group_household type=F": 60, "group_household type=S": 5}


def test_unknown_year_coverage_does_not_fabricate_years(pantry):
    pantry.supported_years = ()
    plan = plan_record_query(_intent("pantry", "food weight", "sum"), [_geo()],
                             dataset="pantry", semantic_router=None)
    assert plan.calls == []
    assert any("No supported years" in note for note in plan.notes)
