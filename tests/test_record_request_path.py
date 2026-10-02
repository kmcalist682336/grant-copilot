import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.chatbot.models import ExtractedAnalysis, ExtractedConcept, ExtractedFilter, ExtractedIntent
from scripts.chatbot.nodes.record_metric_interpreter import interpret_record_metrics, RecordMetricInterpreterError
from scripts.chatbot.orchestrator import _focused_record_request, _is_simple_curated_record_lookup, _run_fetches
from scripts.chatbot.record_planner import _variable_id
from scripts.chatbot.api_cache import APICache
from tests.test_record_dataset_adapters import pantry, _intent
from tests.test_aggregator import _api, _fetch, _plan, _geo
from scripts.chatbot.planner import PlannedCall
from scripts.chatbot.concept_map import ConceptVariables


def test_direct_trend_does_not_opt_into_grant_context():
    intent = _intent("hmda", "mortgage denial rate")
    assert _focused_record_request("denial rate trends for Black women in Buckhead", intent)
    assert not _focused_record_request("Evidence for a mortgage access grant in Buckhead", intent)
    assert not _focused_record_request("Build a case for improving mortgage access", intent)


def test_partial_filters_do_not_bypass_completeness_check():
    intent = _intent("hmda", "mortgage denial rate", filters=[ExtractedFilter(
        dimension=ExtractedConcept(text="applicant race", dataset_hint="hmda"), value_text="Black")])
    assert not _is_simple_curated_record_lookup(intent, None, query="denial rate for Black women")


def test_alias_shortcut_does_not_embed(pantry):
    class Router:
        def route_dataset(self, *args, **kwargs):
            raise AssertionError("Known aliases must not trigger a semantic search")
    variable, route = _variable_id(ExtractedConcept(text="food weight"), Router(),
                                  dataset="pantry", table_id="visits")
    assert (variable, route) == ("kg", None)


def test_interpreter_uses_selected_dataset_and_preserves_scope(pantry):
    original = _intent("pantry", "fulfillment rate")
    original.years = [2022]
    corrected = _intent("pantry", "fulfillment rate", filters=[ExtractedFilter(
        dimension=ExtractedConcept(text="household type", dataset_hint="pantry"), value_text="families")])
    calls = []
    class LLM:
        def extract(self, **kwargs):
            calls.append(kwargs)
            return corrected.model_dump()
    result, notes = interpret_record_metrics("fulfillment rate for families", original, LLM(),
        dataset="pantry", frame=SimpleNamespace(required_record_analyses=[1]))
    assert len(calls) == 1  # A frame must not skip filter interpretation.
    payload = json.loads(calls[0]["user_text"])
    definition = payload["dataset_definition"]
    assert definition["dataset"] == "pantry"
    assert definition["variable_aliases"]["household type"] == "segment"
    assert "hmda" not in json.dumps(payload)
    assert result.years == [2022]
    assert result.analyses[0].filters[0].value_text == "families"


def test_interpreter_rejects_foreign_dataset(pantry):
    class LLM:
        def extract(self, **kwargs):
            return _intent("hmda", "mortgage denial rate").model_dump()
    with pytest.raises(RecordMetricInterpreterError, match="different dataset"):
        interpret_record_metrics("fulfillment rate", _intent("pantry", "fulfillment rate"),
                                 LLM(), dataset="pantry")


def test_separate_connector_timings_preserve_result_order(monkeypatch, tmp_path):
    from scripts.chatbot import orchestrator
    class Caller:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def fetch_all(self, plans):
            return [_fetch([{"x": 1}], p) for p in plans]
    monkeypatch.setattr(orchestrator, "CensusCaller", Caller)
    calls = [PlannedCall(api_call=_api("t", ds, "record", ["x"]),
                        geo_idx=0, concept_idx=0, year=2023,
                        variables=ConceptVariables(value="x")) for ds in ("hmda", "acs/acs5")]
    plan = _plan(calls, [_geo()], [ExtractedConcept(text="metric")])
    timings = {}
    results = asyncio.run(_run_fetches(plan, APICache(tmp_path / "cache.db"), None,
                                      record_caller=Caller(), timings=timings))
    assert [r.plan.dataset for r in results] == ["hmda", "acs/acs5"]
    assert set(timings) == {"census_fetch_s", "record_fetch_s"}
    assert all(v >= 0 for v in timings.values())
