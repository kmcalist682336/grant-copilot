"""Regressions for policy goals and demographic scope in generated answers."""
from scripts.chatbot.aggregator import aggregate_results
from scripts.chatbot.census_caller import RecordFilter
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import ExtractedConcept
from scripts.chatbot.planner import ConceptResolution, PlannedCall, _calls_for_resolved_concept
from scripts.chatbot.synthesizer import _format_value
from tests.test_aggregator import _api, _geo, _plan, _fetch, _intent


def test_housing_investment_does_not_become_an_arbitrary_numeric_proxy():
    concept = ExtractedConcept(text="affordable housing investment")
    resolution = ConceptResolution(concept=concept, tier="tier_semantic")
    assert _calls_for_resolved_concept(
        _geo(), 0, 0, resolution, _intent([concept]), None,
    ) == []


def test_applied_demographic_filters_reach_writer_with_denominator_scope():
    concept = ExtractedConcept(text="mortgage denial rate")
    api = _api("hmda", "hmda", "record", ["n", "d"])
    api.record_filters = [
        RecordFilter("race", "equals", "Black or African American"),
        RecordFilter("sex", "equals", "Female"),
    ]
    api.record_numerator_filters = [RecordFilter("action", "equals", "Denied")]
    call = PlannedCall(api_call=api, geo_idx=0, concept_idx=0, year=2023,
                       variables=ConceptVariables(numerator="n", denominator="d"))
    plan = _plan([call], [_geo()], [concept])
    result = aggregate_results(plan, [_fetch([{"n": 2, "d": 10}], api)])
    payload = _format_value(result.values[0])
    assert payload["ratio"] == 0.2
    source = payload["source_context"][0]
    assert source["dataset"] == "hmda"
    assert [f["value"] for f in source["record_filters"]] == [
        "Black or African American", "Female"]
    assert source["numerator_filters"][0]["value"] == "Denied"
