from types import SimpleNamespace
from dataclasses import replace

import pytest

from scripts.chatbot.dataset_selection import select_dataset, DatasetSelectionError
from scripts.chatbot.models import ExtractedConcept, ExtractedIntent
from scripts.chatbot.orchestrator import _promote_record_concepts_to_analyses
from scripts.chatbot.record_dataset import register_record_dataset
from tests.test_record_dataset_adapters import pantry, _intent


def router(*scores, tables=()):
    def target(ds, score): return SimpleNamespace(target_dataset=ds, aggregate_score=score)
    return SimpleNamespace(route=lambda *a, **kw: SimpleNamespace(
        top_variables=[target(*p) for p in scores], top_tables=[target(*p) for p in tables]))


def test_semantic_winner_can_be_new_dataset(pantry):
    result = select_dataset(ExtractedConcept(text="food distributed", dataset_hint="hmda"),
                            router(("pantry", .8), ("hmda", .3)))
    assert result.dataset == "pantry"  # A hint cannot force HMDA.


def test_census_winner_is_not_hijacked_by_record_runner_up(pantry):
    assert select_dataset(ExtractedConcept(text="household statistic"),
        router(("acs/acs5", .8), ("hmda", .4))).dataset == "census"


@pytest.mark.parametrize("dataset", ["acs/acs5", "acs/acs1/profile", "dec/pl", "dec", "pep/population", "popest"])
def test_census_families_keep_census_path(dataset):
    assert select_dataset(ExtractedConcept(text="population statistic"),
                          router((dataset, .8), ("hmda", .2))).dataset == "census"


def census_map():
    from pathlib import Path
    from scripts.chatbot.concept_map import ConceptMap
    return ConceptMap.from_yaml(Path(__file__).parents[1] / "config/concept_map.yaml")


def test_homeownership_curated_mapping_bypasses_semantic_selection():
    def fail(*args, **kwargs):
        pytest.fail("A curated Census concept must not need semantic source selection")
    intent = ExtractedIntent(concepts=[ExtractedConcept(text="homeownership rate")])
    selected = _promote_record_concepts_to_analyses(
        "What's the homeownership rate in Fulton County?", intent,
        SimpleNamespace(route=fail), cmap=census_map())
    assert selected.analyses == []
    assert selected.concepts[0].dataset_hint == "census"


def test_mixed_query_preserves_census_and_record_paths(pantry):
    intent = ExtractedIntent(concepts=[ExtractedConcept(text="homeownership rate"),
                                     ExtractedConcept(text="fulfillment rate")])
    selected = _promote_record_concepts_to_analyses(
        "homeownership rate and fulfillment rate", intent, None, cmap=census_map())
    assert [c.text for c in selected.concepts] == ["homeownership rate"]
    assert [a.measure.dataset_hint for a in selected.analyses] == ["pantry"]


def test_explicit_record_source_cannot_reinterpret_census_metric():
    with pytest.raises(DatasetSelectionError, match="resolves to Census"):
        select_dataset(ExtractedConcept(text="homeownership rate"), None,
                       cmap=census_map(), requested_dataset="hmda")


def test_homeownership_builds_census_calls(metadata_db):
    from scripts.chatbot.planner import plan_query
    from tests.test_aggregator import _geo
    cmap = census_map()
    selected = _promote_record_concepts_to_analyses(
        "homeownership rate", ExtractedIntent(concepts=[
            ExtractedConcept(text="homeownership rate")]),
        router(("hmda", .99), ("acs/acs5", .1)), cmap=cmap)
    plan = plan_query(selected, [_geo()], cmap, metadata_db)
    assert plan.calls
    assert all(p.api_call.table_id == "B25003" for p in plan.calls)
    assert all(p.api_call.dataset.startswith("acs/") for p in plan.calls)
    assert all(p.variables.numerator == ["B25003_002E"] and
               p.variables.denominator == "B25003_001E" for p in plan.calls)


def test_direct_homeownership_does_not_inject_hmda_frame(monkeypatch, tmp_path, metadata_db):
    import asyncio
    from scripts.chatbot import orchestrator
    from scripts.chatbot.api_cache import APICache
    from scripts.chatbot.decomposition_cache import DecompositionCache
    from scripts.chatbot.synthesizer import SynthesizedAnswer
    from tests.test_aggregator import _geo, _fetch

    intent = ExtractedIntent(concepts=[ExtractedConcept(text="homeownership rate")])
    monkeypatch.setattr(orchestrator, "extract_intent", lambda *a, **kw: intent)
    monkeypatch.setattr(orchestrator, "resolve_intent", lambda *a: [_geo()])
    def forbidden(*args, **kwargs):
        pytest.fail("Direct Census lookup reached narrative frame/record interpreter")
    monkeypatch.setattr(orchestrator, "match_frame", forbidden)
    monkeypatch.setattr(orchestrator, "interpret_record_metrics", forbidden)
    calls = []
    async def fetch_all(self, plans):
        calls.extend(plans)
        return [_fetch([{"B25003_002E": "54", "B25003_001E": "100"}], p) for p in plans]
    monkeypatch.setattr(orchestrator.CensusCaller, "fetch_all", fetch_all)
    monkeypatch.setattr(orchestrator, "synthesize", lambda *a, **kw: SynthesizedAnswer(prose="Homeownership result"))
    response = asyncio.run(orchestrator.answer_query(
        "What's the homeownership rate in Fulton County?", SimpleNamespace(),
        None, metadata_db, census_map(),
        decomp_cache=DecompositionCache(tmp_path / "decomp.db"),
        api_cache=APICache(tmp_path / "api.db"), api_key=None,
        config={"scope_gate": {"enabled": False}, "clarification": {"enabled": False}},
        semantic_router=router(("hmda", .99)), max_comparators=0, trend_lookback_years=0))
    assert response.error is None
    assert calls and all(p.table_id == "B25003" for p in calls)
    assert response.aggregated.values[0].ratio == .54


@pytest.mark.parametrize("scores", [(("hmda", .8), ("pantry", .75)),
                                   (("pantry", .8), ("acs/acs5", .75))])
def test_near_ties_require_clarification(pantry, scores):
    with pytest.raises(DatasetSelectionError, match="Ambiguous"):
        select_dataset(ExtractedConcept(text="ambiguous measure"), router(*scores))


def test_table_and_variable_candidates_are_ranked_together(pantry):
    assert select_dataset(ExtractedConcept(text="food distributed"),
        router(("hmda", .2), tables=(("pantry", .8),))).dataset == "pantry"


def test_unsupported_winner_is_not_replaced(pantry):
    with pytest.raises(DatasetSelectionError, match="not configured"):
        select_dataset(ExtractedConcept(text="metric"), router(("missing", .8), ("hmda", .2)))


@pytest.mark.parametrize("scores", [(), (("pantry", .01),)])
def test_empty_or_weak_matches_stop(pantry, scores):
    with pytest.raises(DatasetSelectionError, match="No confident"):
        select_dataset(ExtractedConcept(text="metric"), router(*scores))


def test_exact_metric_does_not_need_embeddings(pantry):
    assert select_dataset(ExtractedConcept(text="fulfillment rate"), None).dataset == "pantry"


def test_shared_field_alias_does_not_choose_arbitrarily(pantry):
    register_record_dataset(replace(pantry, dataset="other"))
    with pytest.raises(DatasetSelectionError, match="unique"):
        select_dataset(ExtractedConcept(text="household type"), None)


def test_promoted_concept_carries_selected_id_without_mortgage_context(pantry):
    intent = ExtractedIntent(concepts=[ExtractedConcept(text="food distributed")])
    selected = _promote_record_concepts_to_analyses("food distributed", intent,
        router(("pantry", .8), ("hmda", .2)))
    assert selected.concepts == []
    assert selected.analyses[0].measure.dataset_hint == "pantry"
    assert selected.analyses[0].population_context is None


def test_existing_analysis_hint_is_checked(pantry):
    selected = _promote_record_concepts_to_analyses("food distributed",
        _intent("hmda", "food distributed"), router(("pantry", .8)))
    assert selected.analyses[0].measure.dataset_hint == "pantry"


def test_user_named_source_can_resolve_a_tie(pantry):
    intent = ExtractedIntent(concepts=[ExtractedConcept(text="food distributed")])
    selected = _promote_record_concepts_to_analyses("Use pantry for food distributed", intent,
        router(("pantry", .8), ("hmda", .79)))
    assert selected.analyses[0].measure.dataset_hint == "pantry"


def test_user_named_source_still_requires_a_measure_match(pantry):
    with pytest.raises(DatasetSelectionError, match="No confident match"):
        select_dataset(ExtractedConcept(text="unsupported measure"), router(("hmda", .8)),
                       requested_dataset="pantry")


@pytest.mark.parametrize("mapped_connector", [True, False])
def test_selected_source_reaches_interpreter_and_only_its_connector(pantry, monkeypatch, tmp_path, mapped_connector):
    import asyncio
    import json
    from scripts.chatbot import orchestrator
    from scripts.chatbot.api_cache import APICache
    from scripts.chatbot.decomposition_cache import DecompositionCache
    from scripts.chatbot.synthesizer import SynthesizedAnswer
    from tests.test_aggregator import _geo, _fetch

    original = ExtractedIntent(concepts=[ExtractedConcept(text="food aid success")])
    monkeypatch.setattr(orchestrator, "extract_intent", lambda *a, **kw: original)
    monkeypatch.setattr(orchestrator, "resolve_intent", lambda *a: [_geo()])
    monkeypatch.setattr(orchestrator, "synthesize", lambda *a, **kw: SynthesizedAnswer(prose="Synthetic result"))
    calls = []
    interpretations = []
    class LLM:
        def extract(self, **kwargs):
            payload = json.loads(kwargs["user_text"])
            interpretations.append(payload["dataset_definition"]["dataset"])
            return _intent("pantry", "fulfillment rate").model_dump()
    class Caller:
        async def fetch_all(self, plans):
            calls.extend(plans)
            return [_fetch([], p) for p in plans]
    response = asyncio.run(orchestrator.answer_query(
        "food aid success in Test County", LLM(), None, None, None,
        decomp_cache=DecompositionCache(tmp_path / "decomp.db"),
        api_cache=APICache(tmp_path / "api.db"), api_key=None,
        config={"scope_gate": {"enabled": False}, "clarification": {"enabled": False}},
        semantic_router=router(("pantry", .8), ("hmda", .2)),
        max_comparators=0, trend_lookback_years=0,
        record_caller={"pantry": Caller()} if mapped_connector else Caller(),
    ))
    assert interpretations == ["pantry"]
    if mapped_connector:
        assert response.error is None
        assert calls and all(p.dataset == "pantry" for p in calls)
        assert response.metrics.record_calls_total == len(calls)
        assert response.metrics.census_calls_total == 0
    else:
        assert "no connector" in response.error
        assert calls == []


def test_scope_gate_knows_configured_non_census_fields():
    import json
    from scripts.chatbot.nodes.scope_gate import is_in_scope
    class LLM:
        def extract(self, **kwargs):
            assert "weather_records" in json.loads(kwargs["user_text"])["record_sources"]
            return {"answerable": True, "reason": "Temperature is present in the configured source."}
    verdict = is_in_scope("average temperature", LLM(),
        record_sources={"weather_records": {"fields": ["temperature"]}})
    assert verdict.answerable  # No hard-coded Census rejection before routing.
