import json
from pathlib import Path

import pytest

from scripts.chatbot.nodes.record_metric_interpreter import interpret_record_metrics, RecordMetricInterpreterError
from scripts.chatbot.record_dataset import RecordDatasetDefinition, get_record_dataset
from scripts.chatbot.models import ExtractedConcept, ExtractedFilter
from tests.test_record_dataset_adapters import pantry, _intent


def test_hmda_is_metadata_not_a_subclass():
    assert type(get_record_dataset("hmda")) is RecordDatasetDefinition
    assert not (Path(__file__).parents[1] / "scripts/chatbot/hmda_adapter.py").exists()


def test_interpreter_can_find_filter_without_exact_query_alias(pantry):
    corrected = _intent("pantry", "fulfillment rate", filters=[ExtractedFilter(
        dimension=ExtractedConcept(text="household type", dataset_hint="pantry"), value_text="families")])
    class LLM:
        def extract(self, **kwargs):
            context = json.loads(kwargs["user_text"])["dataset_definition"]
            assert "segment" in context["available_fields"]
            return corrected.model_dump()
    result, _ = interpret_record_metrics("fulfillment rate for households with children",
        _intent("pantry", "fulfillment rate"), LLM(), dataset="pantry")
    assert result.analyses[0].filters[0].value_text == "families"


@pytest.mark.parametrize("dataset,field,value", [
    ("pantry", "household type", "invented category"),
    ("hmda", "applicant race", "invented category"),
])
def test_llm_cannot_invent_category(pantry, dataset, field, value):
    intent = _intent(dataset, "fulfillment rate" if dataset == "pantry" else "denial rate",
        filters=[ExtractedFilter(dimension=ExtractedConcept(text=field, dataset_hint=dataset), value_text=value)])
    class LLM:
        def extract(self, **kwargs): return intent.model_dump()
    with pytest.raises(RecordMetricInterpreterError, match="Unsupported"):
        interpret_record_metrics("rate", intent, LLM(), dataset=dataset)
