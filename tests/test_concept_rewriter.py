"""Tests for ConceptRewriter.

Uses a DictMock LLM that returns scripted rewrites so we can verify:
  - concept list → batched LLM call
  - LLM output → validated ConceptRewrite[] in input order
  - order-scrambled output still aligns by original_text
  - missing rewrite raises ConceptRewriterError
  - invalid schema raises ConceptRewriterError
"""
from __future__ import annotations

from typing import Any

import pytest

from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.nodes.concept_rewriter import (
    ConceptRewrite, ConceptRewriteBatch, ConceptRewriterError,
    rewrite_concepts,
)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Returns a fixed dict for every extract() call."""

    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def extract(
        self, *, system_prompt: str, user_text: str, schema: Any,
        temperature: float = 0.1,
    ) -> dict:
        self.calls.append({"user_text": user_text, "temperature": temperature})
        return self.response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rewrite_single_concept_happy_path():
    llm = _ScriptedLLM({
        "rewrites": [
            {
                "original_text": "poverty in Buckhead",
                "embedding_text": "poverty rate",
                "measure_type_sought": "percentage",
                "units_expected": "percent",
                "is_total_likely": False,
                "disparity_eligible": True,
                "trend_eligible": True,
                "is_composite": False,
            }
        ]
    })
    concepts = [ExtractedConcept(
        text="poverty in Buckhead", canonical_hint="poverty rate",
    )]
    geo_refs = [ExtractedGeoRef(text="Buckhead", ref_type="neighborhood")]

    out = rewrite_concepts(concepts, geo_refs, llm)
    assert len(out) == 1
    assert out[0].embedding_text == "poverty rate"
    assert out[0].measure_type_sought == "percentage"
    assert out[0].disparity_eligible is True
    assert out[0].is_composite is False
    # One batched LLM call, regardless of concept count.
    assert len(llm.calls) == 1


def test_rewrite_batches_multiple_concepts_one_call():
    llm = _ScriptedLLM({
        "rewrites": [
            {
                "original_text": "rent burden",
                "embedding_text": "rent burden",
                "measure_type_sought": "percentage",
                "units_expected": "percent",
                "is_total_likely": False,
                "disparity_eligible": True,
                "trend_eligible": True,
                "is_composite": False,
            },
            {
                "original_text": "no vehicle",
                "embedding_text": "share without a vehicle",
                "measure_type_sought": "percentage",
                "units_expected": "percent",
                "is_total_likely": False,
                "disparity_eligible": False,
                "trend_eligible": False,
                "is_composite": False,
            },
        ]
    })
    concepts = [
        ExtractedConcept(text="rent burden", canonical_hint="rent burden"),
        ExtractedConcept(text="no vehicle", canonical_hint="share without a vehicle"),
    ]
    out = rewrite_concepts(concepts, [], llm)
    assert len(out) == 2
    assert out[0].embedding_text == "rent burden"
    assert out[1].embedding_text == "share without a vehicle"
    # Still one call for the batch.
    assert len(llm.calls) == 1


def test_rewrite_realigns_scrambled_output_by_text():
    """LLM returned rewrites in reversed order — we should still map them
    back to the original concept order by original_text."""
    llm = _ScriptedLLM({
        "rewrites": [
            {
                "original_text": "no vehicle",
                "embedding_text": "share without a vehicle",
                "measure_type_sought": "percentage",
                "units_expected": "percent",
                "is_total_likely": False,
                "disparity_eligible": False,
                "trend_eligible": False,
                "is_composite": False,
            },
            {
                "original_text": "rent burden",
                "embedding_text": "rent burden",
                "measure_type_sought": "percentage",
                "units_expected": "percent",
                "is_total_likely": False,
                "disparity_eligible": True,
                "trend_eligible": True,
                "is_composite": False,
            },
        ]
    })
    concepts = [
        ExtractedConcept(text="rent burden", canonical_hint="rent burden"),
        ExtractedConcept(text="no vehicle", canonical_hint="share without a vehicle"),
    ]
    out = rewrite_concepts(concepts, [], llm)
    # Re-aligned by original_text match
    assert out[0].original_text == "rent burden"
    assert out[1].original_text == "no vehicle"


def test_rewrite_missing_entry_raises():
    llm = _ScriptedLLM({
        "rewrites": [
            {
                "original_text": "rent burden",
                "embedding_text": "rent burden",
                "measure_type_sought": "percentage",
                "is_total_likely": False,
                "disparity_eligible": True,
                "trend_eligible": True,
                "is_composite": False,
            }
            # "no vehicle" is missing
        ]
    })
    concepts = [
        ExtractedConcept(text="rent burden"),
        ExtractedConcept(text="no vehicle"),
    ]
    with pytest.raises(ConceptRewriterError, match="did not return rewrites"):
        rewrite_concepts(concepts, [], llm)


def test_rewrite_invalid_schema_raises():
    # Missing required field `embedding_text`
    llm = _ScriptedLLM({
        "rewrites": [
            {
                "original_text": "poverty",
                "measure_type_sought": "percentage",
                "disparity_eligible": True,
            }
        ]
    })
    with pytest.raises(ConceptRewriterError, match="schema validation"):
        rewrite_concepts(
            [ExtractedConcept(text="poverty")], [], llm,
        )


def test_rewrite_empty_concepts_short_circuits():
    """No concepts → no LLM call, empty list."""
    llm = _ScriptedLLM({"rewrites": []})
    out = rewrite_concepts([], [], llm)
    assert out == []
    assert llm.calls == []


def test_rewrite_llm_non_dict_raises():
    class _BadLLM:
        def extract(self, **kwargs):
            return "not a dict"
    with pytest.raises(ConceptRewriterError, match="non-dict"):
        rewrite_concepts(
            [ExtractedConcept(text="poverty")], [], _BadLLM(),
        )


def test_rewrite_llm_call_error_wrapped():
    from scripts.chatbot.llm_client import LLMCallError

    class _FailingLLM:
        def extract(self, **kwargs):
            raise LLMCallError("upstream hiccup")

    with pytest.raises(ConceptRewriterError, match="rewriter LLM call failed"):
        rewrite_concepts(
            [ExtractedConcept(text="poverty")], [], _FailingLLM(),
        )


# ---------------------------------------------------------------------------
# Contract sanity — ConceptRewriteBatch schema covers all fields
# ---------------------------------------------------------------------------

def test_batch_schema_includes_required_fields():
    schema = ConceptRewriteBatch.model_json_schema()
    # Resolve the ConceptRewrite definition to check fields
    defs = schema.get("$defs", {})
    cr = defs.get("ConceptRewrite", {})
    props = cr.get("properties", {})
    for f in (
        "original_text", "embedding_text", "measure_type_sought",
        "is_total_likely", "disparity_eligible", "trend_eligible",
        "is_composite",
    ):
        assert f in props, f"ConceptRewrite.{f} missing from schema"
