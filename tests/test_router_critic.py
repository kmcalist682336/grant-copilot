"""Tests for RouterCritic.

Uses scripted LLM + hand-crafted RoutedResult fixtures. Verifies:
  - accept / retry / fallback decisions flow through
  - empty-candidate concepts short-circuit to fallback without an LLM call
  - order realignment by original_text
  - schema-invalid outputs raise
  - critique_routing tolerates the specific "DP03 outranks B19013" signal
    (the concrete canonical-table-bias case)
"""
from __future__ import annotations

from typing import Any

import pytest

from scripts.chatbot.nodes.concept_rewriter import ConceptRewrite
from scripts.chatbot.nodes.router_critic import (
    CriticVerdict, CriticVerdictBatch, RouterCriticError,
    build_candidate_payload, critique_routing,
)
from scripts.chatbot.semantic_router import (
    RoutedHit, RoutedResult, RoutedTarget,
)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _SequencedLLM:
    """Returns a different response per call — used to simulate the
    'critic drops a concept, retry fills the gap' path."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("_SequencedLLM exhausted")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rewrite(
    text: str, measure: str = "percentage",
    is_total: bool = False,
) -> ConceptRewrite:
    return ConceptRewrite(
        original_text=text,
        embedding_text=text,
        measure_type_sought=measure,  # type: ignore[arg-type]
        units_expected="percent",
        is_total_likely=is_total,
        disparity_eligible=False,
        trend_eligible=True,
        is_composite=False,
    )


def _hit(
    table_id: str, dataset: str, year: int,
    variable_id: str | None, card_type: str,
    text: str, cosine: float = 0.9,
    is_total: int = 0,
) -> RoutedHit:
    return RoutedHit(
        rowid=1, card_type=card_type, text=text,
        target_table_id=table_id, target_dataset=dataset,
        target_year=year, target_variable_id=variable_id,
        cosine=cosine, weight=1.0, polarity="positive",
        weighted_score=cosine, is_total=is_total,
    )


def _target(
    table_id: str, dataset: str, year: int,
    variable_id: str | None, score: float,
    card_type: str, text: str, is_total: int = 0,
) -> RoutedTarget:
    return RoutedTarget(
        target_table_id=table_id, target_dataset=dataset,
        target_year=year, target_variable_id=variable_id,
        aggregate_score=score, n_positive_hits=1, n_negative_hits=0,
        best_hit=_hit(
            table_id, dataset, year, variable_id,
            card_type, text, is_total=is_total,
        ),
        matched_card_types=[card_type],
    )


def _routed(
    query: str,
    tables: list[RoutedTarget] = None,
    variables: list[RoutedTarget] = None,
) -> RoutedResult:
    return RoutedResult(
        query=query,
        top_tables=tables or [],
        top_variables=variables or [],
        n_hits_retrieved=len(tables or []) + len(variables or []),
        n_hits_after_filter=len(tables or []) + len(variables or []),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_accept_decision_pass_through():
    rw = _rewrite("poverty rate")
    rr = _routed(
        "poverty rate",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.10,
            "table_question_canonical", "poverty rate in this area",
        )],
    )
    llm = _ScriptedLLM({
        "verdicts": [{
            "original_text": "poverty rate",
            "decision": "accept",
            "reason": "canonical B17001 top-ranked",
            "rewrite_hint": None,
        }]
    })
    verdicts = critique_routing([rw], [rr], llm)
    assert len(verdicts) == 1
    assert verdicts[0].decision == "accept"
    assert verdicts[0].rewrite_hint is None
    assert len(llm.calls) == 1


def test_retry_decision_includes_rewrite_hint():
    """The DP03-outranks-B19013 scenario — the critic should retry."""
    rw = _rewrite("median household income", measure="currency")
    rr = _routed(
        "median household income",
        tables=[
            _target(
                "DP03", "acs/acs5/sptprofile", 2021, None, 3.93,
                "var_question", "median household income count",
            ),
            _target(
                "B19013", "acs/acs5", 2024, None, 3.51,
                "table_question_canonical",
                "median household income past 12 months",
            ),
        ],
    )
    llm = _ScriptedLLM({
        "verdicts": [{
            "original_text": "median household income",
            "decision": "retry",
            "reason": "profile table DP03 outranked canonical B19013",
            "rewrite_hint": "prefer detailed B-series table B19013 over profile DP03",
        }]
    })
    verdicts = critique_routing([rw], [rr], llm)
    assert verdicts[0].decision == "retry"
    assert "B19013" in verdicts[0].rewrite_hint


def test_empty_candidates_short_circuit_without_llm_call():
    rw = _rewrite("made-up concept")
    rr = _routed("made-up concept")    # no tables, no variables
    llm = _ScriptedLLM({"verdicts": []})
    verdicts = critique_routing([rw], [rr], llm)
    assert len(verdicts) == 1
    assert verdicts[0].decision == "fallback"
    # No LLM call was made — the empty-candidate case is mechanical.
    assert llm.calls == []


def test_mixed_empty_and_populated_sends_only_populated_to_llm():
    """Two concepts: one has candidates, one doesn't. Only the populated
    one goes to the LLM; the empty one falls back mechanically."""
    rw1 = _rewrite("poverty rate")
    rw2 = _rewrite("unicorns")
    rr1 = _routed(
        "poverty rate",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    rr2 = _routed("unicorns")       # empty

    llm = _ScriptedLLM({
        "verdicts": [{
            "original_text": "poverty rate",
            "decision": "accept",
            "reason": "B17001 top-ranked",
            "rewrite_hint": None,
        }]
    })
    verdicts = critique_routing([rw1, rw2], [rr1, rr2], llm)
    assert verdicts[0].decision == "accept"
    assert verdicts[1].decision == "fallback"
    # One LLM call for the populated concept only.
    assert len(llm.calls) == 1


def test_realignment_by_original_text_when_llm_scrambles_order():
    rw1 = _rewrite("rent burden")
    rw2 = _rewrite("poverty rate")
    rr_pop = _routed(
        "x",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    llm = _ScriptedLLM({
        "verdicts": [
            {
                "original_text": "poverty rate",
                "decision": "accept",
                "reason": "",
                "rewrite_hint": None,
            },
            {
                "original_text": "rent burden",
                "decision": "retry",
                "reason": "",
                "rewrite_hint": "rent burden hint",
            },
        ]
    })
    verdicts = critique_routing([rw1, rw2], [rr_pop, rr_pop], llm)
    assert verdicts[0].original_text == "rent burden"
    assert verdicts[0].decision == "retry"
    assert verdicts[1].original_text == "poverty rate"
    assert verdicts[1].decision == "accept"


def test_retry_fills_missing_verdicts(caplog):
    """First critic pass drops 'poverty rate'; the retry pass, handed
    just that missing concept, returns a real verdict. No synthesized
    defaults should be needed."""
    rw1 = _rewrite("rent burden")
    rw2 = _rewrite("poverty rate")
    rr_pop = _routed(
        "x",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    llm = _SequencedLLM([
        {
            "verdicts": [{
                "original_text": "rent burden",
                "decision": "accept",
                "reason": "top hit looks good",
                "rewrite_hint": None,
            }]
        },
        {
            "verdicts": [{
                "original_text": "poverty rate",
                "decision": "retry",
                "reason": "profile table outranked B17001",
                "rewrite_hint": "prefer B17001",
            }]
        },
    ])
    with caplog.at_level("INFO"):
        verdicts = critique_routing(
            [rw1, rw2], [rr_pop, rr_pop], llm,
        )
    assert len(llm.calls) == 2           # original + retry pass
    assert len(verdicts) == 2
    assert verdicts[0].original_text == "rent burden"
    assert verdicts[0].decision == "accept"
    assert verdicts[1].original_text == "poverty rate"
    assert verdicts[1].decision == "retry"
    assert "retrying" in " ".join(rec.message for rec in caplog.records)


def test_retry_still_missing_defaults_to_accept(caplog):
    """Both the first and retry passes drop 'poverty rate'. After the
    retry fails to fill the gap, we fall back to the synthesized
    'accept' default — but the pipeline does not abort."""
    rw1 = _rewrite("rent burden")
    rw2 = _rewrite("poverty rate")
    rr_pop = _routed(
        "x",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    llm = _SequencedLLM([
        {
            "verdicts": [{
                "original_text": "rent burden",
                "decision": "accept",
                "reason": "",
                "rewrite_hint": None,
            }]
        },
        {
            # Retry pass also drops the single concept it was handed.
            "verdicts": [],
        },
    ])
    with caplog.at_level("WARNING"):
        verdicts = critique_routing(
            [rw1, rw2], [rr_pop, rr_pop], llm,
        )
    assert len(llm.calls) == 2
    assert verdicts[1].original_text == "poverty rate"
    assert verdicts[1].decision == "accept"
    assert "synthesized" in verdicts[1].reason
    assert any(
        "didn't verdict" in rec.message
        for rec in caplog.records
    )


def test_invalid_schema_raises():
    rw = _rewrite("poverty rate")
    rr = _routed(
        "x",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    llm = _ScriptedLLM({
        "verdicts": [{
            # missing "decision" field
            "original_text": "poverty rate",
            "reason": "oops",
        }]
    })
    with pytest.raises(RouterCriticError, match="schema validation"):
        critique_routing([rw], [rr], llm)


def test_mismatched_input_lengths_raises():
    rw = _rewrite("rent burden")
    rr1 = _routed("x")
    rr2 = _routed("y")
    llm = _ScriptedLLM({"verdicts": []})
    with pytest.raises(RouterCriticError, match="mismatched lengths"):
        critique_routing([rw], [rr1, rr2], llm)


def test_empty_inputs_return_empty():
    llm = _ScriptedLLM({"verdicts": []})
    assert critique_routing([], [], llm) == []
    assert llm.calls == []


def test_llm_failure_wrapped():
    from scripts.chatbot.llm_client import LLMCallError

    class _FailingLLM:
        def extract(self, **kwargs):
            raise LLMCallError("upstream hiccup")

    rw = _rewrite("poverty rate")
    rr = _routed(
        "x",
        tables=[_target(
            "B17001", "acs/acs5", 2024, None, 4.0,
            "table_question_canonical", "poverty",
        )],
    )
    with pytest.raises(RouterCriticError, match="critic LLM call failed"):
        critique_routing([rw], [rr], _FailingLLM())


# ---------------------------------------------------------------------------
# build_candidate_payload shape
# ---------------------------------------------------------------------------

def test_candidate_payload_includes_tables_and_variables():
    rw = _rewrite("median household income", measure="currency")
    rr = _routed(
        "median household income",
        tables=[_target(
            "B19013", "acs/acs5", 2024, None, 3.51,
            "table_question_canonical", "median household income",
        )],
        variables=[_target(
            "B19013", "acs/acs5", 2024, "B19013_001E", 3.51,
            "var_question", "median household income variable",
            is_total=1,
        )],
    )
    payload = build_candidate_payload(rw, rr)
    assert payload["original_text"] == "median household income"
    assert payload["embedding_text"] == "median household income"
    assert payload["measure_type_sought"] == "currency"
    assert len(payload["top_tables"]) == 1
    assert payload["top_tables"][0]["table_id"] == "B19013"
    assert len(payload["top_variables"]) == 1
    assert payload["top_variables"][0]["variable_id"] == "B19013_001E"
    assert payload["top_variables"][0]["is_total"] is True
