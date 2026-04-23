"""Tests for route_concepts_with_critic.

Wires together the three Phase 1 moving parts (rewriter, router,
critic) with scripted stand-ins so we can verify:
  - accept path → tier_semantic resolution with routed_result attached
  - retry path → second-pass rewrite + reroute → accepted
  - fallback path → tier-cascade invoked when cmap/metadata provided
  - fallback path → unresolved when cmap/metadata unavailable
  - empty concepts list short-circuits

The function returns a RoutingOutcome dataclass; tests inspect both
``outcome.resolutions`` (the planner-ready list) and ``outcome.rewrites``
(the IntentFacets used by downstream Expander stages).
"""
from __future__ import annotations

from typing import Any

import pytest

from scripts.chatbot.agent_routing import RoutingOutcome, route_concepts_with_critic
from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.nodes.router_critic import CriticVerdict
from scripts.chatbot.semantic_router import (
    RoutedHit, RoutedResult, RoutedTarget,
)


# ---------------------------------------------------------------------------
# Scripted stand-ins
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """Returns queued responses in order. Fails if queue exhausts."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                f"_ScriptedLLM exhausted; unexpected call #{len(self.calls)}"
            )
        return self.responses.pop(0)


class _FakeRouter:
    """Returns a pre-scripted RoutedResult indexed by query text."""
    def __init__(self, results: dict[str, RoutedResult]):
        self.results = results
        self.calls: list[tuple[str, list[str] | None]] = []

    def route(self, text: str, *, top_k: int = 10,
              require_coverage_at: list[str] | None = None) -> RoutedResult:
        self.calls.append((text, require_coverage_at))
        if text not in self.results:
            raise KeyError(f"_FakeRouter has no result for {text!r}")
        return self.results[text]


def _hit(table_id: str, score: float = 0.9) -> RoutedHit:
    return RoutedHit(
        rowid=1, card_type="table_question_canonical",
        text="canonical question", target_table_id=table_id,
        target_dataset="acs/acs5", target_year=2024,
        cosine=score, weight=1.0, polarity="positive",
        weighted_score=score, is_total=0,
    )


def _target(table_id: str, score: float = 3.5,
            variable_id: str | None = None) -> RoutedTarget:
    return RoutedTarget(
        target_table_id=table_id, target_dataset="acs/acs5",
        target_year=2024, target_variable_id=variable_id,
        aggregate_score=score, n_positive_hits=1, n_negative_hits=0,
        best_hit=_hit(table_id, score / 4),
        matched_card_types=["table_question_canonical"],
    )


def _routed(query: str, table_id: str,
            score: float = 3.5) -> RoutedResult:
    t = _target(table_id, score=score)
    v = _target(table_id, score=score, variable_id=f"{table_id}_001E")
    return RoutedResult(
        query=query, top_tables=[t], top_variables=[v],
        n_hits_retrieved=2, n_hits_after_filter=2,
    )


# ---------------------------------------------------------------------------
# Accept path
# ---------------------------------------------------------------------------

def test_accept_path_returns_tier_semantic():
    concepts = [ExtractedConcept(
        text="poverty in Buckhead", canonical_hint="poverty rate",
    )]
    geo_refs = [ExtractedGeoRef(text="Buckhead", ref_type="neighborhood")]

    rewrite_response = {"rewrites": [{
        "original_text": "poverty in Buckhead",
        "embedding_text": "poverty rate",
        "measure_type_sought": "percentage",
        "units_expected": "percent",
        "is_total_likely": False,
        "disparity_eligible": True,
        "trend_eligible": True,
        "is_composite": False,
    }]}
    critic_response = {"verdicts": [{
        "original_text": "poverty in Buckhead",
        "decision": "accept",
        "reason": "canonical B17001 top-ranked",
        "rewrite_hint": None,
    }]}
    llm = _ScriptedLLM([rewrite_response, critic_response])
    router = _FakeRouter({"poverty rate": _routed("poverty rate", "B17001")})

    outcome = route_concepts_with_critic(
        concepts, geo_refs, data_levels=["tract"],
        semantic_router=router, llm=llm,
    )
    assert isinstance(outcome, RoutingOutcome)
    assert len(outcome.resolutions) == 1
    assert len(outcome.rewrites) == 1
    assert outcome.resolutions[0].tier == "tier_semantic"
    assert outcome.resolutions[0].routed_result is not None
    assert any("poverty rate" in n for n in outcome.resolutions[0].notes)
    # One rewrite + one critic LLM call; one router call.
    assert len(llm.calls) == 2
    assert len(router.calls) == 1


# ---------------------------------------------------------------------------
# Retry path
# ---------------------------------------------------------------------------

def test_retry_path_runs_second_pass_then_accepts():
    """Critic says retry with hint → rewriter re-runs with hint → router
    returns a better candidate → resolution is tier_semantic.

    Importantly: NO second critic call. We retry-accept after one round.
    """
    concepts = [ExtractedConcept(
        text="income",
        canonical_hint="median household income",
    )]
    geo_refs: list[ExtractedGeoRef] = []

    rewrite1 = {"rewrites": [{
        "original_text": "income",
        "embedding_text": "median household income",
        "measure_type_sought": "currency",
        "units_expected": "dollars",
        "is_total_likely": False,
        "disparity_eligible": True,
        "trend_eligible": True,
        "is_composite": False,
    }]}
    critic = {"verdicts": [{
        "original_text": "income",
        "decision": "retry",
        "reason": "profile DP03 outranked canonical B19013",
        "rewrite_hint": "prefer B19013 over DP03",
    }]}
    rewrite2 = {"rewrites": [{
        "original_text": "income",
        "embedding_text": "median household income B19013",
        "measure_type_sought": "currency",
        "units_expected": "dollars",
        "is_total_likely": False,
        "disparity_eligible": True,
        "trend_eligible": True,
        "is_composite": False,
    }]}
    llm = _ScriptedLLM([rewrite1, critic, rewrite2])
    router = _FakeRouter({
        "median household income": _routed(
            "median household income", "DP03", score=3.9,
        ),
        "median household income B19013": _routed(
            "median household income B19013", "B19013", score=4.1,
        ),
    })

    outcome = route_concepts_with_critic(
        concepts, geo_refs, data_levels=["tract"],
        semantic_router=router, llm=llm,
    )
    assert len(outcome.resolutions) == 1
    assert outcome.resolutions[0].tier == "tier_semantic"
    # Second-pass routed result is what ended up attached.
    rr = outcome.resolutions[0].routed_result
    assert rr.top_tables[0].target_table_id == "B19013"
    # Retry-acceptance note is preserved.
    assert any(
        "retry accepted" in n for n in outcome.resolutions[0].notes
    )
    # LLM calls: rewrite1 + critic + rewrite2 (no second critic)
    assert len(llm.calls) == 3
    # Router called twice: once per rewrite.
    assert len(router.calls) == 2


def test_max_retries_zero_skips_retry_loop():
    """With max_retries=0, a 'retry' verdict is treated as fallback."""
    concepts = [ExtractedConcept(text="income")]

    rewrite1 = {"rewrites": [{
        "original_text": "income",
        "embedding_text": "median household income",
        "measure_type_sought": "currency",
        "units_expected": "dollars",
        "is_total_likely": False,
        "disparity_eligible": True,
        "trend_eligible": True,
        "is_composite": False,
    }]}
    critic = {"verdicts": [{
        "original_text": "income",
        "decision": "retry",
        "reason": "something iffy",
        "rewrite_hint": "hint",
    }]}
    llm = _ScriptedLLM([rewrite1, critic])
    router = _FakeRouter({
        "median household income": _routed(
            "median household income", "DP03",
        ),
    })

    outcome = route_concepts_with_critic(
        concepts, [], data_levels=["tract"],
        semantic_router=router, llm=llm,
        max_retries=0,
    )
    # Without cmap/metadata, fallback surfaces unresolved.
    assert outcome.resolutions[0].tier == "unresolved"
    assert any(
        "no cmap / metadata_db" in n for n in outcome.resolutions[0].notes
    )


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_fallback_without_cmap_surfaces_unresolved():
    concepts = [ExtractedConcept(text="unicorn density")]
    rewrite_response = {"rewrites": [{
        "original_text": "unicorn density",
        "embedding_text": "unicorn density",
        "measure_type_sought": "count",
        "units_expected": "persons",
        "is_total_likely": False,
        "disparity_eligible": False,
        "trend_eligible": False,
        "is_composite": False,
    }]}
    # Empty routed → critic short-circuits to fallback without LLM call.
    llm = _ScriptedLLM([rewrite_response])
    empty_routed = RoutedResult(
        query="unicorn density", top_tables=[], top_variables=[],
        n_hits_retrieved=0, n_hits_after_filter=0,
    )
    router = _FakeRouter({"unicorn density": empty_routed})

    outcome = route_concepts_with_critic(
        concepts, [], data_levels=["tract"],
        semantic_router=router, llm=llm,
    )
    assert outcome.resolutions[0].tier == "unresolved"
    # One rewrite LLM call, zero critic calls (short-circuit).
    assert len(llm.calls) == 1


def test_empty_concepts_returns_empty():
    llm = _ScriptedLLM([])
    router = _FakeRouter({})
    outcome = route_concepts_with_critic(
        [], [], data_levels=[], semantic_router=router, llm=llm,
    )
    assert outcome.resolutions == []
    assert outcome.rewrites == []
    assert llm.calls == []


# ---------------------------------------------------------------------------
# Mixed batch — one accept, one fallback in the same query
# ---------------------------------------------------------------------------

def test_mixed_batch_accept_and_fallback():
    concepts = [
        ExtractedConcept(text="poverty", canonical_hint="poverty rate"),
        ExtractedConcept(text="unicorn count"),
    ]
    rewrite_response = {"rewrites": [
        {
            "original_text": "poverty",
            "embedding_text": "poverty rate",
            "measure_type_sought": "percentage",
            "units_expected": "percent",
            "is_total_likely": False,
            "disparity_eligible": True,
            "trend_eligible": True,
            "is_composite": False,
        },
        {
            "original_text": "unicorn count",
            "embedding_text": "unicorn count",
            "measure_type_sought": "count",
            "units_expected": "persons",
            "is_total_likely": True,
            "disparity_eligible": False,
            "trend_eligible": False,
            "is_composite": False,
        },
    ]}
    # Only "poverty rate" has candidates → critic is asked about it only.
    critic_response = {"verdicts": [{
        "original_text": "poverty",
        "decision": "accept",
        "reason": "",
        "rewrite_hint": None,
    }]}
    llm = _ScriptedLLM([rewrite_response, critic_response])
    router = _FakeRouter({
        "poverty rate": _routed("poverty rate", "B17001"),
        "unicorn count": RoutedResult(
            query="unicorn count", top_tables=[], top_variables=[],
            n_hits_retrieved=0, n_hits_after_filter=0,
        ),
    })

    outcome = route_concepts_with_critic(
        concepts, [], data_levels=["tract"],
        semantic_router=router, llm=llm,
    )
    assert outcome.resolutions[0].tier == "tier_semantic"
    assert outcome.resolutions[1].tier == "unresolved"
    # Rewrite (1) + critic-over-populated (1) = 2 LLM calls.
    assert len(llm.calls) == 2
    # Both routed (one non-empty, one empty).
    assert len(router.calls) == 2
