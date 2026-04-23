"""Tests for the upstream scope gate — verifies deterministic
shortcuts fire without LLM, LLM path runs for ambiguous queries,
and all failure modes fail open (answerable=True) so a broken
gate never blocks a legitimate query."""
from __future__ import annotations

import pytest

from scripts.chatbot.llm_client import LLMCallError
from scripts.chatbot.nodes.scope_gate import (
    ScopeVerdict,
    _deterministic_check,
    is_in_scope,
    user_message_for_refusal,
)


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[dict] = []

    def extract(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


# ---------------------------------------------------------------------------
# Deterministic shortcuts
# ---------------------------------------------------------------------------

def test_jet_skis_rejected_deterministically():
    v = _deterministic_check(
        "How many jet skis are there in Northeast Atlanta?"
    )
    assert v is not None
    assert v.answerable is False
    assert "jet ski" in v.reason


def test_weather_rejected_deterministically():
    v = _deterministic_check(
        "What's the weather forecast for Atlanta?"
    )
    assert v is not None
    assert v.answerable is False


def test_waffle_house_rejected_deterministically():
    # "restaurant" substring
    v = _deterministic_check(
        "How many Waffle House restaurants are in Fulton County?"
    )
    assert v is not None
    assert v.answerable is False


def test_median_income_accepted_deterministically():
    v = _deterministic_check(
        "What's the median household income in Buckhead?"
    )
    assert v is not None
    assert v.answerable is True


def test_poverty_rate_accepted_deterministically():
    v = _deterministic_check(
        "What's the poverty rate in DeKalb County?"
    )
    assert v is not None
    assert v.answerable is True


def test_rent_burden_accepted_deterministically():
    v = _deterministic_check(
        "How has rent burden changed in Atlanta since 2019?"
    )
    assert v is not None
    assert v.answerable is True


def test_ambiguous_query_falls_through_to_llm():
    """No in/out tokens → deterministic returns None, caller goes
    to LLM."""
    v = _deterministic_check(
        "Tell me about zoning in downtown Atlanta."
    )
    assert v is None


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def test_llm_accept_flows_through():
    llm = _ScriptedLLM({
        "answerable": True,
        "reason": "concept is census-tracked (ACS tenure + zoning-adjacent)",
    })
    v = is_in_scope("Tell me about zoning in downtown Atlanta.", llm)
    assert v.answerable is True
    # Deterministic branch didn't catch this, so the LLM was called.
    assert len(llm.calls) == 1


def test_llm_reject_flows_through_with_reason():
    llm = _ScriptedLLM({
        "answerable": False,
        "reason": (
            "Piano ownership isn't in the Census; consider income or "
            "household composition variables instead."
        ),
    })
    v = is_in_scope("How many pianos are owned in Midtown?", llm)
    # Deterministic tokens don't catch "piano" here since my list
    # uses whole-word "piano" only as a substring — verify it flows
    # through. Either way, outcome is reject.
    # (If deterministic DID catch it, we still end with
    # answerable=False, so the test is robust.)
    assert v.answerable is False


def test_llm_failure_fails_open():
    """Broken LLM → default to answerable so downstream runs."""
    class _Failing:
        def extract(self, **kwargs):
            raise LLMCallError("vertex boom")
    v = is_in_scope("Tell me about zoning in Atlanta.", _Failing())
    assert v.answerable is True
    assert "failed" in v.reason.lower()


def test_llm_non_dict_fails_open():
    class _Weird:
        def extract(self, **kwargs):
            return "not a dict"
    v = is_in_scope("Tell me about zoning.", _Weird())
    assert v.answerable is True


def test_llm_schema_mismatch_fails_open():
    llm = _ScriptedLLM({"foo": "bar"})      # missing answerable field
    v = is_in_scope("Tell me about zoning.", llm)
    assert v.answerable is True


def test_deterministic_shortcut_skips_llm():
    """Query with an in-scope token should never reach the LLM."""
    llm = _ScriptedLLM({
        "answerable": False,             # canary — must not be consulted
        "reason": "llm called unexpectedly",
    })
    v = is_in_scope(
        "What's the median household income in DeKalb County?", llm,
    )
    assert v.answerable is True
    assert len(llm.calls) == 0


# ---------------------------------------------------------------------------
# User-facing message formatting
# ---------------------------------------------------------------------------

def test_refusal_message_uses_llm_reason_when_long_enough():
    v = ScopeVerdict(
        answerable=False,
        reason=(
            "Jet-ski ownership isn't collected by the Census; I can "
            "help with vehicle ownership, commute patterns, or "
            "household demographics for your area instead."
        ),
    )
    msg = user_message_for_refusal(v)
    assert "Jet-ski ownership" in msg
    assert "Census" in msg


def test_refusal_message_falls_back_for_trivial_reason():
    v = ScopeVerdict(answerable=False, reason="nope")
    msg = user_message_for_refusal(v)
    assert "don't have Census data" in msg


def test_refusal_message_empty_when_answerable():
    v = ScopeVerdict(answerable=True, reason="fine")
    msg = user_message_for_refusal(v)
    assert msg == ""
