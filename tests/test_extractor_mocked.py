"""Extractor tests — all deterministic, no Vertex calls."""
from __future__ import annotations

import pytest

from scripts.chatbot.extractor import (
    ExtractionError,
    InputTooShort,
    ZipNotSupported,
    extract_intent,
    invalidate_prompt_cache,
)
from scripts.chatbot.llm_client import MockLLMClient
from scripts.chatbot.models import ExtractedIntent


# ---------------------------------------------------------------------------
# Short-circuits
# ---------------------------------------------------------------------------

def test_rejects_five_digit_zip():
    with pytest.raises(ZipNotSupported):
        extract_intent("30303", MockLLMClient())


def test_rejects_zip_with_surrounding_whitespace():
    with pytest.raises(ZipNotSupported):
        extract_intent("  30303  ", MockLLMClient())


def test_non_zip_numeric_not_rejected_as_zip():
    # A 3- or 4-digit number isn't a ZIP; should pass to the LLM path.
    mock = MockLLMClient(responses={
        "123": {"geo_refs": [], "intent_type": "context"},
    })
    intent = extract_intent("123", mock)
    assert intent.geo_refs == []


def test_rejects_empty_input():
    with pytest.raises(InputTooShort):
        extract_intent("", MockLLMClient())


def test_rejects_too_short_input():
    with pytest.raises(InputTooShort):
        extract_intent("ab", MockLLMClient())


def test_rejects_whitespace_only_input():
    with pytest.raises(InputTooShort):
        extract_intent("   ", MockLLMClient())


# ---------------------------------------------------------------------------
# Happy path — single-turn LLM extraction
# ---------------------------------------------------------------------------

def test_returns_extracted_intent_on_valid_response():
    mock = MockLLMClient(responses={
        "income in buckhead": {
            "geo_refs": [
                {
                    "text": "Buckhead",
                    "ref_type": "neighborhood",
                    "qualifier": "in",
                    "scope_hint": "Atlanta",
                }
            ],
            "concepts": [
                {"text": "income", "canonical_hint": "median household income"},
            ],
            "intent_type": "exact",
        },
    })
    intent = extract_intent("income in Buckhead", mock)
    assert isinstance(intent, ExtractedIntent)
    assert len(intent.geo_refs) == 1
    assert intent.geo_refs[0].ref_type == "neighborhood"
    assert intent.geo_refs[0].text == "Buckhead"
    assert intent.intent_type == "exact"
    # Phase G: concepts now flow through
    assert len(intent.concepts) == 1
    assert intent.concepts[0].text == "income"
    assert intent.concepts[0].canonical_hint == "median household income"
    assert intent.concepts[0].is_composite is False
    # Defaults
    assert intent.temporal_intent == "latest"
    assert intent.years == []


def test_temporal_change_with_years_parsed():
    mock = MockLLMClient(responses={
        "poverty in dekalb 2020 to 2023": {
            "geo_refs": [
                {"text": "DeKalb", "ref_type": "administrative",
                 "scope_hint": "Georgia"},
            ],
            "concepts": [
                {"text": "poverty", "canonical_hint": "poverty rate"},
            ],
            "temporal_intent": "change",
            "years": [2020, 2023],
            "intent_type": "exact",
        },
    })
    intent = extract_intent("poverty in DeKalb 2020 to 2023", mock)
    assert intent.temporal_intent == "change"
    assert intent.years == [2020, 2023]


def test_composite_concept_flag_carries_through():
    mock = MockLLMClient(responses={
        "gentrification in npu-m": {
            "geo_refs": [
                {"text": "NPU-M", "ref_type": "neighborhood",
                 "scope_hint": "Atlanta"},
            ],
            "concepts": [
                {"text": "gentrification", "canonical_hint": "gentrification",
                 "is_composite": True},
            ],
            "intent_type": "exact",
        },
    })
    intent = extract_intent("gentrification in NPU-M", mock)
    assert intent.concepts[0].is_composite is True


def test_passes_temperature_and_schema_through():
    mock = MockLLMClient(responses={
        "atlanta": {"geo_refs": [], "intent_type": "context"},
    })
    extract_intent("Atlanta", mock, temperature=0.0)
    assert len(mock.calls) == 1
    call = mock.calls[0]
    assert call["temperature"] == 0.0
    # Schema is the pydantic-derived JSON schema; just verify it's a dict.
    assert isinstance(call["schema"], dict)
    assert "properties" in call["schema"]


# ---------------------------------------------------------------------------
# Retry on validation failure
# ---------------------------------------------------------------------------

class _SequenceMock:
    """LLM mock that returns scripted responses in order, one per call."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        self.calls.append(user_text)
        if not self._responses:
            raise AssertionError("ran out of scripted responses")
        return self._responses.pop(0)


def test_retries_once_on_validation_failure():
    # First response violates the geo_refs[].ref_type Literal; second is valid.
    mock = _SequenceMock([
        {
            "geo_refs": [{"text": "Atlanta", "ref_type": "not_a_valid_type"}],
            "intent_type": "exact",
        },
        {
            "geo_refs": [{"text": "Atlanta", "ref_type": "administrative"}],
            "intent_type": "exact",
        },
    ])
    intent = extract_intent("Atlanta", mock)
    assert intent.intent_type == "exact"
    assert len(mock.calls) == 2
    # Second call carries the hardened suffix.
    assert "schema validation" in mock.calls[1]


def test_gives_up_after_failed_retry():
    # Both responses violate the intent_type Literal.
    mock = _SequenceMock([
        {"geo_refs": [], "intent_type": "bogus"},
        {"geo_refs": [], "intent_type": "also_bogus"},
    ])
    with pytest.raises(ExtractionError):
        extract_intent("Atlanta", mock)
    assert len(mock.calls) == 2


def test_gives_up_if_client_returns_non_dict():
    class StringMock:
        calls: list = []
        def extract(self, **kw):
            self.calls.append(kw)
            return "not a dict"

    with pytest.raises(ExtractionError):
        extract_intent("Atlanta", StringMock())


# ---------------------------------------------------------------------------
# Session context injection
# ---------------------------------------------------------------------------

def test_session_ctx_appended_to_user_message():
    captured: dict = {}

    class CaptureMock:
        def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
            captured["user_text"] = user_text
            captured["system_prompt"] = system_prompt
            return {"geo_refs": [], "intent_type": "context"}

    cap = CaptureMock()
    extract_intent(
        "same place",
        cap,
        session_ctx={
            "recent_resolutions": [
                {"span": "Buckhead", "resolved_as": "SD004 (neighborhood)"}
            ]
        },
    )
    assert "same place" in captured["user_text"]
    assert "SESSION CONTEXT" in captured["user_text"]
    assert "SD004" in captured["user_text"]  # from the session_ctx dict
    # The session-ctx marker must only live on the user-message side so
    # the system prompt (and its cache) stays stable across turns.
    assert "SESSION CONTEXT" not in captured["system_prompt"]
    assert "SD004" not in captured["system_prompt"]


def test_no_session_ctx_keeps_user_message_clean():
    captured: dict = {}

    class CaptureMock:
        def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
            captured["user_text"] = user_text
            return {"geo_refs": [], "intent_type": "context"}

    extract_intent("Atlanta", CaptureMock())
    assert captured["user_text"].strip() == "Atlanta"
    assert "SESSION CONTEXT" not in captured["user_text"]


# ---------------------------------------------------------------------------
# Prompt cache behavior
# ---------------------------------------------------------------------------

def test_prompt_is_loaded_and_cached_within_process():
    """The prompt-loader is memoized — two extract calls hit the same
    rendered system prompt instance."""
    invalidate_prompt_cache()
    seen: list[str] = []

    class CaptureMock:
        def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
            seen.append(system_prompt)
            return {"geo_refs": [], "intent_type": "context"}

    extract_intent("Atlanta", CaptureMock())
    extract_intent("Buckhead neighborhood", CaptureMock())
    assert len(seen) == 2
    assert seen[0] is seen[1]  # identity check — same cached object


# ---------------------------------------------------------------------------
# MockLLMClient behavior
# ---------------------------------------------------------------------------

def test_mock_client_records_calls():
    mock = MockLLMClient(responses={
        "atlanta": {"geo_refs": [], "intent_type": "context"},
    })
    extract_intent("Atlanta", mock, temperature=0.2)
    assert len(mock.calls) == 1
    assert mock.calls[0]["temperature"] == 0.2


def test_mock_client_raises_key_error_for_unmapped_input():
    mock = MockLLMClient(responses={})
    with pytest.raises(KeyError):
        extract_intent("something else entirely", mock)


def test_mock_client_first_line_fallback_matches_session_ctx_appended():
    """When session_ctx is appended, the mock matches on the first line
    so tests don't need to re-script every combination."""
    mock = MockLLMClient(responses={
        "same place": {"geo_refs": [], "intent_type": "context"},
    })
    intent = extract_intent(
        "same place",
        mock,
        session_ctx={"recent": "Buckhead"},
    )
    assert intent.intent_type == "context"
