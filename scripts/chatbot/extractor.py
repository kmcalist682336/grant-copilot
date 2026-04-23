"""STEP 1: Natural language → ExtractedIntent via LLM.

Pre-LLM short-circuits surface common unsupported inputs as typed
exceptions so the orchestrator (Phase 4) can turn them into canned
responses without wasting an API call:

    ZipNotSupported  — 5-digit ZIP code input
    InputTooShort    — < 3 characters of non-whitespace
    ExtractionError  — LLM produced invalid JSON after a retry

Session context (prior resolutions within the same session) is appended
to the user message, not the system prompt, so the system prompt stays
stable across turns — a precondition for Vertex prompt caching.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any, Optional

from pydantic import ValidationError

from scripts.chatbot.llm_client import LLMClient, LLMCallError
from scripts.chatbot.models import (
    ExtractedIntent,
    extracted_intent_json_schema,
)
from scripts.chatbot.prompt_loader import load_and_render


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExtractorError(Exception):
    """Base class for all extractor-side failures."""


class ZipNotSupported(ExtractorError):
    """User typed a bare 5-digit ZIP code; we don't support ZIP queries."""


class InputTooShort(ExtractorError):
    """Input has fewer than 3 non-whitespace characters."""


class ExtractionError(ExtractorError):
    """LLM failed to produce valid ExtractedIntent JSON even after a retry."""


# ---------------------------------------------------------------------------
# Pre-LLM short-circuit patterns
# ---------------------------------------------------------------------------

_ZIP_RE = re.compile(r"^\s*\d{5}\s*$")
_MIN_INPUT_CHARS = 3


# ---------------------------------------------------------------------------
# Prompt loader (cached — the YAML + schema derive from code, so one load
# per process is correct)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    return load_and_render()


def invalidate_prompt_cache() -> None:
    """Drop the in-process prompt cache. Call if the YAML file is edited."""
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Session context formatting
# ---------------------------------------------------------------------------

def _format_session_ctx(session_ctx: dict) -> str:
    """Return the session-context block appended to the user message.

    Kept out of the system prompt so the prompt cache stays valid across
    turns. Format is a tagged JSON block; the prompt template tells the
    LLM what to do with it.
    """
    compact = json.dumps(session_ctx, separators=(",", ": "))
    return f"\n\nPRIOR SESSION CONTEXT:\n{compact}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_intent(
    user_text: str,
    llm_client: LLMClient,
    *,
    session_ctx: Optional[dict] = None,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> ExtractedIntent:
    """Extract structured geographic intent from a user query.

    Args:
        user_text: Raw user input.
        llm_client: Any LLMClient implementation (real or mock).
        session_ctx: Optional dict of prior-turn state. Stringified and
            appended to the user message.
        temperature: Sampling temperature; 0.1 by default.
        system_prompt: Override the rendered prompt (mostly for tests).

    Raises:
        ZipNotSupported, InputTooShort, ExtractionError.
    """
    # Short-circuits ----------------------------------------------------
    text = (user_text or "").strip()
    if _ZIP_RE.match(text):
        raise ZipNotSupported(f"ZIP code input not supported: {text!r}")
    if len(text) < _MIN_INPUT_CHARS:
        raise InputTooShort(
            f"Input too short to resolve geography: {user_text!r}"
        )

    # Prompt + schema ---------------------------------------------------
    if system_prompt is None:
        system_prompt, schema = _get_prompt_and_schema()
    else:
        schema = extracted_intent_json_schema()

    user_message = text
    if session_ctx:
        user_message += _format_session_ctx(session_ctx)

    # First attempt -----------------------------------------------------
    raw = _call_and_log(
        llm_client,
        system_prompt=system_prompt,
        user_text=user_message,
        schema=schema,
        temperature=temperature,
    )
    try:
        return ExtractedIntent.model_validate(raw)
    except ValidationError as first_err:
        logger.info(
            "LLM output failed ExtractedIntent validation; retrying once: %s",
            str(first_err).splitlines()[0],
        )

    # Retry with a hardened suffix -------------------------------------
    retry_message = (
        f"{user_message}\n\n"
        f"IMPORTANT: Your previous JSON failed schema validation. "
        f"Re-emit a strictly-valid ExtractedIntent JSON object. "
        f"Do NOT include prose, explanations, or extra fields."
    )
    raw2 = _call_and_log(
        llm_client,
        system_prompt=system_prompt,
        user_text=retry_message,
        schema=schema,
        temperature=temperature,
    )
    try:
        return ExtractedIntent.model_validate(raw2)
    except ValidationError as second_err:
        raise ExtractionError(
            f"LLM produced invalid ExtractedIntent after retry: {second_err}"
        ) from second_err


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _call_and_log(
    llm_client: LLMClient,
    *,
    system_prompt: str,
    user_text: str,
    schema: Any,
    temperature: float,
) -> dict:
    try:
        raw = llm_client.extract(
            system_prompt=system_prompt,
            user_text=user_text,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        raise ExtractionError(f"LLM call failed: {e}") from e
    if not isinstance(raw, dict):
        raise ExtractionError(
            f"LLM client returned non-dict: {type(raw).__name__}"
        )
    return raw
