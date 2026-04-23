"""FollowupGapFinder — frame-aware "what's missing" suggestions.

Phase 4 post-aggregate LLM node. Given:
  - the user's query
  - the matched GrantFrame (rhetorical_target + required_additional_concepts)
  - what was actually fetched (concept texts that yielded values)
  - what's still missing from the frame's wish list

it asks the LLM to propose 2-4 user-voice followup questions that
specifically target grant-narrative GAPS, not generic related topics.

One LLM call per query. Results attach to QueryResponse and the
synthesizer renders them as "Try next:" bullets at the bottom of the
prose answer.

Public API:
    SuggestedFollowup            — one suggestion
    SuggestedFollowupBatch       — LLM wire format
    find_followups               — entry point
    FollowupGapFinderError       — raised on invalid/unparseable output
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.aggregator import AggregatedResult
from scripts.chatbot.frames import Frame
from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class SuggestedFollowup(BaseModel):
    """One grant-narrative followup suggestion."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(
        ..., description="User-voice followup (under 20 words).",
    )
    rationale: str = Field(
        ..., description="One-sentence why this strengthens the case.",
    )
    frame_relevance: Optional[str] = Field(
        default=None,
        description="Why this followup matches the matched frame.",
    )


class SuggestedFollowupBatch(BaseModel):
    """LLM wire format — 2-4 followups, strongest first."""

    model_config = ConfigDict(extra="ignore")

    followups: list[SuggestedFollowup] = Field(default_factory=list)


class FollowupGapFinderError(RuntimeError):
    """Raised when the LLM call fails or output is unparseable."""


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "followup_gap_finder.yaml"
)


@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    template = load_prompt_template(_PROMPT_PATH)
    schema = SuggestedFollowupBatch.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Gap computation
# ---------------------------------------------------------------------------

def _concept_key(text: str) -> str:
    return (text or "").strip().lower()


def _compute_gap(
    frame: Optional[Frame],
    aggregated: AggregatedResult,
) -> tuple[list[str], list[str]]:
    """Return (fetched_concepts, missing_concepts) as two lists of
    user-facing concept names.

    ``fetched`` is the set of concepts that produced at least one
    non-null AggregatedValue. ``missing`` is what the frame's
    required_additional_concepts list said we'd need minus what we
    actually have.
    """
    # Concepts that actually returned data (had a value, ratio, or
    # components on at least one AggregatedValue).
    fetched_keys: set[str] = set()
    for v in aggregated.values:
        text = (
            v.concept.canonical_hint
            or v.concept.text or ""
        )
        if (v.value is not None or v.ratio is not None
                or v.components is not None):
            fetched_keys.add(_concept_key(text))

    # Reconstruct "display" names for the fetched concepts (preserve
    # case and hint-forms users would recognize).
    fetched_display: dict[str, str] = {}
    for v in aggregated.values:
        text = (
            v.concept.canonical_hint
            or v.concept.text or ""
        )
        fetched_display.setdefault(_concept_key(text), text)

    fetched = [
        fetched_display[k] for k in sorted(fetched_display)
    ]

    if frame is None:
        return fetched, []

    missing: list[str] = []
    required = frame.required_additional_concepts or []
    for req in required:
        if _concept_key(req) not in fetched_keys:
            missing.append(req)

    return fetched, missing


# ---------------------------------------------------------------------------
# User payload
# ---------------------------------------------------------------------------

def _build_user_payload(
    *,
    user_query: str,
    frame: Optional[Frame],
    fetched_concepts: list[str],
    missing_concepts: list[str],
) -> str:
    payload = {
        "user_query": user_query,
        "frame": frame.name if frame else "none",
        "rhetorical_target": (
            frame.rhetorical_target if frame else ""
        ),
        "fetched_concepts": fetched_concepts,
        "missing_concepts": missing_concepts,
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_followups(
    *,
    user_query: str,
    frame: Optional[Frame],
    aggregated: AggregatedResult,
    llm: LLMClient,
    temperature: float = 0.4,
    system_prompt: Optional[str] = None,
) -> list[SuggestedFollowup]:
    """Produce frame-aware followup suggestions.

    ``frame`` is optional — when None, we fall back to an "add trend
    + disparity angle" heuristic inside the LLM prompt.

    Returns 2-4 SuggestedFollowup items. On LLM error or parse failure,
    returns an empty list rather than raising — followups are a
    non-critical enrichment, not a pipeline blocker.
    """
    # Skip the LLM entirely when there's nothing to follow up on.
    if not aggregated.values and not user_query.strip():
        return []

    if system_prompt is None:
        system_prompt, schema = _get_prompt_and_schema()
    else:
        schema = SuggestedFollowupBatch.model_json_schema()

    fetched, missing = _compute_gap(frame, aggregated)
    user_payload = _build_user_payload(
        user_query=user_query,
        frame=frame,
        fetched_concepts=fetched,
        missing_concepts=missing,
    )

    try:
        raw = llm.extract(
            system_prompt=system_prompt,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        logger.warning("followup gap-finder LLM failed: %s", e)
        return []

    if not isinstance(raw, dict):
        logger.warning(
            "followup gap-finder returned non-dict: %s",
            type(raw).__name__,
        )
        return []

    try:
        batch = SuggestedFollowupBatch.model_validate(raw)
    except ValidationError as e:
        logger.warning(
            "followup gap-finder schema validation failed: %s", e,
        )
        return []

    # Cap at 4 — prompt says 2-4 but defensive.
    return list(batch.followups[:4])
