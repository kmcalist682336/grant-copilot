"""GrantFrameMatcher — classify a query into a curated narrative frame.

One LLM call per query (not batched — single FrameMatch output). The
frame registry is injected into the system prompt from
``config/frames.yaml`` so prompt and downstream code can't drift.

The matcher returns a ``FrameMatch`` carrying the chosen frame's name;
callers look up the full ``Frame`` object via ``FrameRegistry.get()``
to access ``required_additional_concepts`` / ``rhetorical_target`` /
``standard_caveats``. Keeping only the name on the wire (a) avoids
sending the full frame dictionary back through the LLM and (b) means
editing ``frames.yaml`` doesn't require re-running the LLM call.

Public API:
    FrameMatch              — wire format
    match_frame             — entry point
    GrantFrameMatcherError  — raised on invalid/unparseable output
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.frames import FrameRegistry, load_default_frames
from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class FrameMatch(BaseModel):
    """Classifier output. Carries only the frame NAME — the caller
    resolves the full Frame object via FrameRegistry."""

    model_config = ConfigDict(extra="ignore")

    frame: str = Field(
        ..., description="Chosen frame name (must exist in "
                         "FrameRegistry).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Classifier confidence, 0..1.",
    )
    rationale: str = Field(
        ..., description="One-sentence why — for observability.",
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GrantFrameMatcherError(RuntimeError):
    """Raised when the matcher LLM call fails or produces invalid output."""


# ---------------------------------------------------------------------------
# Prompt loader (rebuilds on registry change)
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "grant_frame_matcher.yaml"
)


def _render_prompt_with_registry(registry: FrameRegistry) -> str:
    """Load the YAML template and substitute the frame inventory block."""
    template = load_prompt_template(_PROMPT_PATH)
    schema = FrameMatch.model_json_schema()
    rendered = render_system_prompt(template, schema)
    return rendered.replace(
        "{FRAME_INVENTORY}", registry.describe_for_prompt(),
    )


@functools.lru_cache(maxsize=4)
def _get_prompt_and_schema_cached(
    registry_id: int,
) -> tuple[str, dict]:
    # Keyed on id(registry) so tests that mutate a custom registry
    # bust the cache naturally (the registry object changes).
    # `registry` itself is passed through _get_prompt_and_schema below.
    raise RuntimeError("use _get_prompt_and_schema")


def _get_prompt_and_schema(
    registry: FrameRegistry,
) -> tuple[str, dict]:
    """Return (system_prompt, response_schema) for this registry. Not
    LRU-cached on registry identity because FrameRegistry isn't
    hashable by default; the render is cheap enough.
    """
    schema = FrameMatch.model_json_schema()
    return _render_prompt_with_registry(registry), schema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _deterministic_trigger_match(
    user_query: str,
    concepts: list[ExtractedConcept],
    registry: FrameRegistry,
) -> Optional[FrameMatch]:
    """Zero-LLM frame pick via substring match on frame triggers.

    Grant-narrative frames in ``frames.yaml`` carry concrete phrases
    (e.g. 'food pantry', 'rent burden') that are strong signals a
    query belongs to that frame. When the user's text contains at
    least one such phrase for exactly one frame, we can skip the LLM
    entirely — which is both a big latency win and, more importantly,
    a hedge against the frame-matcher LLM call intermittently
    hanging on Vertex (a recurring failure mode we've seen in
    production runs).

    Returns None when 0 frames match or when 2+ frames tie, in which
    case the caller falls through to the LLM classifier.

    Triggers of the form ``"A / B"`` are split into alternatives and
    each tested as its own substring. Triggers containing placeholder
    variables (e.g. ``"what does X look like"``) are skipped here and
    left to the LLM.
    """
    blob = user_query.lower()
    # Include canonical hints + concept text so a query like "food
    # bank grant" (hint=fresh food access) still triggers food_access.
    for c in concepts:
        if c.text:
            blob += "\n" + c.text.lower()
        if c.canonical_hint:
            blob += "\n" + c.canonical_hint.lower()

    import re
    hits: list[tuple[str, float, str]] = []
    for f in registry.frames:
        for trig in (f.triggers or []):
            # Skip placeholder-bearing triggers; leave to the LLM.
            if any(tok in trig for tok in (" X ", " Y ", "{")):
                continue
            for variant in (v.strip() for v in trig.split(" / ")):
                v = variant.lower().strip().strip('"').strip("'")
                if not v:
                    continue
                # Word-boundary match so "ged" doesn't match
                # "chan**ged**" and "esl" doesn't match "v**esl**".
                # Trailing 's?' lets a singular trigger like
                # "veteran" match plurals ("veterans") without
                # duplicating every trigger. Hyphens/spaces inside
                # the trigger are fine — \b only anchors the outer
                # edges of the literal.
                pattern = r"\b" + re.escape(v) + r"s?\b"
                if re.search(pattern, blob):
                    # Longer triggers are more specific → higher
                    # confidence. "food pantry" beats "housing".
                    hits.append((f.name, float(len(v)), v))
                    break               # one hit per frame is enough

    if not hits:
        return None
    hits.sort(key=lambda t: t[1], reverse=True)
    top = hits[0]
    # Tie-break: if multiple frames scored the same longest trigger
    # length, it's actually ambiguous — fall through to the LLM.
    if len(hits) > 1 and hits[1][1] == top[1] and hits[1][0] != top[0]:
        return None
    # Filter out distinct frames that also matched but at a lower
    # score. We already picked the longest-trigger frame.
    chosen = top[0]
    rationale = (
        f"deterministic trigger match on {top[2]!r} "
        f"(skipped LLM classifier)"
    )
    logger.info(
        "grant_frame_matcher: deterministic pick %s via %r",
        chosen, top[2],
    )
    return FrameMatch(frame=chosen, confidence=0.95, rationale=rationale)


def match_frame(
    user_query: str,
    concepts: list[ExtractedConcept],
    geo_refs: list[ExtractedGeoRef],
    llm: LLMClient,
    *,
    registry: Optional[FrameRegistry] = None,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> FrameMatch:
    """Classify one query into a narrative frame.

    Two-tier classifier:
      1. Deterministic trigger match (zero LLM calls) — fires when
         the query text clearly matches exactly one frame via the
         curated trigger list in ``frames.yaml``. This is both fast
         and a hard hedge against the frame-matcher LLM call hanging
         on Vertex.
      2. LLM classifier — runs only when the trigger match is
         ambiguous or empty.

    If the LLM emits a frame name that's not in the registry, we
    fall back to 'other' so downstream code doesn't crash.

    Raises:
        GrantFrameMatcherError on LLM call failure or invalid output.
    """
    reg = registry or load_default_frames()

    # Tier 1: deterministic trigger match. Skip the LLM when the
    # query's phrasing makes the frame obvious.
    pre = _deterministic_trigger_match(user_query, concepts, reg)
    if pre is not None:
        return pre

    if system_prompt is None:
        system_prompt, schema = _get_prompt_and_schema(reg)
    else:
        schema = FrameMatch.model_json_schema()

    payload = {
        "user_query": user_query,
        "concepts": [
            {"text": c.text, "canonical_hint": c.canonical_hint}
            for c in concepts
        ],
        "geo_refs": [g.text for g in geo_refs],
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    try:
        raw = llm.extract(
            system_prompt=system_prompt,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        raise GrantFrameMatcherError(
            f"frame matcher LLM call failed: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise GrantFrameMatcherError(
            f"frame matcher returned non-dict: {type(raw).__name__}"
        )

    try:
        out = FrameMatch.model_validate(raw)
    except ValidationError as e:
        raise GrantFrameMatcherError(
            f"frame matcher output failed schema validation: {e}"
        ) from e

    # Unknown frame name → fall back to 'other'; log for observability.
    if out.frame not in reg.names():
        logger.warning(
            "frame matcher returned unknown frame %r; "
            "coercing to 'other'", out.frame,
        )
        out = FrameMatch(
            frame="other",
            confidence=max(0.0, min(out.confidence, 0.3)),
            rationale=(
                f"coerced from unknown frame {out.frame!r}: {out.rationale}"
            ),
        )

    return out
