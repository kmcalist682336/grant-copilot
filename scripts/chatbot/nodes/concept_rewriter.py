"""ConceptRewriter — first LLM node after extraction.

Turns each ExtractedConcept into:
  (a) a geography-stripped, canonically-phrased ``embedding_text`` the
      semantic router will match against, and
  (b) a typed IntentFacets object (measure_type_sought, disparity_eligible,
      trend_eligible, is_composite, ...) that downstream mechanical
      expansion uses.

One LLM call per query, batched across all concepts. See
``prompts/v1/concept_rewriter.yaml`` for the system prompt.

Public API:
    ConceptRewrite          — per-concept output
    ConceptRewriteBatch     — LLM wire format (list of rewrites)
    rewrite_concepts        — entry point
    ConceptRewriterError    — raised on invalid/unparseable output
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

MeasureType = Literal[
    "count", "median", "percentage", "currency", "ratio", "index",
]


class ConceptRewrite(BaseModel):
    """One concept's rewrite + classification."""

    model_config = ConfigDict(extra="ignore")

    original_text: str = Field(
        ..., description="Verbatim concept text the rewrite came from.",
    )
    embedding_text: str = Field(
        ...,
        description="Geography-free, canonically-phrased query the "
                    "semantic router will embed. 2-6 words typical.",
    )
    measure_type_sought: MeasureType = Field(
        ..., description="What kind of number the user wants.",
    )
    units_expected: Optional[str] = Field(
        default=None,
        description="Free-form ('dollars', 'persons', 'percent', ...).",
    )
    is_total_likely: bool = Field(
        default=False,
        description="True if user is asking for a universe count; hints "
                    "the composer to pick the _001E total.",
    )
    disparity_eligible: bool = Field(
        default=False,
        description="True if the concept has meaningful race/sex/age "
                    "breakdown tables and the user hasn't pinned one. "
                    "Drives J.5 disparity fan-out.",
    )
    trend_eligible: bool = Field(
        default=True,
        description="True if this concept is interesting over time. "
                    "Drives J.6 prior-period fetch.",
    )
    is_composite: bool = Field(
        default=False,
        description="True when the concept can't be answered from one "
                    "variable in one table (food access, gentrification, "
                    "SVI). Frame matcher / template expander take over.",
    )


class ConceptRewriteBatch(BaseModel):
    """Wire format the LLM emits: a list of rewrites in the same order
    as the input concepts."""

    model_config = ConfigDict(extra="ignore")

    rewrites: list[ConceptRewrite] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConceptRewriterError(RuntimeError):
    """Raised when the rewriter LLM call fails or produces invalid output."""


# ---------------------------------------------------------------------------
# Prompt loader (LRU-cached — YAML is derived from code, load once per process)
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "concept_rewriter.yaml"
)


@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    template = load_prompt_template(_PROMPT_PATH)
    schema = ConceptRewriteBatch.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    """Drop the in-process prompt cache. Call if the YAML file is edited."""
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# User-message rendering
# ---------------------------------------------------------------------------

def _build_user_payload(
    concepts: list[ExtractedConcept],
    geo_refs: list[ExtractedGeoRef],
    hints: Optional[dict[str, str]] = None,
) -> str:
    """Render the payload the LLM sees.

    ``geo_refs`` are surfaced so the LLM knows which tokens to strip.
    ``hints`` map original concept text → directive hint from the
    critic on a retry pass; the prompt tells the LLM to follow them.
    """
    hints = hints or {}
    concepts_out = []
    for c in concepts:
        entry = {"text": c.text, "canonical_hint": c.canonical_hint}
        h = hints.get(c.text)
        if h:
            entry["hint"] = h
        concepts_out.append(entry)
    payload = {
        "concepts": concepts_out,
        "geo_refs": [g.text for g in geo_refs],
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_concepts(
    concepts: list[ExtractedConcept],
    geo_refs: list[ExtractedGeoRef],
    llm: LLMClient,
    *,
    hints: Optional[dict[str, str]] = None,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> list[ConceptRewrite]:
    """Rewrite every concept in ``concepts`` in a single batched call.

    Returns a list of ConceptRewrite in the same order as the input.
    If the LLM returns fewer rewrites than concepts (or the order is
    scrambled), we re-align by ``original_text`` match; missing rewrites
    raise ConceptRewriterError.

    ``hints`` is an optional map from concept.text → directive hint
    (produced by the RouterCritic on a retry pass). When present, the
    rewriter follows the hint in re-generating embedding_text.

    Raises:
        ConceptRewriterError: on LLM call failure or invalid output.
    """
    if not concepts:
        return []

    if system_prompt is None:
        system_prompt, schema = _get_prompt_and_schema()
    else:
        schema = ConceptRewriteBatch.model_json_schema()

    user_payload = _build_user_payload(concepts, geo_refs, hints=hints)

    try:
        raw = llm.extract(
            system_prompt=system_prompt,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        raise ConceptRewriterError(f"rewriter LLM call failed: {e}") from e

    if not isinstance(raw, dict):
        raise ConceptRewriterError(
            f"rewriter returned non-dict: {type(raw).__name__}"
        )

    try:
        batch = ConceptRewriteBatch.model_validate(raw)
    except ValidationError as e:
        raise ConceptRewriterError(
            f"rewriter output failed schema validation: {e}"
        ) from e

    return _align_to_input(concepts, batch.rewrites)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _align_to_input(
    concepts: list[ExtractedConcept],
    rewrites: list[ConceptRewrite],
) -> list[ConceptRewrite]:
    """Match rewrites to input concepts by original_text.

    The LLM should return them in order, but a robust alignment by
    text lets us tolerate minor LLM misbehavior (skipped item,
    reordering) without crashing the whole pipeline.
    """
    if len(rewrites) == len(concepts):
        # Trust order; verify with a soft text check, re-align if needed.
        aligned: list[Optional[ConceptRewrite]] = list(rewrites)
        for i, (c, r) in enumerate(zip(concepts, rewrites)):
            if _norm(r.original_text) != _norm(c.text):
                aligned = [None] * len(concepts)
                break
        if all(x is not None for x in aligned):
            return list(aligned)         # type: ignore[list-item]

    by_text = {_norm(r.original_text): r for r in rewrites}
    out: list[ConceptRewrite] = []
    missing: list[str] = []
    for c in concepts:
        hit = by_text.get(_norm(c.text))
        if hit is None:
            missing.append(c.text)
            continue
        out.append(hit)
    if missing:
        raise ConceptRewriterError(
            f"rewriter did not return rewrites for concepts: {missing!r}"
        )
    return out


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())
