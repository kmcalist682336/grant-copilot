"""Phase I: LLM-side decomposition of composite Census concepts.

The planner emits ``needs_llm_decomposition`` whenever a concept is
flagged composite AND the Tier 0.5 cache misses AND no Tier 1 entry
matches. This module is the LLM step that fills that gap: it takes the
composite text, asks the model for ≤ 5 measurable proxy sub-concepts
(biased toward terms in the curated concept_map), validates the output,
caches it via ``DecompositionCache`` (so future identical queries skip
the LLM call), and returns the ``Decomposition`` for the orchestrator
to feed back into ``plan_query``.

System-prompt design
--------------------
The system prompt is short, stable across calls (so Vertex prompt
caching keeps the unit cost low), and carries the full list of
canonical concept_map names verbatim. The model is told to prefer
those exact spellings whenever applicable so the resulting sub-concepts
land on Tier 1 hits when re-planned.

Public API:
    decompose_concept(text, llm, cmap, *, cache=None, model_version=...)
        → Decomposition
    DecompositionLLMError
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.decomposition_cache import (
    Decomposition, DecompositionCache, MAX_SUB_CONCEPTS,
)
from scripts.chatbot.llm_client import LLMCallError, LLMClient

logger = logging.getLogger(__name__)


class DecompositionLLMError(RuntimeError):
    """Raised when the LLM fails to produce a valid Decomposition."""


# ---------------------------------------------------------------------------
# LLM output schema (subset of Decomposition — concept + model_version are
# filled in by the caller, not emitted by the LLM).
# ---------------------------------------------------------------------------

class _DecompositionLLMOutput(BaseModel):
    """The fields we expect the LLM to return."""

    model_config = ConfigDict(extra="ignore")

    sub_concepts: list[str] = Field(
        ...,
        description=(
            "Measurable proxy concepts that, taken together, approximate "
            f"the composite. Capped at {MAX_SUB_CONCEPTS}. Prefer terms "
            "from the curated list provided in the system prompt so the "
            "planner gets Tier 1 hits."
        ),
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Self-rated confidence in the decomposition.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="One-sentence explanation of why these proxies were chosen.",
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You decompose composite US Census concepts into 1-{max_sub} measurable
proxy concepts that, taken together, approximate the composite.

Output STRICT JSON conforming to the schema below. No prose, no
explanations outside JSON.

PRINCIPLES
----------
- Each sub-concept must be a measurable Census-data dimension, not an
  abstract idea. ("median home value" — yes; "neighborhood character"
  — no.)
- Prefer canonical terms from the CURATED LIST below verbatim — those
  hit our fast-path lookup and avoid further LLM calls.
- Use at most {max_sub} sub-concepts. Quality over quantity.
- Skip sub-concepts that don't materially differentiate places at the
  geographies the user asks about.
- Set `confidence` low (~0.3-0.5) when the composite is ambiguous or
  the curated list lacks good matches.

WORKED EXAMPLES
---------------
"gentrification" → ["median home value", "median gross rent",
                    "racial composition", "bachelor's degree or higher",
                    "median household income"]
  rationale: "Standard housing-cost + demographic-shift + education
              proxy set used in displacement-risk literature."
  confidence: 0.85

"fresh food access" → ["snap participation", "households without a vehicle",
                       "poverty rate", "internet access"]
  rationale: "ACS doesn't measure grocery proximity directly; these
              proxies correlate with food-desert designation."
  confidence: 0.55

"economic vitality" → ["median household income", "unemployment rate",
                       "labor force participation", "self-employment income"]
  rationale: "Standard income + employment composite."
  confidence: 0.75

CURATED LIST (prefer these verbatim)
------------------------------------
{curated_list}
"""


def _build_system_prompt(cmap: ConceptMap) -> str:
    """Render the stable system prompt with the concept_map's canonical
    names interpolated."""
    canonicals = sorted({entry.canonical for entry in cmap})
    curated = "\n".join(f"  - {c}" for c in canonicals)
    return _SYSTEM_PROMPT_TEMPLATE.format(
        max_sub=MAX_SUB_CONCEPTS, curated_list=curated,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_concept(
    concept: str,
    llm: LLMClient,
    cmap: ConceptMap,
    *,
    cache: Optional[DecompositionCache] = None,
    model_version: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> Decomposition:
    """Get a Decomposition for ``concept``, hitting the cache if present.

    Args:
        concept: composite concept text (e.g., "gentrification").
        llm: LLMClient for the decomposition call (when cache misses).
        cmap: concept_map — used to bias sub-concept names toward Tier 1.
        cache: optional DecompositionCache; when provided, gets+sets it.
        model_version: identifier stored alongside the cached entry so
            we can invalidate later.
        temperature: sampling temperature for the LLM call.

    Raises:
        DecompositionLLMError: when the LLM call fails or returns an
            unparseable / invalid object.
    """
    if cache is not None:
        hit = cache.get(concept)
        if hit is not None:
            logger.debug("decomp cache hit for %r", concept[:60])
            return hit

    system_prompt = _build_system_prompt(cmap)
    schema = _DecompositionLLMOutput.model_json_schema()

    try:
        raw = llm.extract(
            system_prompt=system_prompt,
            user_text=concept,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        raise DecompositionLLMError(
            f"LLM decomposition call failed for {concept!r}: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise DecompositionLLMError(
            f"LLM returned non-dict for decomposition of {concept!r}: "
            f"{type(raw).__name__}"
        )

    try:
        out = _DecompositionLLMOutput.model_validate(raw)
    except ValidationError as e:
        raise DecompositionLLMError(
            f"LLM output failed validation for {concept!r}: {e}"
        ) from e

    try:
        decomp = Decomposition(
            concept=concept,
            sub_concepts=out.sub_concepts,
            confidence=out.confidence,
            rationale=out.rationale,
            model_version=model_version,
        )
    except ValidationError as e:
        # The Decomposition validator caps sub_concepts at MAX_SUB_CONCEPTS
        # and dedupes — this only triggers if the LLM emits 0 or > N.
        raise DecompositionLLMError(
            f"Decomposition for {concept!r} did not satisfy invariants: {e}"
        ) from e

    if cache is not None:
        cache.set(decomp)
    return decomp
