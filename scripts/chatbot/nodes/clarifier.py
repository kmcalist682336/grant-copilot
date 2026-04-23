"""Clarifier — ask the user to disambiguate ambiguous queries.

Three independent scoped clarifiers run between geo-resolve and
frame-match:

  * **geo**     — ambiguous geography ("Buckhead" as a whole vs
                  North Buckhead vs Buckhead Village; "Decatur" GA
                  vs Decatur IL; etc.)
  * **concept** — vague concepts ("help me", "poverty" without
                  child/senior/deep qualifier, "housing" without
                  ownership/cost-burden split)
  * **other**   — anything outside geo/concept scope: missing
                  temporal window, unclear comparators, etc.

Each scope runs up to ``max_cycles_per_scope`` times. Counters are
independent; exhausting one scope's cycles doesn't consume another's.

Every clarification emits up to 3 labeled options (``(a)``/``(b)``/
``(c)``) plus an always-available freeform fallback. The orchestrator
calls the injected ``ask_user`` callback with the rendered question,
parses the response, and applies the chosen disambiguation to the
intent before moving on.

Public API:
    ClarificationOption
    ClarificationResult
    clarify_scope(scope, ...)
    parse_clarification_response(raw, options)
    render_clarification(result)
    apply_clarification(intent, scope, option, freeform_extra)
"""
from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

ClarifierScope = Literal["geo", "concept", "other"]


# Accepted ExtractedGeoRef.ref_type values. Kept local to avoid a
# circular import on models.RefType and to document the coercion
# policy in one place.
_VALID_REF_TYPES: frozenset[str] = frozenset({
    "administrative", "neighborhood", "corridor", "region",
    "relative", "landmark_area", "national",
})

# LLMs on the clarifier/reviewer prompts sometimes invent a ref_type
# string ("city", "metro", "county") that isn't one of the enum
# values. Map the common synonyms to the closest valid value; drop
# anything else to None so the caller keeps the original ref's type.
_REF_TYPE_SYNONYMS: dict[str, str] = {
    "city": "administrative",
    "cdp": "administrative",
    "place": "administrative",
    "municipality": "administrative",
    "county": "administrative",
    "state": "administrative",
    "country": "national",
    "nation": "national",
    "msa": "administrative",
    "metro": "administrative",
    "metropolitan": "administrative",
    "cbsa": "administrative",
    "zipcode": "administrative",
    "zip": "administrative",
    "zcta": "administrative",
    "tract": "administrative",
    "neighborhood_area": "neighborhood",
    "nbh": "neighborhood",
    "district": "neighborhood",
    "quadrant": "region",
    "side": "region",
    "area": "region",
    "zone": "region",
    "directional": "region",
    "landmark": "landmark_area",
    "radius": "relative",
    "near": "relative",
    "around": "relative",
}


def _coerce_ref_type(raw):
    """Return a valid RefType literal or None.

    Accepts either None, an already-valid value, or a loose LLM
    phrase like 'city' / 'metro' / 'msa'. Anything we can't map
    returns None so the caller keeps whatever ref_type was already
    on the geo — never raises, never crashes the pipeline.
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s in _VALID_REF_TYPES:
        return s
    if s in _REF_TYPE_SYNONYMS:
        mapped = _REF_TYPE_SYNONYMS[s]
        logger.info(
            "clarifier: coerced ref_type %r → %r", raw, mapped,
        )
        return mapped
    logger.warning(
        "clarifier: dropping unknown ref_type %r — keeping "
        "original geo's type", raw,
    )
    return None


class ClarificationOption(BaseModel):
    """One labeled option the user can pick.

    Scope-specific patches tell the orchestrator what to update on
    the intent when the user selects this option.
    """

    model_config = ConfigDict(extra="ignore")

    label: str = Field(
        ...,
        description="Short user-facing description of the option, "
                    "e.g. 'Buckhead as a whole area (33 tracts)'.",
    )
    # Geo-scope patch: replaces the first geo_ref in intent.
    new_geo_text: Optional[str] = Field(
        default=None,
        description="If set, replaces the first geo_ref's text, "
                    "triggering re-resolution.",
    )
    new_geo_ref_type: Optional[str] = Field(
        default=None,
        description="If set, replaces the first geo_ref's ref_type.",
    )
    # Concept-scope patch: replaces the entire concepts list.
    new_concepts: list[str] = Field(
        default_factory=list,
        description="If non-empty, replaces intent.concepts entirely.",
    )
    # Other-scope patch: appended to the query text so downstream
    # stages (planner, synthesizer) see the added constraint.
    query_addendum: Optional[str] = Field(
        default=None,
        description="If set, appended to the query for downstream "
                    "stages (temporal window, data source, etc.).",
    )


class ClarificationResult(BaseModel):
    """LLM output for one clarification scope."""

    model_config = ConfigDict(extra="ignore")

    needs_clarification: bool = Field(
        ...,
        description="False when this scope is clear as-is; True when a "
                    "question is required.",
    )
    scope: ClarifierScope = Field(...)
    question: str = Field(
        default="",
        description="User-facing question; empty when "
                    "needs_clarification is False.",
    )
    options: list[ClarificationOption] = Field(
        default_factory=list,
        description="Up to 3 labeled options. May be empty when the "
                    "question is open-ended.",
    )
    rationale: str = Field(
        default="",
        description="Why the clarifier decided to ask (debug-only).",
    )


class ClarifierError(RuntimeError):
    """LLM call failed or returned unparseable output."""


# ---------------------------------------------------------------------------
# Prompt loading — one prompt per scope
# ---------------------------------------------------------------------------

_PROMPTS_DIR = (
    Path(__file__).resolve().parents[3] / "prompts" / "v1"
)

_SCOPE_PROMPT_FILE = {
    "geo":     "clarifier_geo.yaml",
    "concept": "clarifier_concept.yaml",
    "other":   "clarifier_other.yaml",
}


@functools.lru_cache(maxsize=3)
def _get_prompt_and_schema(scope: ClarifierScope) -> tuple[str, dict]:
    template = load_prompt_template(
        _PROMPTS_DIR / _SCOPE_PROMPT_FILE[scope]
    )
    schema = ClarificationResult.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_OPTION_LABELS = ("a", "b", "c")


def render_clarification(result: ClarificationResult) -> str:
    """Render a ClarificationResult as the plain-text prompt the user
    sees. Always includes the freeform-response footer."""
    lines: list[str] = [result.question.rstrip()]
    lines.append("")
    for i, opt in enumerate(result.options[:3]):
        letter = _OPTION_LABELS[i]
        lines.append(f"  ({letter}) {opt.label}")
    lines.append("")
    lines.append(
        "Pick a letter, or type your own response — any text beyond "
        "a simple (a)/(b)/(c) will be passed through to the pipeline "
        "as additional context."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Matches a leading option letter in common forms:
#   "a", "A", "(a)", "a.", "a)", "A:" — case-insensitive.
_OPTION_HEAD_RE = re.compile(
    r"^\s*\(?([abcABC])[\)\.:\-]?\s*",
)


def parse_clarification_response(
    raw: str, options: list[ClarificationOption],
) -> tuple[Optional[ClarificationOption], str]:
    """Parse a raw user response into (chosen_option, freeform_extra).

    Behavior:
      - Response like "a" / "(a)" / "A." → (options[0], "")
      - "b — but only for pre-2020" → (options[1], "but only for pre-2020")
      - Pure freeform with no leading letter → (None, raw_stripped)
      - Letter out of range (d+ or letter past len(options)) → (None, raw)
    """
    if not raw:
        return None, ""
    text = raw.strip()
    m = _OPTION_HEAD_RE.match(text)
    if not m:
        return None, text
    letter = m.group(1).lower()
    try:
        idx = _OPTION_LABELS.index(letter)
    except ValueError:
        return None, text
    if idx >= len(options):
        return None, text
    # Any text after the matched option label becomes freeform extra.
    remaining = text[m.end():].strip()
    return options[idx], remaining


# ---------------------------------------------------------------------------
# State patching
# ---------------------------------------------------------------------------

def apply_clarification(
    *,
    scope: ClarifierScope,
    intent: ExtractedIntent,
    option: Optional[ClarificationOption],
    freeform_extra: str,
    query: str,
) -> tuple[ExtractedIntent, str]:
    """Return (updated_intent, updated_query) after applying the user's
    clarification choice. Mutates nothing in place — callers should
    replace their intent/query with the return values.

    When ``option`` is None (pure freeform), the freeform text is
    routed to the scope's default slot:
      - geo:     replaces geo_refs[0].text (triggers re-resolution)
      - concept: replaces intent.concepts with a single concept whose
                 text is the freeform string
      - other:   appended to query as context
    """
    # Copy intent so we don't mutate the caller's reference.
    updated_intent = intent.model_copy(deep=True)
    updated_query = query

    if scope == "geo":
        new_text = (
            option.new_geo_text if option and option.new_geo_text
            else freeform_extra or None
        )
        new_type = _coerce_ref_type(
            option.new_geo_ref_type if option else None,
        )
        if new_text:
            if updated_intent.geo_refs:
                # Patch the primary geo_ref; leave qualifier/scope_hint
                # alone unless the LLM specified a new ref_type.
                ref = updated_intent.geo_refs[0]
                effective_type = new_type if new_type else ref.ref_type
                updated_intent.geo_refs[0] = ExtractedGeoRef(
                    text=new_text, ref_type=effective_type,
                    qualifier=ref.qualifier,
                    proximity_tier=ref.proximity_tier,
                    scope_hint=ref.scope_hint,
                )
            else:
                # No prior geo_refs — seed one from the clarification.
                updated_intent.geo_refs = [
                    ExtractedGeoRef(
                        text=new_text,
                        ref_type=new_type or "administrative",
                    )
                ]
        if option and freeform_extra:
            # Any extra freeform context tacks onto the query.
            updated_query = f"{query}\n\n(clarification: {freeform_extra})"

    elif scope == "concept":
        concept_texts = []
        if option and option.new_concepts:
            concept_texts.extend(option.new_concepts)
        elif freeform_extra:
            concept_texts.append(freeform_extra)
        if concept_texts:
            updated_intent.concepts = [
                ExtractedConcept(text=t) for t in concept_texts
            ]
        if option and freeform_extra:
            updated_query = f"{query}\n\n(clarification: {freeform_extra})"

    elif scope == "other":
        addendum = None
        if option and option.query_addendum:
            addendum = option.query_addendum
        if freeform_extra:
            addendum = (
                f"{addendum}; {freeform_extra}" if addendum
                else freeform_extra
            )
        if addendum:
            updated_query = f"{query}\n\n(clarification: {addendum})"

    return updated_intent, updated_query


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _geo_summary(resolved_geos: list[ResolvedGeography]) -> list[dict]:
    """Compact geo summary for the LLM. We pass the TOP resolved geo
    plus its confidence so the clarifier can judge whether to ask."""
    out = []
    for g in resolved_geos[:3]:
        out.append({
            "display_name": g.display_name,
            "geo_level": g.geo_level,
            "confidence": round(g.confidence, 2),
            "tract_count": len(g.tract_geoids or []),
            "notes": list(g.assumption_notes or [])[:2],
        })
    return out


def _user_payload(
    *,
    scope: ClarifierScope,
    query: str,
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
) -> str:
    """The JSON payload the clarifier LLM sees."""
    payload = {
        "scope": scope,
        "user_query": query,
        "extracted_geo_refs": [
            {
                "text": g.text, "ref_type": g.ref_type,
                "qualifier": g.qualifier,
            }
            for g in intent.geo_refs
        ],
        "resolved_geos": _geo_summary(resolved_geos),
        "extracted_concepts": [
            {
                "text": c.text,
                "canonical_hint": c.canonical_hint,
                "is_composite": c.is_composite,
            }
            for c in intent.concepts
        ],
        "temporal_intent": intent.temporal_intent,
        "explicit_years": intent.years,
        "comparison_implied": intent.comparison_implied,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


# Hard wall-clock ceiling on a single clarifier LLM call. If the
# underlying SDK hangs inside its own retry loop (we've seen 3+
# minutes on a single geo-clarifier call), we bail with a
# no-clarification result rather than freezing the whole pipeline.
# Clarification is an enrichment — never worth stalling the answer.
# Outer timeout above the LLM client's own 60s per-call cap. The
# clarifier's LLM step is the only blocking work, so 75s matches
# the plan_reviewer budget and cleanly wraps the 60s inner cap.
_CLARIFIER_HARD_TIMEOUT_S = 75.0


# Observable-ambiguity gate — decides WHICH clarifier scopes (if any)
# to actually call. Focuses on the two signals that gate answer
# correctness: (1) did we hit the right geography? (2) are we going
# to hit the right Census variables?
#
# Trend windows, comparator choice, data-source vintage, and other
# "preference" ambiguities are better served by sensible defaults than
# by asking the user every time. The clarifier's 'other' scope stays
# available for explicit opt-in but is off the default path.


def scopes_to_run(
    *,
    query: str,
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    min_top_confidence: float = 0.90,
    min_confidence_gap: float = 0.10,
) -> tuple[list[ClarifierScope], str]:
    """Return ``(scopes, reason)`` — the subset of {"geo", "concept"}
    that merits a clarification pass for this query, and a one-line
    reason string for logs.

    ``scopes`` is empty when the query's geography and concept
    resolutions are both confident enough to proceed silently.
    """
    fire: list[ClarifierScope] = []
    reasons: list[str] = []

    # --- Geography signal ---------------------------------------------
    if not resolved_geos:
        fire.append("geo")
        reasons.append("no resolved geos")
    else:
        top = resolved_geos[0]
        if top.confidence < min_top_confidence:
            fire.append("geo")
            reasons.append(
                f"geo conf {top.confidence:.2f} < {min_top_confidence}"
            )
        elif len(resolved_geos) > 1 and (
            top.confidence - resolved_geos[1].confidence
        ) < min_confidence_gap:
            fire.append("geo")
            reasons.append(
                f"geo tie {top.confidence:.2f}/"
                f"{resolved_geos[1].confidence:.2f}"
            )
        elif len(intent.geo_refs) > 1:
            fire.append("geo")
            reasons.append(f"{len(intent.geo_refs)} geo refs")

    # --- Concept signal (will we hit the right variable?) -------------
    if not intent.concepts:
        fire.append("concept")
        reasons.append("no concepts")
    elif not any(c.canonical_hint for c in intent.concepts):
        # No concept is pinned to the curated map. The router will
        # still do semantic search, but variable selection gets
        # shakier and worth asking the user about.
        fire.append("concept")
        reasons.append("no concept has canonical_hint")

    if not fire:
        top_name = (
            resolved_geos[0].display_name if resolved_geos
            else "<none>"
        )
        return [], f"high-confidence path: geo={top_name}"
    return fire, "; ".join(reasons)


# Backwards-compatible wrapper: some early callers/tests use the
# boolean form. Maps the new scopes list to the old (run, reason)
# contract.
def should_run_clarifiers(
    *,
    query: str,
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    min_top_confidence: float = 0.90,
) -> tuple[bool, str]:
    scopes, reason = scopes_to_run(
        query=query, intent=intent, resolved_geos=resolved_geos,
        min_top_confidence=min_top_confidence,
    )
    return bool(scopes), reason


def clarify_scope(
    *,
    scope: ClarifierScope,
    query: str,
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    llm: LLMClient,
    temperature: float = 0.2,
    hard_timeout_s: float = _CLARIFIER_HARD_TIMEOUT_S,
) -> ClarificationResult:
    """Run one scope's clarifier. Returns a ClarificationResult with
    ``needs_clarification=False`` when the scope is clear enough to
    skip, or a populated result with a question when disambiguation
    is needed.

    Errors (LLM failure, schema mismatch, hard-timeout) degrade to a
    no-clarification result rather than raising — clarification is an
    enrichment, not a pipeline blocker.
    """
    try:
        system_prompt, schema = _get_prompt_and_schema(scope)
    except FileNotFoundError as e:
        logger.warning(
            "clarifier prompt for scope=%s missing: %s", scope, e,
        )
        return ClarificationResult(
            needs_clarification=False, scope=scope,
        )
    user_payload = _user_payload(
        scope=scope, query=query, intent=intent,
        resolved_geos=resolved_geos,
    )

    def _invoke():
        return llm.extract(
            system_prompt=system_prompt,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )

    # Wrap the LLM call on a daemon thread and time it out by join().
    # Using `with ThreadPoolExecutor()` is NOT correct here — the
    # context manager calls shutdown(wait=True) on exit, which blocks
    # on the still-running worker when Vertex is hung, so the caller-
    # side timeout would be purely cosmetic. daemon=True means the
    # orphaned worker won't block process shutdown either.
    import threading as _threading
    _holder: dict = {"raw": None, "error": None}

    def _run():
        try:
            _holder["raw"] = _invoke()
        except BaseException as e:                     # noqa: BLE001
            _holder["error"] = e

    _worker = _threading.Thread(
        target=_run, name=f"clarifier-{scope}", daemon=True,
    )
    _worker.start()
    _worker.join(timeout=hard_timeout_s)
    if _worker.is_alive():
        logger.warning(
            "clarifier LLM call timed out after %.1fs "
            "(scope=%s) — proceeding without clarification "
            "(worker thread left running)",
            hard_timeout_s, scope,
        )
        return ClarificationResult(
            needs_clarification=False, scope=scope,
        )
    try:
        if _holder["error"] is not None:
            raise _holder["error"]
        raw = _holder["raw"]
    except LLMCallError as e:
        logger.warning("clarifier LLM call failed (scope=%s): %s", scope, e)
        return ClarificationResult(
            needs_clarification=False, scope=scope,
        )
    if not isinstance(raw, dict):
        logger.warning(
            "clarifier returned non-dict for scope=%s: %s",
            scope, type(raw).__name__,
        )
        return ClarificationResult(
            needs_clarification=False, scope=scope,
        )
    # Force the scope field — the LLM should echo it but we don't trust it.
    raw["scope"] = scope
    try:
        result = ClarificationResult.model_validate(raw)
    except ValidationError as e:
        logger.warning(
            "clarifier output schema mismatch (scope=%s): %s", scope, e,
        )
        return ClarificationResult(
            needs_clarification=False, scope=scope,
        )
    return result
