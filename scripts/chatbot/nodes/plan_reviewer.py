"""PlanReviewer — one LLM call after planning to judge whether the
plan actually answers the query, and to surface 0-3 clarifying
questions when it doesn't.

Why post-plan instead of pre-plan:
  * Vector searches + planning are cheap (no LLM). Running them first
    gives the reviewer the FULL context — exact tables, variables,
    geo-levels, comparator geos, year selections — rather than just
    the user's intent.
  * One LLM hop instead of three separate scope passes.
  * The reviewer can catch correctness bugs ("user said 'veterans'
    but plan pulls civilian-pop-18+ _001E") that a pre-plan gate
    can't see.
  * When the plan is already correct, the reviewer returns high
    confidence and no questions — zero-cost skip on the happy path.

Public API:
    PlanReviewClarification — one question the reviewer wants to ask
    PlanReviewResult        — LLM output (confidence + 0-3 clarifications)
    review_plan(...)        — entry point
    PlanReviewerError

The reviewer reuses ``clarifier.ClarificationOption`` for the options
on each question and ``apply_clarification`` for patching intent after
user response.
"""
from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import ExtractedIntent, ResolvedGeography
from scripts.chatbot.nodes.clarifier import (
    ClarificationOption, ClarifierScope,
)
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class PlanReviewClarification(BaseModel):
    """One clarifying question the reviewer wants to ask."""

    model_config = ConfigDict(extra="ignore")

    scope: ClarifierScope = Field(
        ...,
        description="Which part of intent this question's answer "
                    "would patch: 'geo' / 'concept' / 'other'.",
    )
    question: str = Field(
        ...,
        description="User-facing question text, one sentence.",
    )
    options: list[ClarificationOption] = Field(
        default_factory=list,
        description="Up to 3 labeled options. Empty = open-ended.",
    )


class PlanReviewResult(BaseModel):
    """Reviewer's verdict on the plan."""

    model_config = ConfigDict(extra="ignore")

    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="How confident the reviewer is that the plan "
                    "answers the user's query (0=very unsure, "
                    "1=highly confident).",
    )
    reasoning: str = Field(
        default="",
        description="One-sentence rationale for the confidence score.",
    )
    clarifications: list[PlanReviewClarification] = Field(
        default_factory=list,
        description="Up to 3 questions to ask when confidence is low. "
                    "Empty list = proceed without clarification.",
    )


class PlanReviewerError(RuntimeError):
    """LLM call failed or returned unparseable output."""


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "plan_reviewer.yaml"
)


@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    template = load_prompt_template(_PROMPT_PATH)
    schema = PlanReviewResult.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Payload assembly — summarize query + plan for the reviewer
# ---------------------------------------------------------------------------

def _geo_summary(resolved_geos: list[ResolvedGeography]) -> list[dict]:
    out = []
    for g in resolved_geos[:3]:
        out.append({
            "display_name": g.display_name,
            "geo_level": g.geo_level,
            "confidence": round(float(g.confidence), 2),
            "tract_count": len(g.tract_geoids or []),
        })
    return out


def _plan_summary(plan) -> dict:
    """Compact summary of a PlanResult for the reviewer.

    We show:
      - Each concept_resolution: concept text, tier, chosen table
      - For each planned API call: table_id, dataset, variables,
        geo_level, year, role
    """
    out: dict = {
        "concept_resolutions": [],
        "calls": [],
    }
    for cr in getattr(plan, "concept_resolutions", []) or []:
        chosen_table = None
        if getattr(cr, "concept_entry", None) is not None:
            chosen_table = {
                "table_id": cr.concept_entry.table_id,
                "dataset": cr.concept_entry.dataset,
                "source": "tier_1_concept_map",
            }
        elif getattr(cr, "table_candidate", None) is not None:
            chosen_table = {
                "table_id": cr.table_candidate.table_id,
                "dataset": cr.table_candidate.dataset,
                "source": "tier_2_fts",
            }
        elif getattr(cr, "routed_result", None) is not None:
            rr = cr.routed_result
            if rr.top_tables:
                chosen_table = {
                    "table_id": rr.top_tables[0].target_table_id,
                    "dataset": rr.top_tables[0].target_dataset,
                    "source": "tier_semantic",
                }
        out["concept_resolutions"].append({
            "concept_text": cr.concept.text,
            "canonical_hint": cr.concept.canonical_hint,
            "tier": cr.tier,
            "chosen_table": chosen_table,
        })

    for c in (getattr(plan, "calls", []) or [])[:24]:
        out["calls"].append({
            "role": c.role,
            "table_id": c.api_call.table_id,
            "dataset": c.api_call.dataset,
            "variables": list(c.api_call.variables),
            "geo_level": c.api_call.geo_level,
            "year": c.year,
        })
    return out


def _user_payload(
    *, query: str, intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography], plan,
) -> str:
    payload = {
        "user_query": query,
        "extracted_concepts": [
            {
                "text": c.text,
                "canonical_hint": c.canonical_hint,
            }
            for c in intent.concepts
        ],
        "extracted_geo_refs": [
            {"text": g.text, "ref_type": g.ref_type}
            for g in intent.geo_refs
        ],
        "temporal_intent": intent.temporal_intent,
        "explicit_years": intent.years,
        "comparison_implied": intent.comparison_implied,
        "resolved_geos": _geo_summary(resolved_geos),
        "plan": _plan_summary(plan),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Outer timeout on the reviewer. Acts as a belt-and-suspenders safety
# net above the LLM client's own 60s per-call timeout — the reviewer
# can in principle do work before and after the single LLM call, and
# we want one bounded number to stand behind the whole stage. 75s
# leaves the 60s LLM budget room without blowing past user patience.
_REVIEWER_HARD_TIMEOUT_S = 75.0


def review_plan(
    *,
    query: str,
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    plan,
    llm: LLMClient,
    temperature: float = 0.2,
    hard_timeout_s: float = _REVIEWER_HARD_TIMEOUT_S,
) -> PlanReviewResult:
    """Score the plan against the query. Returns a PlanReviewResult
    with confidence + up to 3 clarifications.

    Failures (LLM error, timeout, schema mismatch) degrade to a
    high-confidence (no-clarification) result so the pipeline never
    stalls on a misbehaving reviewer.
    """
    try:
        system_prompt, schema = _get_prompt_and_schema()
    except FileNotFoundError as e:
        logger.warning("plan_reviewer prompt missing: %s", e)
        return PlanReviewResult(
            confidence=1.0,
            reasoning="prompt file missing; skipping review",
        )
    user_payload = _user_payload(
        query=query, intent=intent,
        resolved_geos=resolved_geos, plan=plan,
    )

    def _invoke():
        return llm.extract(
            system_prompt=system_prompt,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )

    import threading as _threading
    holder: dict = {"raw": None, "error": None}

    def _run():
        try:
            holder["raw"] = _invoke()
        except BaseException as e:                     # noqa: BLE001
            holder["error"] = e

    worker = _threading.Thread(
        target=_run, name="plan-reviewer", daemon=True,
    )
    worker.start()
    worker.join(timeout=hard_timeout_s)
    if worker.is_alive():
        logger.warning(
            "plan_reviewer timed out after %.1fs — "
            "proceeding without review (worker thread left running)",
            hard_timeout_s,
        )
        return PlanReviewResult(
            confidence=1.0,
            reasoning=f"reviewer timed out after {hard_timeout_s}s",
        )
    try:
        if holder["error"] is not None:
            raise holder["error"]
        raw = holder["raw"]
    except LLMCallError as e:
        logger.warning("plan_reviewer LLM call failed: %s", e)
        return PlanReviewResult(
            confidence=1.0, reasoning=f"reviewer LLM failed: {e}",
        )
    if not isinstance(raw, dict):
        logger.warning(
            "plan_reviewer non-dict output: %s",
            type(raw).__name__,
        )
        return PlanReviewResult(
            confidence=1.0, reasoning="reviewer returned non-dict",
        )
    try:
        result = PlanReviewResult.model_validate(raw)
    except ValidationError as e:
        logger.warning("plan_reviewer schema mismatch: %s", e)
        return PlanReviewResult(
            confidence=1.0,
            reasoning="reviewer output failed schema validation",
        )
    # Clamp clarifications to 3 — the prompt says 0-3 but don't trust it.
    if len(result.clarifications) > 3:
        result = result.model_copy(update={
            "clarifications": result.clarifications[:3],
        })
    return result
