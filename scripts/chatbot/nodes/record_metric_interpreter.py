"""RecordMetricInterpreter: normalize record-level metric recipes.

This node is intentionally narrow: it does not generate SQL and it does not
choose geographies.  It receives the extractor's structured intent plus a
small set of HMDA card candidates, then returns the same ExtractedIntent shape
with clearer record-level analyses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from scripts.chatbot.frames import Frame
from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import ExtractedIntent
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)
from scripts.chatbot.record_metric_map import load_default_record_metric_map

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "record_metric_interpreter.yaml"
)


class RecordMetricInterpreterError(RuntimeError):
    """Raised when the record metric interpreter cannot return valid JSON."""


def _terms_from_intent(intent: ExtractedIntent) -> list[str]:
    terms: list[str] = []
    for concept in intent.concepts:
        terms.extend([concept.canonical_hint or "", concept.text or ""])
    for analysis in intent.analyses:
        if analysis.measure is not None:
            terms.extend([
                analysis.measure.canonical_hint or "",
                analysis.measure.text or "",
            ])
        for filter_item in analysis.filters:
            terms.extend([
                filter_item.dimension.canonical_hint or "",
                filter_item.dimension.text or "",
                filter_item.value_text or "",
                filter_item.normalized_value_hint or "",
            ])
        for grouping in analysis.groupings:
            terms.extend([grouping.canonical_hint or "", grouping.text or ""])
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = " ".join(str(term).strip().lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(term).strip())
    return out[:12]


def _candidate_cards_for_intent(
    intent: ExtractedIntent,
    semantic_router: Optional[object],
) -> list[dict[str, Any]]:
    if semantic_router is None:
        return []
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for term in _terms_from_intent(intent):
        try:
            routed = semantic_router.route_dataset(
                term, target_dataset="hmda", top_k=5,
            )
        except Exception:
            continue
        for target in routed.top_variables[:5]:
            variable_id = target.target_variable_id
            if not variable_id:
                continue
            key = (term.lower(), variable_id)
            if key in seen:
                continue
            seen.add(key)
            cards.append({
                "query_text": term,
                "target_variable_id": variable_id,
                "target_table_id": target.target_table_id,
                "score": float(target.aggregate_score or 0.0),
                "matched_card": target.best_hit.text if target.best_hit else None,
            })
    return cards[:25]


def _system_prompt() -> tuple[str, dict[str, Any]]:
    schema = ExtractedIntent.model_json_schema()
    template = load_prompt_template(_PROMPT_PATH)
    return render_system_prompt(template, schema), schema


def interpret_record_metrics(
    query: str,
    intent: ExtractedIntent,
    llm: LLMClient,
    *,
    frame: Optional[Frame] = None,
    semantic_router: Optional[object] = None,
    temperature: float = 0.1,
) -> tuple[ExtractedIntent, list[str]]:
    """Return an intent with normalized record analyses.

    Curated frames may already provide record analyses; for those we avoid an
    extra LLM pass.  Direct HMDA questions use this node to turn phrases like
    "denial rate for Black women" into measure/action filters/groupings.
    """
    if not intent.analyses:
        return intent, []
    if frame is not None and frame.required_record_analyses:
        return intent, ["record analyses supplied by matched grant frame"]

    system_prompt, schema = _system_prompt()
    payload = {
        "user_query": query,
        "current_intent": intent.model_dump(mode="json"),
        "semantic_candidate_cards": _candidate_cards_for_intent(
            intent, semantic_router,
        ),
        "curated_record_metric_recipes": (
            load_default_record_metric_map().summaries()
        ),
        "instructions": [
            "Return the full ExtractedIntent JSON shape.",
            "Preserve Census concepts unless they are clearly HMDA concepts.",
            "Prefer curated_record_metric_recipes for HMDA metrics such as denial rate, approval rate, origination rate, average income, and loan amount.",
            "For HMDA rates, use operation='percentage' with an action/status filter on the measure variable.",
            "Filters must come from the user question or be required by the metric wording; do not invent unrelated filters.",
            "Use decoded labels when known, such as 'Application denied', 'Loan originated', 'Female', 'Black or African American', 'White'.",
        ],
    }
    try:
        raw = llm.extract(
            system_prompt=system_prompt,
            user_text=json.dumps(payload, ensure_ascii=False, indent=2),
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as exc:
        raise RecordMetricInterpreterError(str(exc)) from exc
    try:
        return ExtractedIntent.model_validate(raw), [
            "record metric interpreter normalized HMDA analyses",
        ]
    except Exception as exc:
        raise RecordMetricInterpreterError(str(exc)) from exc
