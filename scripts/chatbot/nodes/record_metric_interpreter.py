"""RecordMetricInterpreter: normalize record-level metric recipes.

This node is intentionally narrow: it does not generate SQL and it does not
choose geographies.  It receives the extractor's structured intent plus a
selected dataset's definitions and card candidates, then returns the same ExtractedIntent shape
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
from scripts.chatbot.record_dataset import get_record_dataset

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "record_metric_interpreter.yaml"
)


class RecordMetricInterpreterError(RuntimeError):
    """Raised when the record metric interpreter cannot return valid JSON."""


def _terms_from_intent(intent: ExtractedIntent) -> list[str]:
    terms: list[str] = []
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
    *, dataset: str,
) -> list[dict[str, Any]]:
    if semantic_router is None:
        return []
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    adapter = get_record_dataset(dataset)
    searches = 0
    for term in _terms_from_intent(intent):
        if adapter.variable_alias([term]) or adapter.recipes().lookup(term):
            continue
        if searches >= 3:
            break
        searches += 1
        try:
            routed = semantic_router.route_dataset(
                term, target_dataset=dataset, top_k=5,
            )
        except Exception:
            continue
        for target in routed.top_variables[:5]:
            if target.target_dataset != dataset or target.target_table_id != adapter.table_id:
                continue
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
    dataset: str,
    frame: Optional[Frame] = None,
    semantic_router: Optional[object] = None,
    temperature: float = 0.1,
) -> tuple[ExtractedIntent, list[str]]:
    """Return an intent with normalized record analyses.

    Check population restrictions against the explicitly selected dataset,
    including when a curated frame already supplied a metric recipe.
    """
    if not intent.analyses:
        return intent, []
    adapter = get_record_dataset(dataset)

    system_prompt, schema = _system_prompt()
    payload = {
        "user_query": query,
        "current_intent": intent.model_dump(mode="json"),
        "dataset_definition": adapter.interpretation_context(query, intent),
        "semantic_candidate_cards": _candidate_cards_for_intent(
            intent, semantic_router, dataset=dataset,
        ),
        "instructions": [
            "Return the full ExtractedIntent JSON shape.",
            "Preserve the named metric so its curated recipe supplies the numerator and denominator.",
            "Check every population restriction in the question, even if some filters already exist.",
            "Filters must come from the user question or be required by the metric wording; do not invent unrelated filters.",
            "Use the selected dataset's variable aliases and stored category values. Never borrow another dataset's fields.",
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
        normalized = ExtractedIntent.model_validate(raw)
        if not normalized.analyses:
            raise ValueError("Interpreter returned no record analyses")
        for analysis in normalized.analyses:
            parts = [analysis.measure, *analysis.groupings, *[f.dimension for f in analysis.filters]]
            if any(p is not None and p.dataset_hint not in {dataset, "unknown", "both"} for p in parts):
                raise ValueError("Interpreter returned a different dataset")
            for part in parts:
                if part is not None:
                    part.dataset_hint = dataset
            if analysis.measure is None:
                raise ValueError("Interpreter returned an analysis without a measure")
            if adapter.metric_recipe(analysis, adapter.recipes()) is None and adapter.variable_alias(
                [analysis.measure.canonical_hint, analysis.measure.text]
            ) is None:
                raise ValueError(f"Unrecognized measure: {analysis.measure.text}")
            for grouping in analysis.groupings:
                if adapter.variable_alias([grouping.canonical_hint, grouping.text]) is None:
                    raise ValueError(f"Unrecognized grouping: {grouping.text}")
            for f in analysis.filters:
                variable = adapter.variable_alias([f.dimension.canonical_hint, f.dimension.text])
                if variable is None:
                    raise ValueError(f"Unrecognized filter dimension: {f.dimension.text}")
                if f.operator not in {"is_null", "is_not_null"}:
                    from scripts.chatbot.record_values import _decoded_value
                    adapter.filter_value(variable, _decoded_value(f))
        # Interpretation owns analyses, never geography, years, or context.
        return intent.model_copy(update={"analyses": normalized.analyses}), [
            f"record metric interpreter normalized {dataset} analyses",
        ]
    except Exception as exc:
        raise RecordMetricInterpreterError(str(exc)) from exc
