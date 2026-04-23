"""RouterCritic — LLM critic for semantic retrieval.

For each rewritten concept and its routed candidates, decides whether
to accept the top hit, retry with a better embedding query, or fall
back to the legacy tier cascade.

This is the safety net that keeps semantic retrieval honest: without
it, spurious top hits (canonical-table-bias misses, profile-vs-
detailed-table mix-ups, measure_type mismatches) become the answer.

One LLM call per query, batched across all concepts. See
``prompts/v1/router_critic.yaml`` for the system prompt.

Public API:
    CriticVerdict           — per-concept decision
    CriticVerdictBatch      — LLM wire format
    critique_routing        — entry point
    RouterCriticError       — raised on invalid/unparseable output

    build_candidate_payload — helper: RoutedResult → LLM-input dict
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.nodes.concept_rewriter import ConceptRewrite
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

CriticDecision = Literal["accept", "retry", "fallback"]


class CriticVerdict(BaseModel):
    """Outcome of critiquing one concept's retrieval result."""

    model_config = ConfigDict(extra="ignore")

    original_text: str = Field(
        ..., description="Verbatim concept text this verdict applies to.",
    )
    decision: CriticDecision = Field(
        ..., description="accept | retry | fallback",
    )
    reason: str = Field(
        ...,
        description="One-sentence explanation; used for "
                    "debugging/observability.",
    )
    rewrite_hint: Optional[str] = Field(
        default=None,
        description="Directive passed to the rewriter on a retry. "
                    "Null when decision != 'retry'.",
    )


class CriticVerdictBatch(BaseModel):
    """LLM wire format — one verdict per input concept, in order."""

    model_config = ConfigDict(extra="ignore")

    verdicts: list[CriticVerdict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RouterCriticError(RuntimeError):
    """Raised when the critic LLM call fails or produces invalid output."""


# Candidate-list score floor below which we automatically short-
# circuit to 'fallback' without an LLM call. aggregate_score is the
# sum of weighted cosines across matching cards; a genuinely-related
# concept typically scores >2.0 (multiple cards hit), a nonsense
# query like "jet skis" scores <1.0 (one weakly-adjacent card).
# Conservative on purpose — only catches the very weak matches.
_WEAK_MATCH_SCORE_FLOOR = 1.0


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "router_critic.yaml"
)


@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    template = load_prompt_template(_PROMPT_PATH)
    schema = CriticVerdictBatch.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Candidate payload construction
# ---------------------------------------------------------------------------

def _year_for(target) -> Optional[int]:
    """Choose a year for label lookups against the v1 variables table.

    Post-Phase-2.5 schema: ``target_year`` is typically None. Falls
    back to the latest year advertised in ``years_available``. Returns
    None when nothing usable is set; label lookup then degrades to
    no-label display rather than a cross-join on NULL.
    """
    if target is None:
        return None
    y = getattr(target, "target_year", None)
    if y is not None:
        try:
            return int(y)
        except (TypeError, ValueError):
            pass
    years = getattr(target, "years_available", None) or []
    try:
        return max(int(x) for x in years) if years else None
    except (TypeError, ValueError):
        return None


def _lookup_variable_labels(
    metadata_db,
    keys: list[tuple[str, str, int, str]],
) -> dict[tuple[str, str, int, str], str]:
    """Bulk-lookup label_natural for a set of variable keys.

    keys: list of (table_id, dataset, year, variable_id). Returns a
    dict from key to label (missing keys are simply absent).
    """
    if metadata_db is None or not keys:
        return {}
    out: dict[tuple[str, str, int, str], str] = {}
    placeholders = ",".join("(?,?,?,?)" for _ in keys)
    flat = [v for k in keys for v in k]
    rows = metadata_db.execute(
        f"SELECT table_id, dataset, year, variable_id, label_natural "
        f"FROM variables "
        f"WHERE (table_id, dataset, year, variable_id) IN "
        f"  (VALUES {placeholders})",
        flat,
    ).fetchall()
    for r in rows:
        if r["label_natural"]:
            out[(r["table_id"], r["dataset"], r["year"],
                 r["variable_id"])] = r["label_natural"]
    return out


def build_candidate_payload(
    rewrite: ConceptRewrite,
    routed: object,                               # RoutedResult
    *,
    metadata_db=None,
    top_k_tables: int = 15,
    top_k_variables: int = 15,
) -> dict:
    """Render one concept's routed candidates as the compact dict the
    LLM sees.

    Includes both top-table and top-variable hits; the critic can look
    at either layer. When ``metadata_db`` is provided, each candidate
    is enriched with the variable's natural-language label so the
    critic can spot semantic mismatches the card text would hide
    (e.g., a SNAP-rate variable that's restricted to one race group).
    """
    # Collect variable keys for bulk label lookup. Post-Phase-2.5
    # targets have target_year=None; use the representative year from
    # years_available so label lookups still hit the v1 variables table.
    # Targets with no known year (neither target_year nor years_available)
    # skip label enrichment — the critic still sees the candidate.
    var_keys: list[tuple[str, str, int, str]] = []
    for t in getattr(routed, "top_tables", [])[:top_k_tables]:
        best = t.best_hit
        yr = _year_for(t)
        if best and best.target_variable_id and yr is not None:
            var_keys.append((
                t.target_table_id, t.target_dataset, yr,
                best.target_variable_id,
            ))
    for v in getattr(routed, "top_variables", [])[:top_k_variables]:
        yr = _year_for(v)
        if v.target_variable_id and yr is not None:
            var_keys.append((
                v.target_table_id, v.target_dataset, yr,
                v.target_variable_id,
            ))
    labels = _lookup_variable_labels(metadata_db, var_keys)

    top_tables = []
    for t in getattr(routed, "top_tables", [])[:top_k_tables]:
        best_hit = t.best_hit
        var_label = None
        yr = _year_for(t)
        if best_hit and best_hit.target_variable_id and yr is not None:
            var_label = labels.get((
                t.target_table_id, t.target_dataset, yr,
                best_hit.target_variable_id,
            ))
        top_tables.append({
            "table_id": t.target_table_id,
            "dataset": t.target_dataset,
            "year": yr,
            "score": round(float(t.aggregate_score), 2),
            "n_positive_hits": t.n_positive_hits,
            "n_negative_hits": t.n_negative_hits,
            "best_variable": (
                best_hit.target_variable_id if best_hit else None
            ),
            "best_variable_label": var_label,
            "top_hit_text": (best_hit.text if best_hit else None),
            "top_hit_card_type": (
                best_hit.card_type if best_hit else None
            ),
            "best_variable_is_total": bool(
                getattr(best_hit, "is_total", 0) or 0
            ) if best_hit else None,
        })

    top_variables = []
    for v in getattr(routed, "top_variables", [])[:top_k_variables]:
        best_hit = v.best_hit
        yr = _year_for(v)
        var_label = None
        if v.target_variable_id and yr is not None:
            var_label = labels.get((
                v.target_table_id, v.target_dataset, yr,
                v.target_variable_id,
            ))
        top_variables.append({
            "variable_id": v.target_variable_id,
            "table_id": v.target_table_id,
            "dataset": v.target_dataset,
            "year": yr,
            "score": round(float(v.aggregate_score), 2),
            "label": var_label,
            "is_total": bool(
                getattr(best_hit, "is_total", 0) or 0
            ) if best_hit else None,
            "top_hit_text": (best_hit.text if best_hit else None),
            "top_hit_card_type": (
                best_hit.card_type if best_hit else None
            ),
        })

    return {
        "original_text": rewrite.original_text,
        "embedding_text": rewrite.embedding_text,
        "measure_type_sought": rewrite.measure_type_sought,
        "is_total_likely": rewrite.is_total_likely,
        "top_tables": top_tables,
        "top_variables": top_variables,
    }


def _build_user_payload(
    rewrites: list[ConceptRewrite],
    routed_results: list[object],             # list[RoutedResult]
    metadata_db=None,
) -> str:
    concepts = [
        build_candidate_payload(rw, rr, metadata_db=metadata_db)
        for rw, rr in zip(rewrites, routed_results)
    ]
    return json.dumps({"concepts": concepts}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def critique_routing(
    rewrites: list[ConceptRewrite],
    routed_results: list[object],             # list[RoutedResult]
    llm: LLMClient,
    *,
    metadata_db=None,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
) -> list[CriticVerdict]:
    """Critique each concept's retrieval result. Returns one verdict per
    rewrite in input order.

    A verdict for a concept with zero candidates short-circuits to
    ``fallback`` without an LLM call — obvious, saves latency.

    Raises:
        RouterCriticError: on LLM call failure or invalid output.
    """
    if len(rewrites) != len(routed_results):
        raise RouterCriticError(
            f"mismatched lengths: {len(rewrites)} rewrites vs "
            f"{len(routed_results)} routed_results"
        )
    if not rewrites:
        return []

    # Short-circuit concepts with empty OR semantically-weak candidate
    # lists → fallback without an LLM call. We check two things:
    #
    #   1. Zero candidates: router returned nothing.
    #   2. Best aggregate_score below a floor: the router returned
    #      SOMETHING but its best match is so weak it's almost
    #      certainly noise (e.g. a user asks "how many jet skis in
    #      Atlanta" and the nearest Census table is B25044 "Tenure by
    #      Vehicles Available" — cosine-adjacent in embedding space
    #      but not actually about jet skis). Forcing fallback here
    #      means the tier cascade is tried next; if the cascade also
    #      has nothing, the concept ends up with no plan calls and
    #      the synthesizer will say "Census doesn't track that"
    #      instead of fabricating an answer from a nearest-neighbor
    #      table.
    #
    # Build a mask so we only send the still-ambiguous ones to the
    # critic LLM.
    verdicts: list[Optional[CriticVerdict]] = [None] * len(rewrites)
    to_critique: list[int] = []
    for i, (rw, rr) in enumerate(zip(rewrites, routed_results)):
        tables = getattr(rr, "top_tables", []) or []
        vars_ = getattr(rr, "top_variables", []) or []
        n_cands = len(tables) + len(vars_)
        if n_cands == 0:
            verdicts[i] = CriticVerdict(
                original_text=rw.original_text,
                decision="fallback",
                reason="no semantic candidates returned",
                rewrite_hint=None,
            )
            continue
        best_score = max(
            [getattr(t, "aggregate_score", 0.0) for t in tables]
            + [getattr(v, "aggregate_score", 0.0) for v in vars_]
            + [0.0]
        )
        if best_score < _WEAK_MATCH_SCORE_FLOOR:
            logger.info(
                "router_critic: forcing fallback for %r — best "
                "aggregate_score=%.2f below floor=%.2f (likely "
                "out-of-domain query)",
                rw.original_text, best_score, _WEAK_MATCH_SCORE_FLOOR,
            )
            verdicts[i] = CriticVerdict(
                original_text=rw.original_text,
                decision="fallback",
                reason=(
                    f"weak semantic match (best score {best_score:.2f} "
                    f"< floor {_WEAK_MATCH_SCORE_FLOOR}) — no Census "
                    f"candidate is a confident fit"
                ),
                rewrite_hint=None,
            )
            continue
        to_critique.append(i)

    if to_critique:
        if system_prompt is None:
            system_prompt, schema = _get_prompt_and_schema()
        else:
            schema = CriticVerdictBatch.model_json_schema()

        sub_rewrites = [rewrites[i] for i in to_critique]
        sub_routed = [routed_results[i] for i in to_critique]

        # The critic LLM occasionally drops concepts when handed 5+
        # at once ("median household income" is the typical casualty
        # — it's so obvious the LLM seems to skip emitting a verdict
        # for it). We don't want to accept that silently, so: make
        # the call; if any concepts are missing, re-call with just
        # the missing sub-rewrites. Only after a retry still can't
        # cover a concept do we default it to 'accept' via the
        # existing _align_to_input fallback.
        resolved: dict[int, CriticVerdict] = {}
        remaining_positions = list(range(len(sub_rewrites)))
        max_passes = 2
        for pass_i in range(max_passes):
            if not remaining_positions:
                break
            pass_rewrites = [sub_rewrites[p] for p in remaining_positions]
            pass_routed = [sub_routed[p] for p in remaining_positions]
            user_payload = _build_user_payload(
                pass_rewrites, pass_routed, metadata_db=metadata_db,
            )
            try:
                raw = llm.extract(
                    system_prompt=system_prompt,
                    user_text=user_payload,
                    schema=schema,
                    temperature=temperature,
                )
            except LLMCallError as e:
                if pass_i == 0:
                    # First-pass failure is fatal — caller falls back
                    # to the tier cascade.
                    raise RouterCriticError(
                        f"critic LLM call failed: {e}"
                    ) from e
                logger.warning(
                    "router_critic: retry-pass LLM call failed (%s); "
                    "will default missing concepts to 'accept'", e,
                )
                break

            if not isinstance(raw, dict):
                raise RouterCriticError(
                    f"critic returned non-dict: {type(raw).__name__}"
                )
            try:
                batch = CriticVerdictBatch.model_validate(raw)
            except ValidationError as e:
                raise RouterCriticError(
                    f"critic output failed schema validation: {e}"
                ) from e

            # Match by normalized original_text and record what we got.
            by_text = {
                _norm(v.original_text): v for v in batch.verdicts
            }
            still_missing: list[int] = []
            for p in remaining_positions:
                hit = by_text.get(_norm(sub_rewrites[p].original_text))
                if hit is not None:
                    resolved[p] = hit
                else:
                    still_missing.append(p)

            if still_missing and pass_i + 1 < max_passes:
                missing_texts = [
                    sub_rewrites[p].original_text for p in still_missing
                ]
                logger.info(
                    "router_critic: %d concept(s) missing after pass "
                    "%d: %r — retrying",
                    len(still_missing), pass_i + 1, missing_texts,
                )
            remaining_positions = still_missing

        # Assemble the final sub-verdict list; anything still not
        # resolved after all passes gets a synthesized 'accept' via
        # the existing helper.
        ordered_sub = [resolved.get(p) for p in range(len(sub_rewrites))]
        if any(v is None for v in ordered_sub):
            # _align_to_input handles the synthesized defaults for us;
            # feed it the sub-rewrites and whatever verdicts we have.
            ordered_sub = _align_to_input(
                sub_rewrites,
                [v for v in ordered_sub if v is not None],
            )
        for idx, v in zip(to_critique, ordered_sub):
            verdicts[idx] = v

    return [v for v in verdicts if v is not None]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _align_to_input(
    rewrites: list[ConceptRewrite],
    verdicts: list[CriticVerdict],
) -> list[CriticVerdict]:
    """Match verdicts back to input order by original_text.

    If the critic LLM drops a concept (returns a short verdicts list
    or renames ``original_text``), we synthesize an ``accept`` verdict
    for each missing concept rather than aborting the whole routing
    pass. The router's retrieval already produced a best-guess match
    for those concepts — keeping them at ``accept`` preserves that
    work, and the downstream tier cascade still gets to correct any
    truly broken picks. A WARNING is logged so gap patterns are
    visible in logs.
    """
    if len(verdicts) == len(rewrites):
        aligned: list[Optional[CriticVerdict]] = list(verdicts)
        for c, v in zip(rewrites, verdicts):
            if _norm(v.original_text) != _norm(c.original_text):
                aligned = [None] * len(rewrites)
                break
        if all(x is not None for x in aligned):
            return list(aligned)             # type: ignore[list-item]

    by_text = {_norm(v.original_text): v for v in verdicts}
    out: list[CriticVerdict] = []
    missing: list[str] = []
    for r in rewrites:
        hit = by_text.get(_norm(r.original_text))
        if hit is None:
            missing.append(r.original_text)
            out.append(CriticVerdict(
                original_text=r.original_text,
                decision="accept",
                reason=(
                    "synthesized default — critic LLM did not return "
                    "a verdict for this concept"
                ),
            ))
        else:
            out.append(hit)
    if missing:
        logger.warning(
            "router_critic: defaulting to 'accept' for %d concept(s) "
            "the critic didn't verdict: %r",
            len(missing), missing,
        )
    return out


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())
