"""Orchestrator: full chatbot pipeline glue.

Runs one user query end-to-end. Two modes, picked by whether a
``semantic_router`` is passed in:

Agent-routing mode (Phase 1+; preferred):
  1. extract_intent                 (LLM)
  2. resolve_intent                 (gazetteer)
  3. route_concepts_with_critic     (LLM rewriter + LLM critic + router)
                                    → list[ConceptResolution]
  4. plan_query(concept_resolutions=...)
                                    → API plan from the resolutions
  5. CensusCaller.fetch_all         (HTTP, parallel)
  6. aggregate_results              (sum/ratio/components, tract filter)
  7. synthesize_answer              (LLM)
  8. wrap into QueryResponse

Legacy mode (no router):
  1. extract_intent
  2. resolve_intent
  3. plan_query                     (Tier cascade)
  4. decompose_concept              (LLM, only when plan emits
                                    'needs_llm_decomposition')
                                    → cache → re-plan
  5-8. as above.

Exposes a single async entry point ``answer_query``.

Public API:
    QueryResponse
    answer_query(query, llm, db, metadata_db, cmap, *, decomp_cache,
                 api_cache, api_key, config, semantic_router=None)
                                    → QueryResponse
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

ProgressCb = Callable[[str, Optional[str]], None]


def _progress(
    cb: Optional[ProgressCb], stage: str, detail: Optional[str] = None,
) -> None:
    """Invoke a progress callback, swallowing any errors. Progress
    reporting must never break the pipeline."""
    if cb is None:
        return
    try:
        cb(stage, detail)
    except Exception:                                  # pragma: no cover
        pass

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.agent_routing import (
    RoutingOutcome, route_concepts_with_critic,
)
from scripts.chatbot.derived_data_dump import dump_derived_data
from scripts.chatbot.aggregator import (
    AggregatedResult, aggregate_results,
)
from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import CensusCaller, FetchResult
from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.decomposer import (
    DecompositionLLMError, decompose_concept,
)
from scripts.chatbot.decomposition_cache import DecompositionCache
from scripts.chatbot.extractor import ExtractorError, extract_intent
from scripts.chatbot.frames import (
    Frame, FrameRegistry, load_default_frames,
)
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.llm_client import LLMClient, estimate_cost_usd
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.nodes.anomaly_detector import (
    AnomalyFlag, detect_anomalies,
)
from scripts.chatbot.nodes.comparators import comparator_geos
from scripts.chatbot.nodes.followup_gap_finder import (
    SuggestedFollowup, find_followups,
)
from scripts.chatbot.nodes.grant_frame_matcher import (
    FrameMatch, GrantFrameMatcherError, match_frame,
)
from scripts.chatbot.nodes.clarifier import (
    apply_clarification, parse_clarification_response,
    render_clarification,
)
from scripts.chatbot.nodes.plan_reviewer import (
    PlanReviewResult, review_plan,
)
from scripts.chatbot.nodes.peer_context import (
    PeerContext, get_peer_contexts,
)
from scripts.chatbot.nodes.peer_retriever import PeerRetriever
from scripts.chatbot.nodes.magnitude_contextualizer import (
    MagnitudeFraming, contextualize_magnitudes,
)
from scripts.chatbot.nodes.trend import prior_period_calls
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, data_level_for, plan_query,
)
from scripts.chatbot.synthesizer import (
    SynthesisError, SynthesizedAnswer, synthesize_answer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class StageMetrics(BaseModel):
    """Per-stage timing + token counts for one query turn."""

    model_config = ConfigDict(extra="ignore")

    extract_s: float = 0.0
    resolve_s: float = 0.0
    clarify_s: float = 0.0
    plan_s: float = 0.0
    # Fine-grained plan-stage breakdown — attribution for investigating
    # slow plan_s. Sum of these three ≈ plan_s (a small residual is
    # dispatch + Python overhead). agent_routing is tracked as
    # decompose_s since it's a separate LLM-heavy stage.
    plan_query_s: float = 0.0
    plan_trend_s: float = 0.0
    plan_decomp_retry_s: float = 0.0
    decompose_s: float = 0.0
    fetch_s: float = 0.0
    aggregate_s: float = 0.0
    synthesize_s: float = 0.0
    total_s: float = 0.0

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0

    census_calls_total: int = 0
    census_cache_hits: int = 0
    census_failures: int = 0


class QueryResponse(BaseModel):
    """End-to-end response for one user query."""

    model_config = ConfigDict(
        extra="ignore", arbitrary_types_allowed=True,
    )

    query: str
    intent: ExtractedIntent
    resolved_geos: list[ResolvedGeography]
    plan: PlanResult
    aggregated: Optional[AggregatedResult] = None
    answer: Optional[SynthesizedAnswer] = None
    frame_match: Optional[FrameMatch] = None

    # Phase 4 realism agents — None when aggregated is empty.
    magnitude_framings: list[MagnitudeFraming] = Field(default_factory=list)
    anomaly_flags: list[AnomalyFlag] = Field(default_factory=list)
    followups: list[SuggestedFollowup] = Field(default_factory=list)
    # Phase 3: grant-narrative peer context. Empty when the primary
    # geo is a type PeerRetriever can't anchor (neighborhoods, tracts,
    # state, us) or when the PeerRetriever isn't wired in.
    peer_contexts: list["PeerContext"] = Field(default_factory=list)

    error: Optional[str] = None
    metrics: StageMetrics = Field(default_factory=StageMetrics)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _llm_usage_snapshot(llm: LLMClient) -> dict[str, int]:
    """Snapshot the LLM client's running usage counters, or zeros."""
    if not hasattr(llm, "get_usage_counters"):
        return {"calls": 0, "input_tokens": 0,
                "cached_tokens": 0, "output_tokens": 0}
    raw = llm.get_usage_counters()
    return {
        "calls": int(raw.get("calls", 0) or 0),
        "input_tokens": int(raw.get("input_tokens", 0) or 0),
        "cached_tokens": int(raw.get("cached_tokens", 0) or 0),
        "output_tokens": int(raw.get("output_tokens", 0) or 0),
    }


def _accumulate_metrics(
    metrics: StageMetrics, llm: LLMClient,
) -> None:
    """Roll the LLM's running counters into the metrics record."""
    snap = _llm_usage_snapshot(llm)
    metrics.llm_calls = snap["calls"]
    metrics.llm_input_tokens = snap["input_tokens"]
    metrics.llm_cached_tokens = snap["cached_tokens"]
    metrics.llm_output_tokens = snap["output_tokens"]
    metrics.llm_cost_usd = estimate_cost_usd(snap)


def _write_response_txt(
    folder: Path, query: str,
    answer: Optional["SynthesizedAnswer"], err: Optional[str],
    peer_contexts: Optional[list] = None,
) -> None:
    """Drop a human-readable ``response.txt`` into the dump folder.

    Mirrors what the REPL shows: query, prose, key findings, caveats,
    sources with variable labels, AND peer comparisons. Self-
    contained so a reader can open one file and understand the whole
    answer without digging into CSVs."""
    lines: list[str] = []
    lines.append(f"QUERY\n{'=' * 60}\n{query.rstrip()}\n")
    if err:
        lines.append(f"ERROR\n{'=' * 60}\n{err}\n")
    if answer is None:
        if not err:
            lines.append("ANSWER\n" + "=" * 60 + "\n(no synthesized answer)\n")
    else:
        lines.append("ANSWER\n" + "=" * 60 + "\n" + (answer.prose or "") + "\n")
        if answer.key_findings:
            lines.append("KEY FINDINGS\n" + "=" * 60)
            for kf in answer.key_findings:
                lines.append(f"  - {kf}")
            lines.append("")
        if answer.caveats:
            lines.append("CAVEATS\n" + "=" * 60)
            for cv in answer.caveats:
                lines.append(f"  - {cv}")
            lines.append("")
        if answer.citations:
            lines.append("SOURCES\n" + "=" * 60)
            for c in answer.citations:
                title = f"  - {c.table_title}" if c.table_title else ""
                lines.append(
                    f"  [{c.dataset}/{c.year}/{c.table_id}]{title}"
                )
                for cv in (c.variables or []):
                    lbl = cv.label if cv.label else "(no label)"
                    lines.append(f"      · {cv.variable_id}  {lbl}")
            lines.append("")
    # Peer comparisons block — explicitly listed with per-peer
    # explanations so the reader sees WHY each peer geography was
    # chosen (proximity + population + matched features).
    if peer_contexts:
        lines.append("PEER COMPARISONS\n" + "=" * 60)
        lines.append(
            "  Different geographic units that might be useful "
            "comparison points:"
        )
        for ctx in peer_contexts:
            axis_label = (
                getattr(ctx, "axis_description", "") or
                getattr(ctx, "axis", "")
            )
            scope = getattr(ctx, "pool_scope", "") or ""
            scope_tail = f" ({scope})" if scope else ""
            lines.append(f"\n  {axis_label}{scope_tail}")
            peers = getattr(ctx, "peers", []) or []
            if not peers:
                lines.append("      (no peers found)")
                continue
            for p in peers:
                rank = getattr(p, "rank", 0)
                name = getattr(p, "geo_name", "")
                expl = getattr(p, "match_explanation", "") or ""
                expl_tail = f" — {expl}" if expl else ""
                lines.append(f"      {rank:>2}. {name}{expl_tail}")
        lines.append("")
    (folder / "response.txt").write_text("\n".join(lines))


async def _run_fetches(
    plan: PlanResult,
    api_cache: APICache,
    api_key: Optional[str],
    *,
    max_concurrent: int = 20,
) -> list[FetchResult]:
    """Issue every PlannedCall in parallel via the shared CensusCaller."""
    if not plan.calls:
        return []
    plans = [c.api_call for c in plan.calls]
    async with CensusCaller(
        api_key=api_key, cache=api_cache,
        max_concurrent=max_concurrent,
    ) as caller:
        return await caller.fetch_all(plans)


def _decompose_pending(
    plan: PlanResult,
    llm: LLMClient,
    cmap: ConceptMap,
    decomp_cache: DecompositionCache,
) -> int:
    """For every concept resolution flagged ``needs_llm_decomposition``,
    call the LLM decomposer and write the result through the cache.

    Returns the number of concepts decomposed (so the caller can decide
    whether to re-plan).
    """
    decomposed = 0
    for cr in plan.concept_resolutions:
        if cr.tier != "needs_llm_decomposition":
            continue
        text = cr.concept.canonical_hint or cr.concept.text
        try:
            decompose_concept(
                text, llm, cmap, cache=decomp_cache,
            )
            decomposed += 1
        except DecompositionLLMError as e:
            logger.warning(
                "decomposition failed for %r: %s — concept will remain "
                "unresolved", text[:80], e,
            )
    return decomposed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def answer_query(
    query: str,
    llm: LLMClient,
    db,
    metadata_db,
    cmap: Optional[ConceptMap],
    *,
    decomp_cache: DecompositionCache,
    api_cache: APICache,
    api_key: Optional[str],
    config: dict,
    fetch_max_concurrent: int = 20,
    max_decomposition_attempts: int = 1,
    semantic_router: Optional[object] = None,    # SemanticRouter
    frame_registry: Optional[FrameRegistry] = None,
    max_comparators: int = 4,
    trend_lookback_years: int = 3,
    peer_retriever: Optional[PeerRetriever] = None,
    universe_picker: Optional[object] = None,
    ask_user: Optional[Any] = None,
    progress_cb: Optional[ProgressCb] = None,
) -> QueryResponse:
    """End-to-end pipeline for one user query.

    Errors are caught at stage boundaries — a partial response (with
    ``error`` set on QueryResponse) is returned rather than raising,
    so the REPL/UI can render a graceful failure.
    """
    metrics = StageMetrics()
    if hasattr(llm, "reset_usage_counters"):
        llm.reset_usage_counters()
    t_start = time.time()

    # Capture every log record from this query into a string buffer
    # so we can write per-query logs.txt alongside the CSVs in the
    # dump folder. Attached to the root scripts.chatbot logger so
    # any descendant module lands in the capture.
    import io as _io
    import logging as _logging
    _log_buffer = _io.StringIO()
    _log_handler = _logging.StreamHandler(_log_buffer)
    _log_handler.setLevel(_logging.INFO)
    _log_handler.setFormatter(_logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    _root_logger = _logging.getLogger("scripts.chatbot")
    _root_logger.addHandler(_log_handler)

    # Derived-data folder is committed at query-start so the stamp
    # reflects when the user asked, not when we finished writing.
    from datetime import datetime
    dd_cfg = config.get("derived_data", {}) or {}
    dd_folder: Optional[Path] = None
    if dd_cfg.get("enabled", True):
        dd_base = Path(dd_cfg.get("base_dir", "derived_data"))
        # Stamp down to microseconds — seconds alone wasn't fine-
        # grained enough when testing fired multiple cached queries
        # back-to-back and produced folder collisions. %f is 6-digit
        # microseconds, so a million distinct stamps per second.
        dd_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = dd_base / f"data_{dd_stamp}"
        n = 0
        while candidate.exists():
            n += 1
            candidate = dd_base / f"data_{dd_stamp}_{n}"
        dd_folder = candidate

    # 0. Scope gate --------------------------------------------------
    # One LLM call (usually short-circuited deterministically) that
    # decides whether the query is answerable from Census data at
    # all. Out-of-scope queries like "how many jet skis are there in
    # Northeast Atlanta" return early with a friendly refusal
    # instead of fabricating an answer from the nearest-neighbor
    # Census table. Disable via ``scope_gate.enabled: false`` in
    # chatbot.yaml. Fails open on LLM errors (proceeds with the
    # query) so a broken gate never blocks legitimate work.
    sg_cfg = config.get("scope_gate", {}) or {}
    if sg_cfg.get("enabled", True):
        from scripts.chatbot.nodes.scope_gate import (
            is_in_scope, user_message_for_refusal,
        )
        _progress(progress_cb, "Checking query scope")
        try:
            sv = is_in_scope(
                query, llm,
                temperature=float(sg_cfg.get("temperature", 0.0)),
                hard_timeout_s=float(sg_cfg.get("hard_timeout_s", 20.0)),
            )
        except Exception as e:                         # pragma: no cover
            logger.warning(
                "scope_gate unexpected error (%s); failing open", e,
            )
            sv = None
        if sv is not None and not sv.answerable:
            msg = user_message_for_refusal(sv)
            logger.info("scope_gate: rejected query — %s", sv.reason)
            _progress(progress_cb, "Query out of Census scope")
            _accumulate_metrics(metrics, llm)
            metrics.total_s = time.time() - t_start
            return QueryResponse(
                query=query,
                intent=ExtractedIntent(geo_refs=[]),
                resolved_geos=[],
                plan=PlanResult(
                    intent=ExtractedIntent(geo_refs=[]),
                    resolved_geos=[], concept_resolutions=[], calls=[],
                ),
                answer=SynthesizedAnswer(
                    prose=msg, key_findings=[], caveats=[], citations=[],
                ),
                error=None, metrics=metrics,
            )

    # 1. Extraction --------------------------------------------------
    _progress(progress_cb, "Extracting intent from query")
    try:
        t0 = time.time()
        intent = extract_intent(
            query, llm,
            temperature=config.get("vertex_ai", {}).get(
                "temperature", 0.1,
            ),
        )
        metrics.extract_s = time.time() - t0
    except ExtractorError as e:
        metrics.total_s = time.time() - t_start
        _accumulate_metrics(metrics, llm)
        return QueryResponse(
            query=query,
            intent=ExtractedIntent(geo_refs=[]),
            resolved_geos=[], plan=PlanResult(
                intent=ExtractedIntent(geo_refs=[]),
                resolved_geos=[], concept_resolutions=[], calls=[],
            ),
            error=f"extraction failed: {e}", metrics=metrics,
        )

    # 1a. Strip 'difference' (exclusion) compounds --------------------
    # "Atlanta excluding Buckhead" comes in as a `difference` compound
    # over two geo_refs. The planner/composer don't yet produce plan
    # calls for exclusion shapes, so rather than silently return a
    # zero-calls answer we drop the compound, run the query against
    # the unrestricted primary geography, and surface a caveat to
    # the user. Exclusion is on the roadmap.
    exclusion_caveat: Optional[str] = None
    if intent.compounds and any(
        getattr(c, "op", None) == "difference" for c in intent.compounds
    ):
        dropped: list[str] = []
        kept_compounds = []
        for c in intent.compounds:
            if getattr(c, "op", None) == "difference":
                p_idx = getattr(c, "primary_ref_idx", 0)
                f_idx = getattr(c, "filter_ref_idx", 0)
                primary_txt = (
                    intent.geo_refs[p_idx].text
                    if 0 <= p_idx < len(intent.geo_refs) else "?"
                )
                filter_txt = (
                    intent.geo_refs[f_idx].text
                    if 0 <= f_idx < len(intent.geo_refs) else "?"
                )
                dropped.append(f"{primary_txt!r} excluding {filter_txt!r}")
            else:
                kept_compounds.append(c)
        intent = intent.model_copy(update={"compounds": kept_compounds})
        exclusion_caveat = (
            "Geographic exclusions are on the roadmap but not yet "
            "supported — I treated the query as the whole primary "
            "area without subtracting: " + "; ".join(dropped) + ". "
            "The numbers below therefore include the excluded sub-"
            "area."
        )
        _progress(
            progress_cb,
            "Exclusion not supported — defaulting to whole area",
            "; ".join(dropped),
        )
        logger.info(
            "dropped %d difference compound(s): %s",
            len(dropped), dropped,
        )

    # 2. Geographic resolution ---------------------------------------
    _progress(
        progress_cb, "Resolving geographies",
        f"{len(intent.geo_refs)} reference(s)",
    )
    t0 = time.time()
    resolved = resolve_intent(intent, db)
    metrics.resolve_s = time.time() - t0
    if resolved:
        _progress(
            progress_cb, "Geographies resolved",
            ", ".join(
                f"{g.display_name} ({g.geo_level})" for g in resolved[:3]
            ),
        )

    # NOTE: Clarification now runs AFTER the plan stage (see section 5b
    # below). This is cheaper and better-informed — the reviewer sees
    # the actual plan, not just the intent.

    # 3. Frame match (Phase 2) ---------------------------------------
    # Classify the query into a curated grant-narrative frame. The
    # frame's required_additional_concepts get appended to the intent
    # before routing, so they go through the same rewriter+critic loop.
    # Skipped in legacy (no router) mode and when no concepts were
    # extracted (pure-context queries).
    frame_match: Optional[FrameMatch] = None
    frame: Optional[Frame] = None
    intent_for_routing = intent
    if semantic_router is not None and intent.concepts:
        _progress(progress_cb, "Matching grant-narrative frame")
        registry = frame_registry or load_default_frames()
        t0 = time.time()
        try:
            frame_match = match_frame(
                query, intent.concepts, intent.geo_refs, llm,
                registry=registry,
                temperature=config.get("vertex_ai", {}).get(
                    "temperature", 0.1,
                ),
            )
            frame = registry.get(frame_match.frame)
            if frame_match:
                _progress(
                    progress_cb, "Frame matched",
                    f"{frame_match.frame} (conf={frame_match.confidence:.2f})",
                )
        except GrantFrameMatcherError as e:
            logger.warning(
                "frame matcher failed (%s); proceeding without frame", e,
            )
        metrics.decompose_s += time.time() - t0

        # Augment intent.concepts with frame-required additions, deduped
        # against existing canonical names.
        if frame and frame.required_additional_concepts:
            existing = {
                ((c.canonical_hint or c.text) or "").strip().lower()
                for c in intent.concepts
            }
            new_concepts: list[ExtractedConcept] = []
            for canon in frame.required_additional_concepts:
                key = canon.strip().lower()
                if key in existing:
                    continue
                existing.add(key)
                new_concepts.append(ExtractedConcept(
                    text=canon, canonical_hint=canon,
                ))
            if new_concepts:
                intent_for_routing = intent.model_copy(update={
                    "concepts": list(intent.concepts) + new_concepts,
                })

    # 4. Agent routing (rewrite → route → critique) -------------------
    # When a SemanticRouter is available, run the Phase 1 agent chain
    # BEFORE the planner so concepts are resolved via the
    # rewriter+critic loop. The planner then just builds API calls
    # from the resolved list. Tier cascade survives as a critic-
    # triggered fallback inside route_concepts_with_critic.
    pre_resolutions: Optional[list[ConceptResolution]] = None
    routing_outcome: Optional[RoutingOutcome] = None
    if semantic_router is not None and intent_for_routing.concepts:
        _progress(
            progress_cb, "Routing concepts to Census tables",
            f"{len(intent_for_routing.concepts)} concept(s)",
        )
        t0 = time.time()
        data_levels = sorted({data_level_for(g) for g in resolved}) \
            if resolved else []
        try:
            routing_outcome = route_concepts_with_critic(
                intent_for_routing.concepts, intent_for_routing.geo_refs,
                data_levels, semantic_router, llm,
                cmap=cmap, metadata_db=metadata_db,
                decomp_cache=decomp_cache,
                temperature=config.get("vertex_ai", {}).get(
                    "temperature", 0.1,
                ),
            )
            pre_resolutions = routing_outcome.resolutions
        except Exception as e:                     # pragma: no cover
            logger.warning(
                "agent routing failed (%s); planner will fall through "
                "to internal tier cascade", e,
            )
            pre_resolutions = None
        metrics.decompose_s += time.time() - t0    # routing time

    # 4b. Comparator extras (J.1) — for any sub-state primary geo,
    #     synthesize the county→MSA→state→US chain and pass to plan.
    extra_geos: list[tuple[ResolvedGeography, str]] = []
    if semantic_router is not None and resolved and max_comparators > 0:
        seen_keys: set[tuple[str, str]] = set()
        for g in resolved:
            for cg, role in comparator_geos(
                g, db, max_comparators=max_comparators,
            ):
                key = (cg.geo_level, cg.geo_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                extra_geos.append((cg, role))

    # 5. Plan --------------------------------------------------------
    _progress(progress_cb, "Planning Census API calls")
    t0 = time.time()
    plan = plan_query(
        intent_for_routing, resolved, cmap, metadata_db,
        decomp_cache=decomp_cache,
        semantic_router=semantic_router,
        concept_resolutions=pre_resolutions,
        extra_geos=extra_geos or None,
        universe_picker=universe_picker,
    )
    _progress(
        progress_cb, "Plan ready",
        f"{len(plan.calls)} call(s) queued",
    )
    metrics.plan_query_s = time.time() - t0
    metrics.plan_s = metrics.plan_query_s

    # 5a. Plan reviewer — one LLM call to judge whether the plan
    # actually answers the query. When confidence is low, the
    # reviewer emits 1-3 clarifying questions keyed on the real plan
    # content (tables/variables/geo-level chosen). We ask the user,
    # patch intent, and re-run planning once.
    clar_cfg = config.get("clarification", {}) or {}
    clarification_enabled = clar_cfg.get("enabled", True)
    max_review_cycles = int(clar_cfg.get("max_review_cycles",
                                          clar_cfg.get("max_cycles_per_scope", 1)))
    confidence_threshold = float(clar_cfg.get("min_confidence", 0.85))
    if clarification_enabled and max_review_cycles > 0:
        _progress(progress_cb, "Reviewing plan for correctness")
        t0 = time.time()
        cycles_used = 0
        while cycles_used < max_review_cycles:
            try:
                review = review_plan(
                    query=query, intent=intent,
                    resolved_geos=resolved, plan=plan, llm=llm,
                    temperature=config.get("vertex_ai", {}).get(
                        "temperature", 0.2,
                    ),
                )
            except Exception as e:                     # pragma: no cover
                logger.warning("plan_reviewer failed: %s", e)
                break
            logger.info(
                "plan_reviewer: confidence=%.2f clarifications=%d reasoning=%s",
                review.confidence, len(review.clarifications),
                (review.reasoning or "")[:160],
            )
            _progress(
                progress_cb, "Plan review complete",
                f"confidence={review.confidence:.2f}, "
                f"{len(review.clarifications)} clarification(s)",
            )
            if review.confidence >= confidence_threshold or \
                    not review.clarifications:
                break
            # Auto mode: no user to ask, so auto-picking (a) is just
            # guessing. The original plan — even if the reviewer thinks
            # it's imperfect — is strictly better than risking a
            # busted re-resolve (e.g. LLM-suggested geo name doesn't
            # exist in the gazetteer). Surface the review confidence
            # in the log and proceed with the original plan.
            if ask_user is None:
                logger.info(
                    "plan_reviewer: confidence=%.2f below threshold %.2f "
                    "but no ask_user wired; proceeding with original plan "
                    "(skipping %d auto clarifications)",
                    review.confidence, confidence_threshold,
                    len(review.clarifications),
                )
                break
            # Snapshot pre-clarification state so we can revert if the
            # patched intent re-resolves to nothing.
            resolved_before = resolved
            plan_before = plan
            intent_before = intent
            query_before = query
            _progress(progress_cb, "Asking for clarification")
            # Ask each clarification in order; patch intent in place.
            for clar in review.clarifications[:3]:
                rendered = render_clarification(clar)
                try:
                    raw_response = ask_user(rendered)
                except Exception as e:                 # pragma: no cover
                    logger.warning("ask_user failed: %s", e)
                    raw_response = "a"
                option, freeform_extra = parse_clarification_response(
                    raw_response, clar.options,
                )
                intent, query = apply_clarification(
                    scope=clar.scope, intent=intent, option=option,
                    freeform_extra=freeform_extra, query=query,
                )
            _progress(progress_cb, "Re-planning with clarification")
            # Re-resolve + re-plan once with the patched intent. We
            # explicitly DON'T pass pre_resolutions here — those were
            # computed from the original concepts and are stale after
            # a concept-scope clarification. Re-running concept
            # resolution via the semantic router (deterministic +
            # cheap) is correct; that adds no LLM cost.
            try:
                resolved_new = resolve_intent(intent, db)
                if not resolved_new:
                    # The patched geo doesn't resolve (LLM suggested a
                    # name not in the gazetteer, e.g. "Buckhead area"
                    # when the catalog key is "Buckhead"). Revert
                    # rather than proceed with an empty plan.
                    logger.warning(
                        "post-clarify re-resolve returned no geos "
                        "(patched text=%r); reverting to original plan",
                        intent.geo_refs[0].text if intent.geo_refs else None,
                    )
                    resolved = resolved_before
                    plan = plan_before
                    intent = intent_before
                    query = query_before
                else:
                    plan_new = plan_query(
                        intent, resolved_new, cmap, metadata_db,
                        decomp_cache=decomp_cache,
                        semantic_router=semantic_router,
                        concept_resolutions=None,
                        extra_geos=extra_geos or None,
                        universe_picker=universe_picker,
                    )
                    if not plan_new.calls:
                        logger.warning(
                            "post-clarify re-plan produced 0 calls; "
                            "reverting to original plan",
                        )
                        resolved = resolved_before
                        plan = plan_before
                        intent = intent_before
                        query = query_before
                    else:
                        resolved = resolved_new
                        plan = plan_new
            except Exception as e:                     # pragma: no cover
                logger.warning(
                    "post-clarify re-resolve/re-plan failed (%s); "
                    "reverting to original plan", e,
                )
                resolved = resolved_before
                plan = plan_before
                intent = intent_before
                query = query_before
            cycles_used += 1
        metrics.clarify_s = time.time() - t0

    # 5b. Trend prior_period (J.6) — fan out (latest-N) calls for
    #     trend-eligible concepts (by IntentFacets from rewrites).
    if (semantic_router is not None and routing_outcome
            and routing_outcome.rewrites and trend_lookback_years > 0):
        _progress(progress_cb, "Expanding trend calls (prior periods)")
        t0 = time.time()
        try:
            trend_extras = prior_period_calls(
                plan, routing_outcome.rewrites,
                intent_for_routing, metadata_db,
                lookback_years=trend_lookback_years,
            )
            plan.calls.extend(trend_extras)
        except Exception as e:                     # pragma: no cover
            logger.warning(
                "trend expansion failed (%s); skipping prior_period", e,
            )
        metrics.plan_trend_s = time.time() - t0
        metrics.plan_s += metrics.plan_trend_s

    # 5. LLM decomposition for any pending composites ----------------
    # Skipped when the semantic router is in play: composites surface
    # as multi-table semantic hits naturally; no decomposition needed.
    if semantic_router is None:
        attempt = 0
        while attempt < max_decomposition_attempts:
            pending = sum(
                1 for cr in plan.concept_resolutions
                if cr.tier == "needs_llm_decomposition"
            )
            if pending == 0:
                break
            t0 = time.time()
            n_decomposed = _decompose_pending(plan, llm, cmap, decomp_cache)
            metrics.decompose_s += time.time() - t0
            if n_decomposed == 0:
                break
            t0 = time.time()
            plan = plan_query(
                intent, resolved, cmap, metadata_db,
                decomp_cache=decomp_cache,
                universe_picker=universe_picker,
            )
            metrics.plan_s += time.time() - t0
            attempt += 1

    # 5. Census fetches ----------------------------------------------
    _progress(
        progress_cb, "Fetching data from Census API",
        f"{len(plan.calls)} call(s)",
    )
    t0 = time.time()
    fetch_results = await _run_fetches(
        plan, api_cache, api_key, max_concurrent=fetch_max_concurrent,
    )
    metrics.fetch_s = time.time() - t0
    metrics.census_calls_total = len(fetch_results)
    metrics.census_cache_hits = sum(
        1 for r in fetch_results if r.cache_hit
    )
    metrics.census_failures = sum(
        1 for r in fetch_results if not r.succeeded
    )
    _progress(
        progress_cb, "Fetches complete",
        f"{metrics.census_cache_hits}/{metrics.census_calls_total} cached, "
        f"{metrics.census_failures} failed",
    )

    # 6. Aggregation -------------------------------------------------
    _progress(progress_cb, "Aggregating results")
    t0 = time.time()
    if fetch_results:
        try:
            aggregated = aggregate_results(plan, fetch_results)
        except Exception as e:                 # pragma: no cover
            logger.exception("aggregator crashed")
            _accumulate_metrics(metrics, llm)
            metrics.total_s = time.time() - t_start
            return QueryResponse(
                query=query, intent=intent, resolved_geos=resolved,
                plan=plan, error=f"aggregator failed: {e}", metrics=metrics,
            )
    else:
        aggregated = AggregatedResult(values=[], fetch_failures=[])
    metrics.aggregate_s = time.time() - t0

    # 6b. Phase 4 realism agents -------------------------------------
    # Three post-aggregate enrichments that sit between aggregation
    # and synthesis. All defensive — any failure degrades to no-op
    # rather than blocking the answer.
    magnitude_framings: list[MagnitudeFraming] = []
    anomaly_flags: list[AnomalyFlag] = []
    followups: list[SuggestedFollowup] = []
    peer_contexts: list[PeerContext] = []

    if aggregated.values:
        _progress(progress_cb, "Framing magnitudes against comparators")
        try:
            magnitude_framings = contextualize_magnitudes(aggregated)
        except Exception as e:                     # pragma: no cover
            logger.warning("MagnitudeContextualizer failed: %s", e)
        try:
            anomaly_flags = detect_anomalies(magnitude_framings)
        except Exception as e:                     # pragma: no cover
            logger.warning("AnomalyDetector failed: %s", e)
        _progress(progress_cb, "Finding relevant follow-ups")
        try:
            followups = find_followups(
                user_query=query,
                frame=frame,
                aggregated=aggregated,
                llm=llm,
                temperature=config.get("vertex_ai", {}).get(
                    "temperature", 0.4,
                ),
            )
        except Exception as e:                     # pragma: no cover
            logger.warning("FollowupGapFinder failed: %s", e)

    # Peer context is also post-aggregate. It doesn't need LLM calls
    # and is read-only against peer_features.sqlite; still defensive
    # in case the file is missing or the anchor isn't in the index.
    if peer_retriever is not None:
        _progress(progress_cb, "Retrieving peer comparisons")
        try:
            peer_contexts = get_peer_contexts(
                resolved_geos=resolved, intent=intent_for_routing,
                query=query, peer_retriever=peer_retriever,
            )
        except Exception as e:                     # pragma: no cover
            logger.warning("peer_context failed: %s", e)

    # 7. Synthesis ---------------------------------------------------
    _progress(progress_cb, "Writing answer prose")
    t0 = time.time()
    answer: Optional[SynthesizedAnswer] = None
    err: Optional[str] = None
    try:
        answer = synthesize_answer(
            query, intent, plan, aggregated, llm,
            temperature=config.get("vertex_ai", {}).get(
                "synth_temperature", 0.2,
            ),
            frame=frame,
            magnitude_framings=magnitude_framings,
            anomaly_flags=anomaly_flags,
            followups=followups,
            peer_contexts=peer_contexts,
        )
    except SynthesisError as e:
        err = f"synthesis failed: {e}"
        logger.warning(err)
    metrics.synthesize_s = time.time() - t0

    # Prepend the exclusion caveat so the user sees the scope
    # change above the LLM's own caveats.
    if answer is not None and exclusion_caveat:
        answer = answer.model_copy(update={
            "caveats": [exclusion_caveat] + list(answer.caveats or []),
        })

    # Attach plain-English labels to every cited variable so the
    # Sources section can show, under each source, which variables
    # fed the answer (with their descriptions). Label lookup is a
    # handful of indexed SQLite reads — cheap; still catch and
    # fail-soft so a stale DB can't break the rendered answer.
    if answer is not None and answer.citations:
        try:
            from scripts.chatbot.synthesizer import (
                enrich_citation_variables,
            )
            enriched = enrich_citation_variables(
                answer.citations, metadata_db,
            )
            answer = answer.model_copy(update={"citations": enriched})
        except Exception as e:                         # pragma: no cover
            logger.warning(
                "enrich_citation_variables failed (%s); "
                "rendering citations without variable labels", e,
            )

    # 8. Derived-data dump -------------------------------------------
    # Write CSVs of the data that fed this answer + a variables.txt
    # sidecar decoding every variable_id into plain English. Opt-out
    # via ``derived_data.enabled: false`` in chatbot.yaml. Failures
    # are swallowed — a broken dump must not fail the user's answer.
    if dd_folder is not None and aggregated.values:
        _progress(progress_cb, "Writing derived-data CSVs")
        try:
            written = dump_derived_data(
                query=query, plan=plan, aggregated=aggregated,
                fetch_results=fetch_results,
                metadata_db=metadata_db, folder=dd_folder,
            )
            if written is not None:
                logger.info("derived_data dumped to %s", written)
                _progress(
                    progress_cb, "Derived data saved",
                    str(written),
                )
        except Exception as e:                         # pragma: no cover
            logger.warning("derived_data dump failed: %s", e)

    # 8b. Per-query response + logs exports -------------------------
    # Drops ``response.txt`` (the prose answer + findings + caveats +
    # sources) and ``logs.txt`` (every log record emitted during this
    # query) alongside the CSVs in the dump folder. Both are append-
    # only artifacts — failures here must never surface to the user.
    if dd_folder is not None and dd_folder.exists():
        try:
            _write_response_txt(
                dd_folder, query, answer, err,
                peer_contexts=peer_contexts,
            )
        except Exception as e:                         # pragma: no cover
            logger.warning("response.txt write failed: %s", e)
        try:
            _root_logger.removeHandler(_log_handler)
            _log_handler.flush()
            (dd_folder / "logs.txt").write_text(_log_buffer.getvalue())
        except Exception as e:                         # pragma: no cover
            logger.warning("logs.txt write failed: %s", e)
    else:
        # No dump folder — still detach the log handler so we don't
        # leak per-query handlers across runs in long-lived callers.
        try:
            _root_logger.removeHandler(_log_handler)
        except Exception:                              # pragma: no cover
            pass

    # 9. Wrap up -----------------------------------------------------
    _accumulate_metrics(metrics, llm)
    metrics.total_s = time.time() - t_start
    return QueryResponse(
        query=query,
        intent=intent_for_routing,
        resolved_geos=resolved,
        plan=plan, aggregated=aggregated, answer=answer,
        frame_match=frame_match,
        magnitude_framings=magnitude_framings,
        anomaly_flags=anomaly_flags,
        followups=followups,
        peer_contexts=peer_contexts,
        error=err, metrics=metrics,
    )


def answer_query_sync(*args: Any, **kwargs: Any) -> QueryResponse:
    """Synchronous convenience wrapper for non-async callers
    (REPL one-shot, scripts). Spawns its own event loop."""
    return asyncio.run(answer_query(*args, **kwargs))
