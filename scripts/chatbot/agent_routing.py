"""Agent-pipeline routing glue.

Bundles the Phase 1 nodes into a single function that takes
ExtractedConcepts and returns a ``RoutingOutcome``:

    1. ConceptRewriter      — geo-strip + classify each concept (LLM)
    2. SemanticRouter       — vector retrieval per rewrite
    3. RouterCritic         — accept / retry / fallback per concept (LLM)
    4. retry loop (max 1)   — re-rewrite with critic hint, re-route
    5. tier-cascade fallback — for critic-flagged misses only

The return carries both ``resolutions`` (what the planner needs) and
``rewrites`` (what later expansion stages need — IntentFacets drive
J.5 disparity / J.6 trend fan-out).

See ``docs/AGENT_PIPELINE_PLAN.md`` §7 Phase 1 for the role this
module plays in the full DAG.

Public API:
    RoutingOutcome              — return type
    route_concepts_with_critic  — entry point
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.decomposition_cache import DecompositionCache
from scripts.chatbot.llm_client import LLMClient
from scripts.chatbot.models import ExtractedConcept, ExtractedGeoRef
from scripts.chatbot.nodes.concept_rewriter import (
    ConceptRewrite, rewrite_concepts,
)
from scripts.chatbot.nodes.router_critic import (
    CriticVerdict, critique_routing,
)
from scripts.chatbot.planner import (
    ConceptResolution, resolve_concept,
)
from scripts.chatbot.semantic_router import RoutedResult, SemanticRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RoutingOutcome:
    """Output of the agent-routing layer.

    Carries the planner-ready resolutions plus the per-concept
    rewrites the Expander needs for downstream fan-out (J.5
    disparity, J.6 trend).
    """
    resolutions: list[ConceptResolution] = field(default_factory=list)
    rewrites: list[ConceptRewrite] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def route_concepts_with_critic(
    concepts: list[ExtractedConcept],
    geo_refs: list[ExtractedGeoRef],
    data_levels: list[str],
    semantic_router: SemanticRouter,
    llm: LLMClient,
    *,
    cmap: Optional[ConceptMap] = None,
    metadata_db: Optional[sqlite3.Connection] = None,
    decomp_cache: Optional[DecompositionCache] = None,
    max_retries: int = 1,
    top_k: int = 20,
    temperature: float = 0.1,
) -> RoutingOutcome:
    """Run rewrite → route → critique → (retry|fallback) for every concept.

    Returns a ``RoutingOutcome`` with resolutions and rewrites both
    in the same order as ``concepts``. Each resolution carries:
      - ``tier="tier_semantic"`` + ``routed_result`` on accept/retry success
      - Whatever tier ``resolve_concept`` produced on fallback (legacy path)
      - ``tier="unresolved"`` on critic-fallback when no cmap/metadata_db
        is provided

    Notes on each resolution include which pass produced it (``pass=1``
    for first-try accept, ``pass=2`` for retry-accept, ``fallback``
    for tier-cascade) so downstream code can surface the routing
    provenance in UI / metrics.
    """
    if not concepts:
        return RoutingOutcome()

    # ---- Pass 0: Tier 1 short-circuit ----------------------------------
    # Curated concept_map entries always win over the semantic router:
    # the cmap is hand-maintained, the router guesses. If a concept
    # hits Tier 1, use that mapping directly and skip the rewriter /
    # router / critic entirely for that concept. This also catches
    # frame-added concepts (e.g. "share without a vehicle" from the
    # food_access frame) that alias a Tier 1 entry ("households
    # without a vehicle" → B25044) — without this short-circuit the
    # semantic router would route them to whatever is most cosine-
    # adjacent, which is often the wrong table.
    from scripts.chatbot.planner import _tier1_lookup

    tier1_hits: dict[int, tuple] = {}
    unresolved_concepts: list[ExtractedConcept] = []
    unresolved_indices: list[int] = []
    if cmap is not None:
        for i, c in enumerate(concepts):
            entry, key_used = _tier1_lookup(cmap, c)
            if entry is not None:
                tier1_hits[i] = (entry, key_used)
            else:
                unresolved_concepts.append(c)
                unresolved_indices.append(i)
        if tier1_hits:
            logger.info(
                "agent_routing: Tier 1 short-circuit for %d of %d "
                "concept(s): %r",
                len(tier1_hits), len(concepts),
                [c.text for i, c in enumerate(concepts) if i in tier1_hits],
            )
    else:
        unresolved_concepts = list(concepts)
        unresolved_indices = list(range(len(concepts)))

    # If everything was a Tier 1 hit, build resolutions directly and
    # return — no LLM calls needed at all.
    if not unresolved_concepts:
        resolutions: list[ConceptResolution] = []
        empty_rewrites: list[ConceptRewrite] = []
        for i, c in enumerate(concepts):
            entry, key_used = tier1_hits[i]
            resolutions.append(ConceptResolution(
                concept=c,
                tier="tier_1_concept_map",
                concept_entry=entry,
                notes=[f"tier 1 match via key={key_used!r}"],
            ))
            empty_rewrites.append(ConceptRewrite(
                original_text=c.text, embedding_text=c.text,
                measure_type_sought="count", units_expected=None,
                is_total_likely=False, disparity_eligible=False,
                trend_eligible=True, is_composite=False,
            ))
        return RoutingOutcome(
            resolutions=resolutions, rewrites=empty_rewrites,
        )

    # ---- Pass 1: rewrite + route + critique ---------------------------
    # Run only on the concepts that missed Tier 1.
    rewrites = rewrite_concepts(
        unresolved_concepts, geo_refs, llm, temperature=temperature,
    )
    routed_results = [
        _route_one(semantic_router, rw.embedding_text, data_levels, top_k)
        for rw in rewrites
    ]
    verdicts = critique_routing(
        rewrites, routed_results, llm,
        metadata_db=metadata_db,
        temperature=temperature,
    )

    # ---- Retry loop ---------------------------------------------------
    # Index arithmetic here is LOCAL to unresolved_concepts — the
    # retry loop operates on the same arrays (rewrites, routed_
    # results, verdicts) that were built from unresolved_concepts.
    if max_retries > 0:
        retry_local = [
            i for i, v in enumerate(verdicts) if v.decision == "retry"
        ]
        if retry_local:
            retry_concepts = [unresolved_concepts[i] for i in retry_local]
            hints = {
                unresolved_concepts[i].text: verdicts[i].rewrite_hint or ""
                for i in retry_local
                if verdicts[i].rewrite_hint
            }
            new_rewrites = rewrite_concepts(
                retry_concepts, geo_refs, llm,
                hints=hints, temperature=temperature,
            )
            new_routed = [
                _route_one(
                    semantic_router, rw.embedding_text,
                    data_levels, top_k,
                )
                for rw in new_rewrites
            ]
            for loc_i, retry_i in enumerate(retry_local):
                rewrites[retry_i] = new_rewrites[loc_i]
                routed_results[retry_i] = new_routed[loc_i]
                verdicts[retry_i] = CriticVerdict(
                    original_text=verdicts[retry_i].original_text,
                    decision="accept",
                    reason=(
                        f"retry accepted (was: "
                        f"{verdicts[retry_i].reason})"
                    ),
                    rewrite_hint=None,
                )

    # ---- Build semantic resolutions for the unresolved concepts -------
    sem_res_by_global: dict[int, ConceptResolution] = {}
    for loc_i, global_i in enumerate(unresolved_indices):
        concept = unresolved_concepts[loc_i]
        rewrite = rewrites[loc_i]
        routed = routed_results[loc_i]
        verdict = verdicts[loc_i]
        if verdict.decision == "accept":
            sem_res_by_global[global_i] = _resolution_from_semantic(
                concept, rewrite, routed, verdict,
            )
        elif verdict.decision == "fallback":
            sem_res_by_global[global_i] = _resolution_from_fallback(
                concept, rewrite, verdict,
                cmap=cmap, metadata_db=metadata_db,
                data_levels=data_levels, decomp_cache=decomp_cache,
            )
        else:
            logger.warning(
                "leftover retry verdict for %r — treating as fallback",
                concept.text,
            )
            sem_res_by_global[global_i] = _resolution_from_fallback(
                concept, rewrite, verdict,
                cmap=cmap, metadata_db=metadata_db,
                data_levels=data_levels, decomp_cache=decomp_cache,
            )

    # ---- Assemble final outputs in the original concepts order --------
    # Tier 1 hits synthesize an empty ConceptRewrite since no LLM
    # rewrite was done for them; consumers that key off rewrites
    # (trend expansion via IntentFacets) still get an entry per
    # concept slot.
    final_resolutions: list[ConceptResolution] = []
    final_rewrites: list[ConceptRewrite] = []
    loc_counter = 0
    for i, concept in enumerate(concepts):
        if i in tier1_hits:
            entry, key_used = tier1_hits[i]
            final_resolutions.append(ConceptResolution(
                concept=concept,
                tier="tier_1_concept_map",
                concept_entry=entry,
                notes=[f"tier 1 match via key={key_used!r}"],
            ))
            # measure_type_sought is a required Literal — use a
            # bland placeholder since Tier 1 resolutions don't route
            # through the router + critic anyway. "count" is the
            # widest fit and won't trigger any measure-specific
            # downstream branching.
            final_rewrites.append(ConceptRewrite(
                original_text=concept.text, embedding_text=concept.text,
                measure_type_sought="count", units_expected=None,
                is_total_likely=False, disparity_eligible=False,
                trend_eligible=True, is_composite=False,
            ))
        else:
            final_resolutions.append(sem_res_by_global[i])
            final_rewrites.append(rewrites[loc_counter])
            loc_counter += 1

    return RoutingOutcome(
        resolutions=final_resolutions, rewrites=final_rewrites,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _route_one(
    router: SemanticRouter, text: str, data_levels: list[str], top_k: int,
) -> RoutedResult:
    """Wrapper that consistently passes coverage filter through."""
    return router.route(
        text, top_k=top_k,
        require_coverage_at=data_levels if data_levels else None,
    )


def _resolution_from_semantic(
    concept: ExtractedConcept,
    rewrite: ConceptRewrite,
    routed: RoutedResult,
    verdict: CriticVerdict,
) -> ConceptResolution:
    """Build a tier_semantic ConceptResolution from accepted routing."""
    notes = [
        f"rewrite: {rewrite.embedding_text!r}",
        f"critic: {verdict.decision} — {verdict.reason}",
    ]
    if routed.top_variables:
        tv = routed.top_variables[0]
        notes.append(
            f"top variable {tv.target_dataset}/{tv.target_table_id}/"
            f"{tv.target_variable_id} score={tv.aggregate_score:.2f}"
        )
    elif routed.top_tables:
        tt = routed.top_tables[0]
        notes.append(
            f"table-only hit {tt.target_dataset}/{tt.target_table_id} "
            f"score={tt.aggregate_score:.2f}"
        )
    # If the router genuinely returned nothing (shouldn't hit here
    # because those short-circuit to fallback in the critic), mark
    # unresolved so downstream doesn't try to build calls from empty.
    if not routed.top_variables and not routed.top_tables:
        return ConceptResolution(
            concept=concept, tier="unresolved",
            routed_result=routed, rewrite=rewrite,
            notes=notes + [
                "semantic router returned no candidates despite "
                "critic accept — treated as unresolved",
            ],
        )
    return ConceptResolution(
        concept=concept, tier="tier_semantic",
        routed_result=routed, rewrite=rewrite, notes=notes,
    )


def _resolution_from_fallback(
    concept: ExtractedConcept,
    rewrite: ConceptRewrite,
    verdict: CriticVerdict,
    *,
    cmap: Optional[ConceptMap],
    metadata_db: Optional[sqlite3.Connection],
    data_levels: list[str],
    decomp_cache: Optional[DecompositionCache],
) -> ConceptResolution:
    """Run the legacy tier cascade for a concept the critic flagged as
    fallback. If cmap/metadata_db aren't provided, surface unresolved
    with a note explaining what happened.
    """
    header = [
        f"rewrite: {rewrite.embedding_text!r}",
        f"critic: fallback — {verdict.reason}",
    ]
    if cmap is None or metadata_db is None:
        return ConceptResolution(
            concept=concept, tier="unresolved",
            notes=header + [
                "tier-cascade fallback requested but no cmap / "
                "metadata_db provided — cannot resolve",
            ],
        )
    resolved = resolve_concept(
        concept, cmap, metadata_db, data_levels,
        decomp_cache=decomp_cache,
    )
    # Preserve the critic's reason in the resolution's notes so
    # downstream metrics can see the fallback trail.
    resolved.notes = header + resolved.notes
    return resolved
