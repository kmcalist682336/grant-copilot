"""J.6 — trend prior-period fetch.

When the user query has ``temporal_intent="latest"`` AND the
ConceptRewriter flagged the concept as ``trend_eligible=True``, also
fetch the same (geo × concept × table) at (latest - N) so the
synthesizer can frame trend direction + magnitude.

The prior fetch carries ``role="prior_period"`` so downstream code
(MagnitudeContextualizer in Phase 4, synthesizer prompt) can use it
without confusing it with a regular trend-mode pull.

Public API:
    prior_period_calls(plan, rewrites, intent, metadata_db,
                       *, lookback_years=3) -> list[PlannedCall]
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional
from urllib.parse import quote

from scripts.chatbot.census_caller import APIPlanCall
from scripts.chatbot.metadata_search import find_supported_years
from scripts.chatbot.models import ExtractedIntent
from scripts.chatbot.nodes.concept_rewriter import ConceptRewrite
from scripts.chatbot.planner import (
    PlannedCall, PlanResult, _ttl_for_dataset,
)

logger = logging.getLogger(__name__)


def _rebuild_url_for_year(call: APIPlanCall, year: int) -> str:
    """Same URL as ``call`` but with the target year substituted in
    the path. Variables / for / in clauses are unchanged."""
    # Simple split — the URL shape is always
    # https://api.census.gov/data/{year}/{dataset}?...
    # so we can swap the segment after /data/.
    prefix = "https://api.census.gov/data/"
    if not call.url.startswith(prefix):
        return call.url
    rest = call.url[len(prefix):]
    # rest = "{year}/{dataset}?..."
    slash = rest.find("/")
    if slash < 0:
        return call.url
    after_year = rest[slash:]
    return f"{prefix}{year}{after_year}"


def _pick_prior_year(
    metadata_db: sqlite3.Connection,
    table_id: str, dataset: str, data_level: str,
    primary_year: int, lookback_years: int,
) -> Optional[int]:
    """Pick the supported year closest to (primary_year - lookback_years)
    that's <= the requested target. Returns None when no prior vintage
    is available for this (table, dataset, geo_level) combo."""
    supported = find_supported_years(
        metadata_db, table_id, dataset, [data_level],
    )
    if not supported:
        return None
    target = primary_year - lookback_years
    candidates = [y for y in supported if y <= target]
    if candidates:
        return max(candidates)
    # If no year is at or below the target, pick the oldest available.
    return min(supported)


def prior_period_calls(
    plan: PlanResult,
    rewrites: list[ConceptRewrite],
    intent: ExtractedIntent,
    metadata_db: sqlite3.Connection,
    *,
    lookback_years: int = 3,
) -> list[PlannedCall]:
    """Generate ``role="prior_period"`` calls for trend-eligible concepts.

    Returns an empty list when:
      - intent.temporal_intent != "latest" (multi-year already in play)
      - no concept in ``rewrites`` is trend_eligible
      - no prior year is available for the (table, dataset) combos

    The returned list is meant to be appended to ``plan.calls`` by the
    Expander; we don't mutate the plan here so the caller controls
    placement and metrics.
    """
    if intent.temporal_intent != "latest":
        return []
    if not plan.calls or not rewrites:
        return []

    # Index rewrites by original concept text for lookup-by-concept_idx.
    # Concept order in plan.intent.concepts == order in rewrites.
    if len(rewrites) != len(plan.intent.concepts):
        logger.debug(
            "prior_period_calls: rewrites length (%d) != concepts (%d); "
            "skipping trend expansion",
            len(rewrites), len(plan.intent.concepts),
        )
        return []

    extras: list[PlannedCall] = []
    for call in plan.calls:
        # Only fan out from primary calls — we already skip if comparator
        # / disparity / sub roles are present (those are themselves
        # expansions; layering trend on top would explode call count).
        if call.role != "primary":
            continue
        ci = call.concept_idx
        if ci < 0 or ci >= len(rewrites):
            continue
        rw = rewrites[ci]
        if not rw.trend_eligible:
            continue
        prior_year = _pick_prior_year(
            metadata_db, call.api_call.table_id, call.api_call.dataset,
            call.api_call.geo_level, call.year, lookback_years,
        )
        if prior_year is None or prior_year == call.year:
            continue
        new_url = _rebuild_url_for_year(call.api_call, prior_year)
        new_api = APIPlanCall(
            url=new_url,
            table_id=call.api_call.table_id,
            variables=list(call.api_call.variables),
            geo_level=call.api_call.geo_level,
            geo_filter_ids=list(call.api_call.geo_filter_ids),
            year=prior_year,
            dataset=call.api_call.dataset,
            ttl_seconds=_ttl_for_dataset(call.api_call.dataset),
        )
        extras.append(PlannedCall(
            api_call=new_api,
            geo_idx=call.geo_idx,
            concept_idx=call.concept_idx,
            year=prior_year,
            role="prior_period",
            variables=call.variables,
            tract_filter=list(call.tract_filter),
        ))
    return extras
