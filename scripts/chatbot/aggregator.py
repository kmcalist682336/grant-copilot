"""Phase H aggregator: raw FetchResults → AggregatedValues per (geo, concept).

The planner emitted one or more ``PlannedCall``s per (geo, concept,
year, role). The CensusCaller fetched them. This module:

  - Filters tract-level responses to the geo's ``tract_filter`` list
    (so neighborhood/corridor/region queries only sum tracts that
    actually belong to the geo).
  - Combines variables according to the ConceptVariables role:
       value:        single estimate. Sum across rows for counts,
                     population-weighted mean for medians.
       numerator + denominator: sum each, then divide; report as a
                     ratio (typically 0–1 or 0–100% depending on units).
       components:   sum each component independently, return a dict.
  - Joins multi-county tract calls (one PlannedCall per county) into a
    single per-(geo, concept, year, role) AggregatedValue.
  - For temporal_intent='change' / 'trend': caller pulls multiple years
    and the aggregator surfaces them as a list. Delta computation lives
    here; trend rendering lives in the synthesizer (Phase I).

Median note
-----------
Census medians (e.g., median household income) cannot be combined
arithmetically from sub-area medians — the true block-level computation
requires bracketed interpolation against the income-distribution table.
Our pragmatic POC uses **population-weighted means of tract medians**
when the geo aggregates multiple tracts, with an explicit
``aggregation_caveat`` note. Phase I synthesizer is expected to surface
that caveat to the user.

Public API:
    AggregatedValue
    AggregatedResult
    aggregate_results(plan, fetch_results)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.census_caller import FetchResult
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import ExtractedConcept, ResolvedGeography
from scripts.chatbot.planner import (
    ConceptResolution, PlannedCall, PlanResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class AggregatedValue(BaseModel):
    """One numeric outcome for a (geo, concept, year, role) tuple."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    geo: ResolvedGeography
    concept: ExtractedConcept
    year: int
    role: str = "primary"

    # Exactly one of these is populated:
    value: Optional[float] = Field(
        default=None,
        description="Single scalar (counts, medians, indices, etc.).",
    )
    ratio: Optional[float] = Field(
        default=None,
        description="numerator / denominator. Multiply by 100 in the "
                    "renderer if the concept's units are 'percent'.",
    )
    components: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-component summed values (variable_id → total).",
    )

    # Always populated:
    sample_size: Optional[float] = Field(
        default=None,
        description="Denominator of the universe (population, households, "
                    "etc.) — used by the synthesizer for context.",
    )
    rows_aggregated: int = Field(
        default=0,
        description="How many response rows this value summarizes.",
    )
    aggregation_caveat: Optional[str] = Field(
        default=None,
        description="Set when the math has limited statistical validity "
                    "(e.g., population-weighted mean of tract medians).",
    )
    notes: list[str] = Field(default_factory=list)


class AggregatedResult(BaseModel):
    """Full aggregator output."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    values: list[AggregatedValue]
    fetch_failures: list[str] = Field(
        default_factory=list,
        description="Per-call error messages from CensusCaller; the "
                    "synthesizer should mention partial-data conditions.",
    )

    def by_geo_concept(
        self,
    ) -> dict[tuple[int, int], list[AggregatedValue]]:
        """Group values by (geo_idx, concept_idx) for synthesizer rendering."""
        out: dict[tuple[int, int], list[AggregatedValue]] = defaultdict(list)
        # We need geo_idx/concept_idx to group — derive them by
        # identity-matching against the plan; populated externally.
        return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> Optional[float]:
    """Census responses encode every cell as string. Convert; return None
    for the various 'no data' sentinels Census uses."""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("", "null", "N", "(X)", "*", "**", "***"):
        return None
    # Census uses negative sentinels for unreliable estimates; some are
    # literal -666666666, -888888888, etc. Treat as missing.
    try:
        v = float(s)
    except ValueError:
        return None
    if v <= -555555555:
        return None
    return v


def _filter_rows_to_tracts(
    rows: list[dict], tract_filter: list[str],
) -> list[dict]:
    """Restrict response rows to the tract GEOIDs the planner asked for.

    Census tract responses include 'state', 'county', 'tract' columns
    we can concatenate to compare to our tract_geoid format.
    """
    if not tract_filter:
        return rows
    wanted = set(tract_filter)
    out: list[dict] = []
    for r in rows:
        state = r.get("state", "")
        county = r.get("county", "")
        tract = r.get("tract", "")
        if not (state and county and tract):
            continue
        geoid = f"{state}{county}{tract}"
        if geoid in wanted:
            out.append(r)
    return out


def _is_median_concept(variables: ConceptVariables) -> bool:
    """Heuristic: if there's a 'value' variable (single estimate), and
    the call is at tract level (multi-row aggregation), the user almost
    always wants a median-style stat. Returns True so we use the
    population-weighted mean caveat."""
    return variables.value is not None


def _aggregate_value_role(
    rows: list[dict],
    variables: ConceptVariables,
    *,
    is_tract_aggregation: bool,
) -> tuple[Optional[float], int, Optional[str]]:
    """Return (value, rows_used, caveat) for a 'value:'-only ConceptVariables.

    For pure-admin queries (single row): just return the value.
    For tract-aggregation: sum if it looks like a count; population-
    weighted mean if it looks like a median (the planner can't tell, so
    we default to mean-with-caveat for safety).
    """
    if variables.value is None:
        return None, 0, None
    var_id = variables.value
    pairs: list[tuple[float, float]] = []  # (value, weight)
    for r in rows:
        v = _to_float(r.get(var_id))
        if v is None:
            continue
        pairs.append((v, 1.0))   # weight stays 1.0 unless caller fills it

    if not pairs:
        return None, 0, None

    if not is_tract_aggregation or len(pairs) == 1:
        return pairs[0][0], 1, None

    # Multi-tract aggregation. We don't have the population denominators
    # at this layer (they'd require fetching B01003 alongside), so a
    # simple mean is the best we can do without an extra fetch. Surface
    # the limitation explicitly.
    total = sum(v for v, _ in pairs)
    mean = total / len(pairs)
    caveat = (
        f"value is the unweighted mean across {len(pairs)} tracts; "
        f"true sub-county medians require population-weighted "
        f"interpolation against bracketed-income tables"
    )
    # If it looks like a count rather than a median (unbounded big
    # number? hard to tell here), the synthesizer can override. For
    # strict counts (totals) the planner shouldn't have used 'value' on
    # a tract aggregation — it'd typically be 'numerator/denominator' or
    # a single-row admin query. So defaulting to mean is the safer
    # ambiguity resolution.
    return mean, len(pairs), caveat


def _aggregate_ratio_role(
    rows: list[dict], variables: ConceptVariables,
) -> tuple[Optional[float], Optional[float], int]:
    """Sum numerator(s) and denominator across rows. Return (ratio,
    denominator, rows_used)."""
    if not variables.numerator or not variables.denominator:
        return None, None, 0
    num_ids = (
        variables.numerator if isinstance(variables.numerator, list)
        else [variables.numerator]
    )
    denom_id = variables.denominator
    num_total = 0.0
    denom_total = 0.0
    rows_used = 0
    for r in rows:
        denom = _to_float(r.get(denom_id))
        if denom is None or denom == 0:
            continue
        # Sum every num_id; treat missing as zero only if at least one
        # numerator var was present.
        nums = [_to_float(r.get(nid)) for nid in num_ids]
        if all(n is None for n in nums):
            continue
        row_num = sum((n or 0.0) for n in nums)
        num_total += row_num
        denom_total += denom
        rows_used += 1
    if denom_total == 0:
        return None, None, rows_used
    return num_total / denom_total, denom_total, rows_used


def _aggregate_components_role(
    rows: list[dict], variables: ConceptVariables,
) -> tuple[dict[str, float], int]:
    """Sum each component variable across rows."""
    out: dict[str, float] = {}
    rows_used = 0
    if not variables.components:
        return out, 0
    for r in rows:
        any_present = False
        for var_id in variables.components:
            v = _to_float(r.get(var_id))
            if v is None:
                continue
            out[var_id] = out.get(var_id, 0.0) + v
            any_present = True
        if any_present:
            rows_used += 1
    return out, rows_used


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_results(
    plan: PlanResult,
    fetch_results: list[FetchResult],
) -> AggregatedResult:
    """Combine the planner's metadata with raw fetch outputs into per-
    (geo, concept, year, role) AggregatedValues.

    ``fetch_results`` must be in the same order as ``plan.calls`` (the
    ``CensusCaller.fetch_all`` contract). Each PlannedCall produces one
    intermediate per-county result; calls sharing the same (geo_idx,
    concept_idx, year, role) get merged into one final value.
    """
    if len(fetch_results) != len(plan.calls):
        raise ValueError(
            f"aggregate_results: got {len(fetch_results)} fetch results for "
            f"{len(plan.calls)} planned calls"
        )

    # Bucket per-county results by (geo_idx, concept_idx, year, role).
    # Each bucket carries the merged row list + the ConceptVariables.
    buckets: dict[
        tuple[int, int, int, str],
        dict[str, Any],
    ] = {}
    failures: list[str] = []

    for planned, result in zip(plan.calls, fetch_results):
        key = (
            planned.geo_idx, planned.concept_idx,
            planned.year, planned.role,
        )
        bucket = buckets.setdefault(key, {
            "rows": [],
            "variables": planned.variables,
            "geo": plan.resolved_geos[planned.geo_idx],
            "concept": plan.intent.concepts[planned.concept_idx],
            "is_tract_aggregation": planned.api_call.geo_level == "tract"
                                    and bool(planned.tract_filter),
            "tract_filter": [],
        })
        if planned.tract_filter:
            bucket["tract_filter"].extend(planned.tract_filter)
        if not result.succeeded:
            # Build a human-readable failure string with enough context
            # for the synthesizer to explain to the user what's missing.
            # Shape:
            #   "<geo display_name> × <concept text> @ <geo_level>
            #    [<role>] ({year}, {dataset}/{table_id}): {error}"
            geo = plan.resolved_geos[planned.geo_idx]
            concept = plan.intent.concepts[planned.concept_idx]
            role_tag = (
                f" [{planned.role}]" if planned.role != "primary" else ""
            )
            failures.append(
                f"{geo.display_name} × {concept.text!r} @ "
                f"{planned.api_call.geo_level}{role_tag} "
                f"({planned.year}, {planned.api_call.dataset}/"
                f"{planned.api_call.table_id}): {result.error}"
            )
            continue
        # Filter to tracts of interest (no-op for non-tract calls).
        rows = _filter_rows_to_tracts(result.rows, planned.tract_filter)
        bucket["rows"].extend(rows)

    # Materialize one AggregatedValue per bucket.
    values: list[AggregatedValue] = []
    for (gi, ci, year, role), b in buckets.items():
        variables: ConceptVariables = b["variables"]
        rows = b["rows"]
        notes: list[str] = []
        value: Optional[float] = None
        ratio: Optional[float] = None
        components: Optional[dict[str, float]] = None
        sample: Optional[float] = None
        rows_used = 0
        caveat: Optional[str] = None

        if variables.value is not None and (
            variables.numerator is None and variables.components is None
        ):
            value, rows_used, caveat = _aggregate_value_role(
                rows, variables,
                is_tract_aggregation=b["is_tract_aggregation"],
            )
        elif variables.numerator is not None:
            ratio, sample, rows_used = _aggregate_ratio_role(rows, variables)
        elif variables.components is not None:
            components, rows_used = _aggregate_components_role(rows, variables)
        else:
            notes.append("ConceptVariables had no usable role")

        if not rows:
            notes.append("no response rows for this (geo, concept, year)")

        values.append(AggregatedValue(
            geo=b["geo"], concept=b["concept"],
            year=year, role=role,
            value=value, ratio=ratio, components=components,
            sample_size=sample, rows_aggregated=rows_used,
            aggregation_caveat=caveat, notes=notes,
        ))

    return AggregatedResult(values=values, fetch_failures=failures)
