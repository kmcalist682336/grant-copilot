"""Deterministic planner for record-level datasets.

This module is intentionally small and dataset-agnostic at its public
boundary.  HMDA is the first record-level dataset, so its variable aliases
and decoded filter values live here until they can be moved into a shared
dataset registry.  The LLM supplies *which* dimensions and values were
explicitly requested; this module chooses variables, builds a structured
``APIPlanCall``, and leaves SQL generation to ``DuckDBCaller``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from scripts.chatbot.census_caller import APIPlanCall, RecordFilter
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import (
    ExtractedAnalysis, ExtractedConcept, ExtractedIntent, ExtractedFilter,
    ResolvedGeography,
)
from scripts.chatbot.planner import ConceptResolution, PlanResult, PlannedCall

logger = logging.getLogger(__name__)


# Stable IDs from the HMDA variable registry/cards.  These aliases are a
# guardrail for the high-value dimensions where semantic search can otherwise
# confuse primary- and co-applicant fields.  Unknown dimensions still use the
# semantic router and are never silently substituted.
_HMDA_ALIASES: dict[str, str] = {
    "income": "c1aa5d4f3f72",
    "applicant income": "c1aa5d4f3f72",
    "loan applicant income": "c1aa5d4f3f72",
    "applicant sex": "6057363dc2e9",
    "primary applicant sex": "6057363dc2e9",
    "sex of applicant": "6057363dc2e9",
    "applicant race": "38ad9c360a98",
    "applicant race 1": "38ad9c360a98",
    "race of applicant": "38ad9c360a98",
    "action taken": "906bb78b0f70",
    "mortgage application outcome": "906bb78b0f70",
    "county": "6422e2d2aab7",
    "county code": "6422e2d2aab7",
}

_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "sex": {
        "female": "Female", "woman": "Female", "women": "Female",
        "male": "Male", "man": "Male", "men": "Male",
        "joint": "Joint",
    },
    "race": {
        "black": "Black or African American",
        "african american": "Black or African American",
        "white": "White",
        "asian": "Asian",
        "native hawaiian": "Native Hawaiian or Other Pacific Islander",
    },
    "action": {
        "originated": "Loan originated",
        "loan originated": "Loan originated",
        "denied": "Application denied",
        "denial": "Application denied",
        "withdrawn": "Application withdrawn by applicant",
        "closed for incompleteness": "File closed for incompleteness",
    },
}


def _key(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _variable_id(
    concept: ExtractedConcept,
    semantic_router: Optional[object],
) -> tuple[str, Optional[object]]:
    """Resolve one concept to an HMDA variable ID.

    Explicit aliases win for primary HMDA fields.  The semantic router is a
    fallback for variables not yet added to the small demo alias map.
    """
    candidates = [_key(concept.canonical_hint), _key(concept.text)]
    for candidate in candidates:
        if candidate in _HMDA_ALIASES:
            return _HMDA_ALIASES[candidate], None

    if semantic_router is None:
        raise ValueError(
            f"No deterministic HMDA alias for {concept.text!r} and no "
            "semantic router was configured"
        )
    search_text = (concept.canonical_hint or concept.text).strip()
    routed = semantic_router.route_dataset(
        search_text, target_dataset="hmda", top_k=10,
    )
    for target in routed.top_variables:
        if target.target_variable_id:
            return target.target_variable_id, routed
    raise ValueError(
        f"HMDA variable could not be resolved for {search_text!r}"
    )


def _decoded_value(filter_item: ExtractedFilter) -> Any:
    """Map common natural-language values to decoded HMDA labels.

    If no explicit alias exists, preserve the user's value.  DuckDB compares
    categorical values case-insensitively, so this remains useful for labels
    already emitted by the codebook without inventing a new category.
    """
    raw = (
        filter_item.normalized_value_hint
        or filter_item.value_text
        or ""
    ).strip()
    dim = _key(filter_item.dimension.canonical_hint or filter_item.dimension.text)
    for token, mapping in _VALUE_ALIASES.items():
        if token in dim:
            return mapping.get(raw.lower(), raw)
    return raw


def _geo_prefixes(geo: ResolvedGeography) -> list[str]:
    """Return FIPS prefixes usable against a record census_tract column."""
    if geo.tract_geoids:
        return sorted(set(str(value) for value in geo.tract_geoids))
    if geo.geo_level == "state":
        return [str(geo.geo_id)[:2]]
    if geo.geo_level == "county":
        return [str(geo.geo_id)[:5]]
    # A place/MSA may not have a tract list in the gazetteer.  Do not guess
    # a prefix; the caller will query the available record set explicitly.
    return []


def _record_analysis(intent: ExtractedIntent) -> list[ExtractedAnalysis]:
    analyses: list[ExtractedAnalysis] = []
    for analysis in intent.analyses:
        parts = [analysis.measure, *analysis.groupings]
        parts.extend(f.dimension for f in analysis.filters)
        if any(
            part is not None and part.dataset_hint in ("hmda", "both")
            for part in parts
        ):
            analyses.append(analysis)
    return analyses


def has_record_analysis(intent: ExtractedIntent) -> bool:
    return bool(_record_analysis(intent))


def plan_record_query(
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    *,
    semantic_router: Optional[object],
    table_id: str = "hmda",
    dataset: str = "hmda",
    file_glob: str = "*.parquet",
    record_id_column: str = "record_id",
) -> PlanResult:
    """Build a record-level plan with parameterized filter metadata.

    The returned plan uses the existing ``PlanResult``/``PlannedCall``
    contracts, so the current orchestrator, aggregator, citations, and Docker
    response mapper can consume it without a second pipeline.
    """
    analyses = _record_analysis(intent)
    if not analyses:
        return PlanResult(
            intent=intent, resolved_geos=resolved_geos,
            concept_resolutions=[], calls=[],
            notes=["no record-level analysis was extracted"],
        )

    if not resolved_geos:
        raise ValueError(
            "Record queries without a geography are not yet enabled; "
            "provide a state, county, tract, or another supported area."
        )

    concepts = list(intent.concepts)
    resolutions: list[ConceptResolution] = []
    calls: list[PlannedCall] = []
    notes: list[str] = []
    years = list(intent.years) or [2024]

    for analysis in analyses:
        if analysis.measure is None:
            notes.append("record analysis has no measure; skipped")
            continue
        if analysis.operation not in {"value", "count", "sum", "average", "median"}:
            notes.append(
                f"operation {analysis.operation!r} needs an explicit "
                "record aggregation recipe; no SQL was generated"
            )
            continue

        measure = analysis.measure
        measure_id, measure_route = _variable_id(measure, semantic_router)
        concept_idx = len(concepts)
        concepts.append(measure)
        resolutions.append(ConceptResolution(
            concept=measure,
            tier="tier_semantic" if measure_route is not None else "tier_1_concept_map",
            routed_result=measure_route,
        ))

        record_filters: list[RecordFilter] = []
        for filter_item in analysis.filters:
            filter_id, _ = _variable_id(
                filter_item.dimension, semantic_router,
            )
            if filter_item.operator in {"is_null", "is_not_null"}:
                record_filters.append(RecordFilter(
                    variable_id=filter_id,
                    operator=filter_item.operator,
                ))
            else:
                value = _decoded_value(filter_item)
                if value == "":
                    raise ValueError(
                        f"Filter {filter_item.dimension.text!r} has no value"
                    )
                record_filters.append(RecordFilter(
                    variable_id=filter_id,
                    operator=filter_item.operator,
                    value=value,
                ))

        for year in years:
            for geo_idx, geo in enumerate(resolved_geos):
                geo_prefixes = _geo_prefixes(geo)
                api_call = APIPlanCall(
                    url=f"record://{dataset}/{year}/{table_id}",
                    table_id=table_id,
                    variables=[measure_id],
                    geo_level="record",
                    geo_filter_ids=[],
                    geo_prefixes=geo_prefixes,
                    year=int(year),
                    dataset=dataset,
                    ttl_seconds=24 * 60 * 60,
                    record_filters=list(record_filters),
                )
                calls.append(PlannedCall(
                    api_call=api_call,
                    geo_idx=geo_idx,
                    concept_idx=concept_idx,
                    year=int(year),
                    role="primary",
                    operation=analysis.operation,
                    variables=ConceptVariables(value=measure_id),
                    tract_filter=[],
                ))

    # Preserve the original Census concepts and append record measures so
    # concept indexes in existing Census calls remain stable in mixed plans.
    final_intent = intent.model_copy(update={"concepts": concepts})
    return PlanResult(
        intent=final_intent,
        resolved_geos=resolved_geos,
        concept_resolutions=resolutions,
        calls=calls,
        notes=notes,
    )
