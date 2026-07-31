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
import sqlite3
from itertools import product
from typing import Any, Optional

from scripts.chatbot.census_caller import APIPlanCall, RecordFilter
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import (
    ExtractedAnalysis, ExtractedConcept, ExtractedIntent, ExtractedFilter,
    ResolvedGeography,
)
from scripts.chatbot.metadata_search import find_supported_years
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, PlannedCall, _pick_years,
)

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
    "applicant age": "78651f637517",
    "age of applicant": "78651f637517",
    "borrower age": "78651f637517",
    "action taken": "906bb78b0f70",
    "mortgage application outcome": "906bb78b0f70",
    "application status": "906bb78b0f70",
    "loan application status": "906bb78b0f70",
    "mortgage applications": "906bb78b0f70",
    "loan applications": "906bb78b0f70",
    "applications": "906bb78b0f70",
    "county": "6422e2d2aab7",
    "county code": "6422e2d2aab7",
    "loan amount": "c02eb39025e6",
    "mortgage amount": "c02eb39025e6",
    "requested loan amount": "c02eb39025e6",
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
    "status": {
        "originated": "Loan originated",
        "loan originated": "Loan originated",
        "denied": "Application denied",
        "denial": "Application denied",
        "withdrawn": "Application withdrawn by applicant",
        "closed for incompleteness": "File closed for incompleteness",
    },
    "outcome": {
        "originated": "Loan originated",
        "loan originated": "Loan originated",
        "denied": "Application denied",
        "denial": "Application denied",
        "withdrawn": "Application withdrawn by applicant",
        "closed for incompleteness": "File closed for incompleteness",
    },
    "age": {
        "under 25": "<25",
        "younger than 25": "<25",
        "less than 25": "<25",
        "25 to 34": "25-34",
        "25-34": "25-34",
        "25 through 34": "25-34",
        "35 to 44": "35-44",
        "35-44": "35-44",
    },
}


def _key(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _variable_id(
    concept: ExtractedConcept,
    semantic_router: Optional[object],
) -> tuple[str, Optional[object]]:
    """Resolve one concept to an HMDA variable ID.

    The card-backed semantic router is used when available, while curated
    aliases remain guardrails for high-risk primary-vs-co-applicant fields.
    """
    candidates = [_key(concept.canonical_hint), _key(concept.text)]
    alias_id = next(
        (_HMDA_ALIASES[candidate] for candidate in candidates
         if candidate in _HMDA_ALIASES),
        None,
    )

    routed = None
    if semantic_router is not None:
        search_text = (concept.canonical_hint or concept.text).strip()
        if search_text:
            routed = semantic_router.route_dataset(
                search_text, target_dataset="hmda", top_k=10,
            )
            for target in routed.top_variables:
                if not target.target_variable_id:
                    continue
                if alias_id is None or target.target_variable_id == alias_id:
                    return target.target_variable_id, routed
            if alias_id is not None:
                return alias_id, routed

    if alias_id is not None:
        return alias_id, routed
    raise ValueError(
        f"HMDA variable could not be resolved for "
        f"{(concept.canonical_hint or concept.text)!r}"
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
    return _decoded_value_for_dimension(
        filter_item.dimension,
        raw,
    )


def _decoded_value_for_dimension(
    dimension: ExtractedConcept,
    raw: str,
) -> str:
    """Map a value using the decoded labels for a dimension."""
    dim = _key(dimension.canonical_hint or dimension.text)
    for token, mapping in _VALUE_ALIASES.items():
        if token in dim:
            return mapping.get(raw.lower(), raw)
    return raw


def _grouping_alternatives(
    analysis: ExtractedAnalysis,
    semantic_router: Optional[object],
) -> tuple[list[tuple[list[RecordFilter], str]], Optional[str]]:
    """Expand explicit grouping values into deterministic filter variants.

    The current record connector already supports filters, so a comparison
    such as Black versus White is represented as two calls rather than adding
    an LLM-generated GROUP BY clause to DuckDB.
    """
    if not analysis.groupings:
        return [([], "primary")], None

    raw_values = {
        _key(key): values
        for key, values in analysis.grouping_values.items()
    }
    dimensions: list[tuple[str, ExtractedConcept, list[str]]] = []
    for grouping in analysis.groupings:
        keys = [
            _key(grouping.canonical_hint),
            _key(grouping.text),
        ]
        values: list[str] = []
        for key in keys:
            if key and raw_values.get(key):
                values = raw_values[key]
                break
        if not values:
            return [], (
                f"grouping {grouping.text!r} has no explicit comparison "
                "values; no grouped record calls were generated"
            )
        dimensions.append((keys[0] or keys[1], grouping, values))

    variants: list[tuple[list[RecordFilter], str]] = []
    value_lists = [values for _, _, values in dimensions]
    for combination in product(*value_lists):
        filters: list[RecordFilter] = []
        labels: list[str] = []
        for (_, grouping, _), raw_value in zip(dimensions, combination):
            variable, _ = _variable_id(grouping, semantic_router)
            decoded = _decoded_value_for_dimension(grouping, str(raw_value))
            filters.append(RecordFilter(
                variable_id=variable,
                operator="equals",
                value=decoded,
            ))
            labels.append(f"{grouping.text}={decoded}")
        variants.append((filters, "group_" + ";".join(labels)))
    return variants, None


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


def _admin_place_tract_prefixes(
    geo: ResolvedGeography,
    geo_db: Optional[sqlite3.Connection],
) -> list[str]:
    """Expand a GA admin place to tracts for record-level filtering only."""
    if (
        geo_db is None
        or geo.geo_level != "place"
        or not geo.geo_id.startswith("13")
    ):
        return []
    rows = geo_db.execute(
        """
        SELECT t.geoid AS tract_geoid
        FROM admin_geographies AS t
        JOIN admin_geographies AS p
          ON p.geoid = ?
        WHERE t.geo_type = 'tract'
          AND t.state_fips = p.state_fips
          AND MbrIntersects(t.geom, p.geom)
          AND ST_Intersects(t.geom, p.geom)
        ORDER BY t.geoid
        """,
        (geo.geo_id,),
    ).fetchall()
    return [str(row["tract_geoid"]) for row in rows]


def _record_geo_prefixes(
    geo: ResolvedGeography,
    geo_db: Optional[sqlite3.Connection],
) -> list[str]:
    """Return geography filters for record-level data without changing Census."""
    prefixes = _geo_prefixes(geo)
    if prefixes:
        return prefixes
    return _admin_place_tract_prefixes(geo, geo_db)


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


def _rate_concept_label(numerator_filters: list[RecordFilter]) -> Optional[str]:
    for filter_item in numerator_filters:
        value = str(filter_item.value or "").strip().lower()
        if value == "application denied":
            return "mortgage denial rate"
        if value == "loan originated":
            return "mortgage approval rate"
    return None


def _record_supported_years(
    intent: ExtractedIntent,
    metadata_db: Optional[sqlite3.Connection],
    table_id: str,
    dataset: str,
) -> list[int]:
    supported: list[int] = []
    if metadata_db is not None:
        try:
            supported = find_supported_years(
                metadata_db, table_id, dataset, ["tract"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "record coverage lookup failed for %s/%s: %s",
                dataset, table_id, exc,
            )
    if supported:
        return supported
    fallback = list(range(2024, 2017, -1))
    if intent.years:
        oldest = min(intent.years)
        newest = max(max(intent.years), fallback[0])
        return list(range(newest, oldest - 1, -1))
    return fallback


def _pick_record_years(
    intent: ExtractedIntent,
    supported_years: list[int],
) -> list[int]:
    if not supported_years:
        return []
    if intent.years and intent.temporal_intent not in {"change", "trend"}:
        wanted = sorted(set(intent.years))
        return [year for year in wanted if year in supported_years] or wanted
    return _pick_years(intent, supported_years)


def plan_record_query(
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    *,
    semantic_router: Optional[object],
    table_id: str = "hmda",
    dataset: str = "hmda",
    file_glob: str = "*.parquet",
    record_id_column: str = "record_id",
    geo_db: Optional[sqlite3.Connection] = None,
    metadata_db: Optional[sqlite3.Connection] = None,
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
        return PlanResult(
            intent=intent,
            resolved_geos=resolved_geos,
            concept_resolutions=[],
            calls=[],
            notes=[
                "record-level analysis detected, but no geography was "
                "resolved; provide a state, county, tract, city, or another "
                "supported area",
            ],
        )

    concepts = list(intent.concepts)
    resolutions: list[ConceptResolution] = []
    calls: list[PlannedCall] = []
    notes: list[str] = []
    supported_years = _record_supported_years(
        intent, metadata_db, table_id, dataset,
    )
    years = _pick_record_years(intent, supported_years) or (list(intent.years) or [2024])

    for analysis in analyses:
        if analysis.measure is None:
            notes.append("record analysis has no measure; skipped")
            continue
        if analysis.operation not in {
            "value", "count", "sum", "average", "median", "percentage",
        }:
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
        numerator_filters: list[RecordFilter] = []
        for filter_item in analysis.filters:
            filter_id, _ = _variable_id(
                filter_item.dimension, semantic_router,
            )
            if filter_item.operator in {"is_null", "is_not_null"}:
                record_filter = RecordFilter(
                    variable_id=filter_id,
                    operator=filter_item.operator,
                )
            else:
                value = _decoded_value(filter_item)
                if value == "":
                    raise ValueError(
                        f"Filter {filter_item.dimension.text!r} has no value"
                    )
                record_filter = RecordFilter(
                    variable_id=filter_id,
                    operator=filter_item.operator,
                    value=value,
                )
            if analysis.operation == "percentage" and filter_id == measure_id:
                numerator_filters.append(record_filter)
            else:
                record_filters.append(record_filter)

        if analysis.operation == "percentage" and not numerator_filters:
            notes.append(
                f"percentage analysis for {measure.text!r} has no "
                "numerator condition; skipped"
            )
            continue
        if analysis.operation == "percentage":
            label = _rate_concept_label(numerator_filters)
            if label:
                concepts[concept_idx] = measure.model_copy(update={
                    "text": label,
                    "canonical_hint": label,
                })

        grouping_variants, grouping_note = _grouping_alternatives(
            analysis, semantic_router,
        )
        if grouping_note:
            notes.append(grouping_note)
            continue

        for year in years:
            for geo_idx, geo in enumerate(resolved_geos):
                geo_prefixes = _record_geo_prefixes(geo, geo_db)
                for grouping_filters, role in grouping_variants:
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
                        record_filters=(
                            list(record_filters) + list(grouping_filters)
                        ),
                        record_numerator_filters=list(numerator_filters),
                    )
                    variables = (
                        ConceptVariables(
                            numerator="__record_numerator__",
                            denominator="__record_denominator__",
                        )
                        if analysis.operation == "percentage"
                        else ConceptVariables(value=measure_id)
                    )
                    calls.append(PlannedCall(
                        api_call=api_call,
                        geo_idx=geo_idx,
                        concept_idx=concept_idx,
                        year=int(year),
                        role=role,
                        operation=analysis.operation,
                        variables=variables,
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
