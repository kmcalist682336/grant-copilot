"""Deterministic planner for record-level datasets.

Dataset rules are supplied by the explicitly selected registered adapter.
Categorical filter values are resolved
through the record value registry after variable routing.  The LLM supplies
*which* dimensions and values were explicitly requested; this module chooses
variables, normalizes allowed filter values, builds a structured
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
    ExtractedAnalysis, ExtractedConcept, ExtractedIntent,
    ExtractedFilter, ResolvedGeography,
)
from scripts.chatbot.metadata_search import find_supported_years
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, PlannedCall, _pick_years,
)
from scripts.chatbot.record_metric_map import (
    RecordMetricMap,
)

from scripts.chatbot.record_dataset import get_record_dataset
from scripts.chatbot.record_values import _decoded_value

logger = logging.getLogger(__name__)


def _key(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _record_filter_signature(filter_item: RecordFilter) -> tuple[str, str, str]:
    return (
        filter_item.variable_id,
        filter_item.operator,
        repr(filter_item.value),
    )


def _variable_id(
    concept: ExtractedConcept,
    semantic_router: Optional[object],
    *,
    dataset: str,
    table_id: str,
) -> tuple[str, Optional[object]]:
    """Resolve one concept to a record-level variable ID.

    The card-backed semantic router is used when available, while curated
    dataset-scoped aliases remain guardrails for high-risk fields.
    """
    alias_id = get_record_dataset(dataset).variable_alias([concept.canonical_hint, concept.text])
    if alias_id is not None:
        return alias_id, None

    routed = None
    if semantic_router is not None:
        search_text = (concept.canonical_hint or concept.text).strip()
        if search_text:
            routed = semantic_router.route_dataset(
                search_text, target_dataset=dataset, top_k=10,
            )
            for target in routed.top_variables:
                if target.target_dataset != dataset:
                    continue
                if target.target_table_id and target.target_table_id != table_id:
                    continue
                if not target.target_variable_id:
                    continue
                if alias_id is None or target.target_variable_id == alias_id:
                    return target.target_variable_id, routed
            if alias_id is not None:
                return alias_id, routed

    if alias_id is not None:
        return alias_id, routed
    raise ValueError(
        f"{dataset}/{table_id} variable could not be resolved for "
        f"{(concept.canonical_hint or concept.text)!r}"
    )


def _decoded_filter_value(
    filter_item: ExtractedFilter,
    *,
    dataset: str,
    table_id: str,
    variable_id: str,
) -> Any:
    """Resolve a filter value using the selected record variable as the key."""
    legacy_value = _decoded_value(filter_item)
    return get_record_dataset(dataset).filter_value(variable_id, legacy_value)


def _grouping_alternatives(
    analysis: ExtractedAnalysis,
    semantic_router: Optional[object],
    *,
    dataset: str,
    table_id: str,
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
            variable, _ = _variable_id(
                grouping,
                semantic_router,
                dataset=dataset,
                table_id=table_id,
            )
            decoded = get_record_dataset(dataset).filter_value(variable, str(raw_value).strip())
            filters.append(RecordFilter(
                variable_id=variable,
                operator="equals",
                value=decoded,
            ))
            labels.append(f"{grouping.text}={decoded}")
        variants.append((filters, "group_" + ";".join(labels)))
    return variants, None


def _record_analysis(intent: ExtractedIntent, *, dataset: str) -> list[ExtractedAnalysis]:
    analyses: list[ExtractedAnalysis] = []
    for analysis in intent.analyses:
        if analysis.measure is not None and analysis.measure.dataset_hint not in {
            dataset, "both", "unknown",
        }:
            continue
        parts = [analysis.measure, *analysis.groupings]
        parts.extend(f.dimension for f in analysis.filters)
        if any(
            part is not None and part.dataset_hint in (dataset, "both")
            for part in parts
        ):
            analyses.append(analysis)
    return analyses


def record_analyses(intent: ExtractedIntent, *, dataset: str) -> list[ExtractedAnalysis]:
    """Public wrapper for the record-analysis selector.

    Orchestration code needs to make the same Census-vs-record decision as the
    planner before deciding whether an optional record-only LLM pass is worth
    running.  Keep that definition centralized here so the fast path and the
    actual planner cannot drift apart.
    """
    get_record_dataset(dataset)
    return _record_analysis(intent, dataset=dataset)


def has_record_analysis(intent: ExtractedIntent, *, dataset: str) -> bool:
    return bool(record_analyses(intent, dataset=dataset))


def record_metric_recipe_for_analysis(
    analysis: ExtractedAnalysis,
    *,
    dataset: str,
    table_id: Optional[str] = None,
    record_metric_map: Optional[RecordMetricMap] = None,
):
    """Return the deterministic record metric recipe for an analysis, if any."""
    adapter = get_record_dataset(dataset)
    if table_id is not None and table_id != adapter.table_id:
        raise ValueError("Selected table does not match the dataset adapter")
    metric_map = record_metric_map or adapter.recipes()
    metric_map = RecordMetricMap([r for r in metric_map.recipes
        if r.dataset == dataset and r.table_id == adapter.table_id])
    return get_record_dataset(dataset).metric_recipe(analysis, metric_map)


def _normalized_operator_value(
    operator: str,
    value: Any,
) -> tuple[str, Any]:
    """Make extracted filter operators match DuckDB's expected value shape."""
    if operator == "equals" and isinstance(value, list):
        return "in", value
    if operator == "not_equals" and isinstance(value, list):
        return "not_in", value
    if operator in {"in", "not_in"} and (
        isinstance(value, (str, bytes)) or value is None
    ):
        if isinstance(value, str) and "," in value:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if parts:
                return operator, parts
        return operator, [value]
    return operator, value


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
                metadata_db, table_id, dataset, [get_record_dataset(dataset).coverage_level],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "record coverage lookup failed for %s/%s: %s",
                dataset, table_id, exc,
            )
    if supported:
        return supported
    return sorted(set(get_record_dataset(dataset).supported_years), reverse=True)


def _pick_record_years(
    intent: ExtractedIntent,
    supported_years: list[int],
    *,
    lookback_years: int = 3,
) -> list[int]:
    if not supported_years:
        return []
    if intent.years and intent.temporal_intent not in {"change", "trend"}:
        wanted = sorted(set(intent.years))
        return [year for year in wanted if year in supported_years]
    if intent.temporal_intent == "latest":
        latest = max(supported_years)
        if lookback_years <= 0:
            return [latest]

        # Record metrics do not pass through the Census concept-rewrite
        # trend expander: their deterministic measure and filters are built
        # here.  Add the closest supported prior vintage here instead so the
        # cloned call keeps the exact same numerator, denominator, filters,
        # group, and geography scope as the latest call.
        target = latest - lookback_years
        prior_candidates = [year for year in supported_years if year <= target]
        prior = (
            max(prior_candidates)
            if prior_candidates
            else min(supported_years)
        )
        if prior == latest:
            return [latest]
        return sorted({prior, latest})
    return _pick_years(intent, supported_years)


def _record_year_role(
    *,
    base_role: str,
    year: int,
    years: list[int],
) -> str:
    """Label record calls so existing trend/context tools can read them.

    Census trend expansion emits older comparison years with
    ``role="prior_period"``. Record plans expand explicit change/trend
    questions and automatically contextualize latest-only metrics, so mark
    every non-latest year the same way instead of inventing a second trend
    representation. Grouped record comparisons keep their group label in
    the suffix, e.g.
    ``prior_period.group_applicant race=White``.
    """
    if len(years) < 2:
        return base_role
    latest_year = max(years)
    if year == latest_year:
        return base_role
    if base_role == "primary":
        return "prior_period"
    return f"prior_period.{base_role}"


def plan_record_query(
    intent: ExtractedIntent,
    resolved_geos: list[ResolvedGeography],
    *,
    semantic_router: Optional[object],
    dataset: str,
    table_id: Optional[str] = None,
    file_glob: str = "*.parquet",
    record_id_column: str = "record_id",
    geo_db: Optional[sqlite3.Connection] = None,
    metadata_db: Optional[sqlite3.Connection] = None,
    record_metric_map: Optional[RecordMetricMap] = None,
    trend_lookback_years: int = 3,
) -> PlanResult:
    """Build a record-level plan with parameterized filter metadata.

    The returned plan uses the existing ``PlanResult``/``PlannedCall``
    contracts, so the current orchestrator, aggregator, citations, and Docker
    response mapper can consume it without a second pipeline.
    """
    adapter = get_record_dataset(dataset)
    if table_id is not None and table_id != adapter.table_id:
        raise ValueError("Selected table does not match the dataset adapter")
    table_id = adapter.table_id
    analyses = _record_analysis(intent, dataset=dataset)
    metric_map = record_metric_map or adapter.recipes()
    metric_map = RecordMetricMap([r for r in metric_map.recipes
        if r.dataset == dataset and r.table_id == table_id])
    if not analyses:
        return PlanResult(
            intent=intent, resolved_geos=resolved_geos,
            concept_resolutions=[], calls=[],
            notes=["no record-level analysis was extracted"],
        )

    concepts = list(intent.concepts)
    resolutions: list[ConceptResolution] = []
    calls: list[PlannedCall] = []
    notes: list[str] = []
    if not resolved_geos:
        default_geo = adapter.default_geography()
        if default_geo is None:
            return PlanResult(
                intent=intent,
                resolved_geos=resolved_geos,
                concept_resolutions=[],
                calls=[],
                notes=[
                    "record-level analysis detected, but no geography was "
                    "resolved; provide a state, county, tract, city, or "
                    "another supported area",
                ],
            )
        resolved_geos = [default_geo]
        notes.extend(default_geo.assumption_notes)

    supported_years = _record_supported_years(
        intent, metadata_db, table_id, dataset,
    )
    years = _pick_record_years(
        intent,
        supported_years,
        lookback_years=trend_lookback_years,
    )
    if not years:
        notes.append("No supported years available for the requested record analysis")

    for analysis in analyses:
        if analysis.measure is None:
            notes.append("record analysis has no measure; skipped")
            continue
        recipe = adapter.metric_recipe(analysis, metric_map)
        operation = recipe.operation if recipe is not None else analysis.operation
        if operation not in {
            "value", "count", "sum", "average", "median", "percentage",
        }:
            notes.append(
                f"operation {analysis.operation!r} needs an explicit "
                "record aggregation recipe; no SQL was generated"
            )
            continue

        if recipe is not None:
            measure = recipe.measure_concept()
            measure_id = recipe.measure.variable_id
            measure_route = None
            notes.append(f"record metric recipe matched: {recipe.canonical}")
        else:
            measure = adapter.percentage_measure(analysis)
            measure_id, measure_route = _variable_id(
                measure,
                semantic_router,
                dataset=dataset,
                table_id=table_id,
            )
        concept_idx = len(concepts)
        concepts.append(measure)
        resolutions.append(ConceptResolution(
            concept=measure,
            tier="tier_semantic" if measure_route is not None else "tier_1_concept_map",
            routed_result=measure_route,
        ))

        record_filters: list[RecordFilter] = (
            recipe.record_filter_objects() if recipe is not None else []
        )
        numerator_filters: list[RecordFilter] = (
            recipe.numerator_filter_objects() if recipe is not None else []
        )
        record_filter_keys = {
            _record_filter_signature(item) for item in record_filters
        }
        numerator_filter_keys = {
            _record_filter_signature(item) for item in numerator_filters
        }
        for filter_item in analysis.filters:
            if adapter.ambiguous_filter(filter_item):
                notes.append(
                    f"skipped ambiguous record filter "
                    f"{filter_item.dimension.text!r}={filter_item.value_text!r}; "
                    f"{adapter.ambiguous_filter_help}"
                )
                continue
            filter_id, _ = _variable_id(
                filter_item.dimension,
                semantic_router,
                dataset=dataset,
                table_id=table_id,
            )
            if filter_item.operator in {"is_null", "is_not_null"}:
                record_filter = RecordFilter(
                    variable_id=filter_id,
                    operator=filter_item.operator,
                )
            else:
                value = _decoded_filter_value(
                    filter_item,
                    dataset=dataset,
                    table_id=table_id,
                    variable_id=filter_id,
                )
                if value == "":
                    raise ValueError(
                        f"Filter {filter_item.dimension.text!r} has no value"
                    )
                operator, value = _normalized_operator_value(
                    filter_item.operator, value,
                )
                record_filter = RecordFilter(
                    variable_id=filter_id,
                    operator=operator,
                    value=value,
                )
            if recipe is not None and operation == "percentage" and filter_id == measure_id:
                # The curated recipe owns numerator math.  Do not let an
                # extracted status/action filter accidentally restrict the
                # denominator or add a contradictory numerator predicate.
                continue
            signature = _record_filter_signature(record_filter)
            if operation == "percentage" and filter_id == measure_id:
                if signature not in numerator_filter_keys:
                    numerator_filters.append(record_filter)
                    numerator_filter_keys.add(signature)
            elif signature not in record_filter_keys:
                record_filters.append(record_filter)
                record_filter_keys.add(signature)

        if operation == "percentage" and not numerator_filters:
            notes.append(
                f"percentage analysis for {measure.text!r} has no "
                "numerator condition; skipped"
            )
            continue
        if operation == "percentage":
            label = recipe.canonical if recipe is not None else adapter.rate_label(numerator_filters)
            if label:
                concepts[concept_idx] = measure.model_copy(update={
                    "text": label,
                    "canonical_hint": label,
                })

        grouping_variants, grouping_note = _grouping_alternatives(
            analysis,
            semantic_router,
            dataset=dataset,
            table_id=table_id,
        )
        if grouping_note:
            notes.append(grouping_note)
            continue

        for year in years:
            for geo_idx, geo in enumerate(resolved_geos):
                geo_prefixes = adapter.geography_values(geo, geo_db)
                for grouping_filters, role in grouping_variants:
                    planned_role = _record_year_role(
                        base_role=role,
                        year=int(year),
                        years=years,
                    )
                    api_call = APIPlanCall(
                        url=f"record://{dataset}/{year}/{table_id}",
                        table_id=table_id,
                        variables=[measure_id],
                        geo_level="record",
                        geo_filter_ids=[],
                        geo_prefixes=geo_prefixes,
                        record_geography_column=adapter.geography_column,
                        record_geography_match=adapter.geography_match,
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
                        if operation == "percentage"
                        else ConceptVariables(value=measure_id)
                    )
                    calls.append(PlannedCall(
                        api_call=api_call,
                        geo_idx=geo_idx,
                        concept_idx=concept_idx,
                        year=int(year),
                        role=planned_role,
                        operation=operation,
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
