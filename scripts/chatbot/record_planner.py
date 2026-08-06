"""Deterministic planner for record-level datasets.

This module is intentionally small and dataset-agnostic at its public
boundary.  HMDA is the first record-level dataset, so its high-value variable
aliases live here as guardrails.  Categorical filter values are resolved
through the record value registry after variable routing.  The LLM supplies
*which* dimensions and values were explicitly requested; this module chooses
variables, normalizes allowed filter values, builds a structured
``APIPlanCall``, and leaves SQL generation to ``DuckDBCaller``.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from itertools import product
from typing import Any, Optional

from scripts.chatbot.census_caller import APIPlanCall, RecordFilter
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.models import (
    ExtractedAnalysis, ExtractedConcept, ExtractedGeoRef, ExtractedIntent,
    ExtractedFilter, ResolvedGeography,
)
from scripts.chatbot.metadata_search import find_supported_years
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, PlannedCall, _pick_years,
)
from scripts.chatbot.record_metric_map import (
    RecordMetricMap, load_default_record_metric_map,
)
from scripts.chatbot.record_variable_aliases import resolve_record_variable_alias
from scripts.chatbot.record_value_registry import resolve_record_filter_value

logger = logging.getLogger(__name__)


def _default_record_geography(dataset: str) -> Optional[ResolvedGeography]:
    """Default geography for record datasets when the user omits one.

    The current HMDA record store is a Georgia-focused demo extract.  Rather
    than returning zero planned calls for "average loan amount for Asian male
    applicants", default to Georgia statewide and keep the assumption in the
    plan notes so synthesis can disclose it.
    """
    if dataset != "hmda":
        return None
    source_ref = ExtractedGeoRef(
        text="Georgia",
        ref_type="administrative",
    )
    return ResolvedGeography(
        geo_id="13",
        geo_level="state",
        geo_type="state",
        display_name="Georgia",
        tract_geoids=[],
        county_geoid=None,
        api_for_clause="state:13",
        api_in_clause="",
        confidence=0.80,
        assumption_notes=[
            "No geography was specified; defaulted HMDA record query to "
            "Georgia statewide because the local HMDA record store is "
            "Georgia-focused.",
        ],
        data_level_available="state",
        source_ref=source_ref,
    )


def _key(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def _record_filter_signature(filter_item: RecordFilter) -> tuple[str, str, str]:
    return (
        filter_item.variable_id,
        filter_item.operator,
        repr(filter_item.value),
    )


def _metric_recipe_for_analysis(
    analysis: ExtractedAnalysis,
    record_metric_map: RecordMetricMap,
    *,
    dataset: str,
    table_id: str,
):
    if analysis.measure is None:
        return None
    measure_lookup_texts: list[Optional[str]] = [
        analysis.measure.canonical_hint,
        analysis.measure.text,
    ]
    operation_prefixes = {
        "average": "average",
        "median": "median",
        "sum": "total",
        "count": "count of",
    }
    prefix = operation_prefixes.get(analysis.operation)
    if prefix:
        for text in (analysis.measure.canonical_hint, analysis.measure.text):
            if text:
                measure_lookup_texts.append(f"{prefix} {text}")
    status_lookup_texts: list[Optional[str]] = []
    if analysis.operation == "percentage":
        for filter_item in analysis.filters:
            dimension_text = _key(
                f"{filter_item.dimension.text} "
                f"{filter_item.dimension.canonical_hint or ''}"
            )
            if any(
                token in dimension_text
                for token in {"status", "outcome", "action taken"}
            ):
                status_variable_id = resolve_record_variable_alias(
                    dataset=dataset,
                    table_id=table_id,
                    texts=[
                        filter_item.dimension.canonical_hint,
                        filter_item.dimension.text,
                    ],
                )
                raw_value = (
                    filter_item.normalized_value_hint
                    or filter_item.value_text
                    or ""
                )
                status_lookup_texts.extend([
                    filter_item.value_text,
                    filter_item.normalized_value_hint,
                    f"{raw_value} rate",
                    f"mortgage {raw_value} rate",
                    f"application {raw_value} rate",
                ])
                normalized_value = _decoded_value(filter_item)
                if status_variable_id is not None:
                    normalized_value = resolve_record_filter_value(
                        dataset=dataset,
                        table_id=table_id,
                        variable_id=status_variable_id,
                        raw_value=normalized_value,
                    )
                values = (
                    normalized_value
                    if isinstance(normalized_value, list)
                    else [normalized_value]
                )
                value_set = {str(value).strip().lower() for value in values}
                if value_set & {"application denied", "denied", "denial"}:
                    status_lookup_texts.append("mortgage denial rate")
                if (
                    "loan originated" in value_set
                    and "application approved but not accepted" in value_set
                ):
                    status_lookup_texts.append("mortgage approval rate")
                elif value_set & {"loan originated", "originated"}:
                    status_lookup_texts.append("mortgage origination rate")
                if value_set & {
                    "application withdrawn by applicant",
                    "withdrawn",
                }:
                    status_lookup_texts.append("mortgage withdrawal rate")
    # For percentage/rate questions, outcome/status filters define the metric.
    # Put them before generic measure text such as "loan applications" so
    # "approval rate of mortgage applications" cannot be swallowed by the
    # broader "mortgage application count" recipe.
    lookup_texts = status_lookup_texts + measure_lookup_texts
    recipe = record_metric_map.lookup_any(lookup_texts)
    if recipe is None:
        return None
    if recipe.dataset != dataset or recipe.table_id != table_id:
        return None
    return recipe


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
    alias_id = resolve_record_variable_alias(
        dataset=dataset,
        table_id=table_id,
        texts=[concept.canonical_hint, concept.text],
    )

    routed = None
    if semantic_router is not None:
        search_text = (concept.canonical_hint or concept.text).strip()
        if search_text:
            routed = semantic_router.route_dataset(
                search_text, target_dataset=dataset, top_k=10,
            )
            for target in routed.top_variables:
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


def _decoded_value(filter_item: ExtractedFilter) -> Any:
    """Parse extracted scalar/list values without dataset-specific aliases.

    LLM extraction sometimes returns list-valued normalized hints as a JSON
    string (for example ``["Loan originated", "Application approved but not
    accepted"]``).  Normalize that before building RecordFilter objects so
    DuckDB sees a real iterable for IN predicates, not one literal string.
    """
    raw_value = (
        filter_item.normalized_value_hint
        or filter_item.value_text
        or ""
    )
    if isinstance(raw_value, list):
        return [str(value).strip() for value in raw_value]
    raw = str(raw_value).strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(value).strip() for value in parsed]
    return raw


def _decoded_filter_value(
    filter_item: ExtractedFilter,
    *,
    dataset: str,
    table_id: str,
    variable_id: str,
) -> Any:
    """Resolve a filter value using the selected record variable as the key."""
    legacy_value = _decoded_value(filter_item)
    return resolve_record_filter_value(
        dataset=dataset,
        table_id=table_id,
        variable_id=variable_id,
        raw_value=legacy_value,
    )


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
            decoded = resolve_record_filter_value(
                dataset=dataset,
                table_id=table_id,
                variable_id=variable,
                raw_value=str(raw_value).strip(),
            )
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
    """Expand a GA admin place to tracts for record-level filtering only.

    Prefer the gazetteer's curated ``admin_place_tract_map``.  That is the
    same lookup used by validation/debug queries and avoids subtle differences
    from ad-hoc geometry intersections at query time.  The spatial fallback is
    retained for older gazetteers that do not have the mapping table.
    """
    if (
        geo_db is None
        or geo.geo_level != "place"
        or not geo.geo_id.startswith("13")
    ):
        return []
    try:
        rows = geo_db.execute(
            """
            SELECT tract_geoid
            FROM admin_place_tract_map
            WHERE admin_geoid = ?
            ORDER BY tract_geoid
            """,
            (geo.geo_id,),
        ).fetchall()
        mapped = [str(row["tract_geoid"]) for row in rows]
        if mapped:
            return mapped
    except sqlite3.Error:
        logger.debug(
            "admin_place_tract_map unavailable; falling back to spatial "
            "place/tract intersection",
            exc_info=True,
        )
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


def record_analyses(intent: ExtractedIntent) -> list[ExtractedAnalysis]:
    """Public wrapper for the record-analysis selector.

    Orchestration code needs to make the same Census-vs-record decision as the
    planner before deciding whether an optional record-only LLM pass is worth
    running.  Keep that definition centralized here so the fast path and the
    actual planner cannot drift apart.
    """
    return _record_analysis(intent)


def has_record_analysis(intent: ExtractedIntent) -> bool:
    return bool(_record_analysis(intent))


def record_metric_recipe_for_analysis(
    analysis: ExtractedAnalysis,
    *,
    dataset: str = "hmda",
    table_id: str = "hmda",
    record_metric_map: Optional[RecordMetricMap] = None,
):
    """Return the deterministic record metric recipe for an analysis, if any."""
    metric_map = record_metric_map or load_default_record_metric_map()
    return _metric_recipe_for_analysis(
        analysis,
        metric_map,
        dataset=dataset,
        table_id=table_id,
    )


def _rate_concept_label(numerator_filters: list[RecordFilter]) -> Optional[str]:
    for filter_item in numerator_filters:
        raw_value = filter_item.value
        values = (
            [str(value).strip().lower() for value in raw_value]
            if isinstance(raw_value, list)
            else [str(raw_value or "").strip().lower()]
        )
        if "application denied" in values:
            return "mortgage denial rate"
        if (
            "loan originated" in values
            and "application approved but not accepted" in values
        ):
            return "mortgage approval rate"
        if "loan originated" in values:
            return "mortgage origination rate"
    return None


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


def _is_ambiguous_record_filter(filter_item: ExtractedFilter) -> bool:
    dim = _key(filter_item.dimension.canonical_hint or filter_item.dimension.text)
    value = _key(filter_item.normalized_value_hint or filter_item.value_text)
    return "age" in dim and value in {
        "middle aged", "middle-aged", "working age", "working-age",
        "adult", "adults",
    }




def _is_rate_like_record_measure(concept: ExtractedConcept) -> bool:
    """True when the requested measure is a rate over an outcome column."""
    text = _key(f"{concept.text} {concept.canonical_hint or ''}")
    return any(
        phrase in text
        for phrase in {
            "denial rate", "approval rate", "origination rate",
            "application denial", "application approval",
            "loan denial", "loan approval",
            "denied applications", "approved applications",
            "originated applications",
        }
    )


def _is_record_status_dimension(concept: ExtractedConcept) -> bool:
    """True for HMDA action/status/outcome dimensions."""
    text = _key(f"{concept.text} {concept.canonical_hint or ''}")
    return any(
        phrase in text
        for phrase in {
            "action taken", "application status", "loan application status",
            "mortgage application outcome", "application outcome",
            "loan outcome", "status", "outcome",
        }
    )


def _percentage_measure_dimension(
    analysis: ExtractedAnalysis,
) -> ExtractedConcept:
    """Choose the real record column for percentage/rate calculations.

    A user-facing measure such as "denial rate" is not itself an HMDA
    column. The actual column is the application status/action_taken
    variable, while the requested outcome (for example "Application denied")
    is the numerator condition.
    """
    measure = analysis.measure
    if measure is None:
        raise ValueError("record percentage analysis has no measure")
    if analysis.operation != "percentage":
        return measure
    if _is_record_status_dimension(measure):
        return measure
    if not _is_rate_like_record_measure(measure):
        return measure
    for filter_item in analysis.filters:
        if _is_record_status_dimension(filter_item.dimension):
            return filter_item.dimension
    return measure

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


def _record_year_role(
    *,
    base_role: str,
    year: int,
    years: list[int],
    temporal_intent: str,
) -> str:
    """Label record calls so existing trend/context tools can read them.

    Census trend expansion emits older comparison years with
    ``role="prior_period"``.  Record plans already expand change/trend
    questions into multiple years, so mark every non-latest year the same
    way instead of inventing a second trend representation.  Grouped record
    comparisons keep their group label in the suffix, e.g.
    ``prior_period.group_applicant race=White``.
    """
    if temporal_intent not in {"change", "trend"} or len(years) < 2:
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
    table_id: str = "hmda",
    dataset: str = "hmda",
    file_glob: str = "*.parquet",
    record_id_column: str = "record_id",
    geo_db: Optional[sqlite3.Connection] = None,
    metadata_db: Optional[sqlite3.Connection] = None,
    record_metric_map: Optional[RecordMetricMap] = None,
) -> PlanResult:
    """Build a record-level plan with parameterized filter metadata.

    The returned plan uses the existing ``PlanResult``/``PlannedCall``
    contracts, so the current orchestrator, aggregator, citations, and Docker
    response mapper can consume it without a second pipeline.
    """
    analyses = _record_analysis(intent)
    metric_map = record_metric_map or load_default_record_metric_map()
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
        default_geo = _default_record_geography(dataset)
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
    years = _pick_record_years(intent, supported_years) or (list(intent.years) or [2024])

    for analysis in analyses:
        if analysis.measure is None:
            notes.append("record analysis has no measure; skipped")
            continue
        recipe = _metric_recipe_for_analysis(
            analysis, metric_map, dataset=dataset, table_id=table_id,
        )
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
            measure = _percentage_measure_dimension(analysis)
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
            if _is_ambiguous_record_filter(filter_item):
                notes.append(
                    f"skipped ambiguous record filter "
                    f"{filter_item.dimension.text!r}={filter_item.value_text!r}; "
                    "use a concrete HMDA age bin such as '<25' or '25-34'"
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
            label = recipe.canonical if recipe is not None else _rate_concept_label(numerator_filters)
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
                geo_prefixes = _record_geo_prefixes(geo, geo_db)
                for grouping_filters, role in grouping_variants:
                    planned_role = _record_year_role(
                        base_role=role,
                        year=int(year),
                        years=years,
                        temporal_intent=intent.temporal_intent,
                    )
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
