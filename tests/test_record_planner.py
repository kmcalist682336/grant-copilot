"""Focused tests for record-level planning behavior."""
from __future__ import annotations

import pytest

from scripts.chatbot.models import (
    ExtractedAnalysis,
    ExtractedConcept,
    ExtractedFilter,
    ExtractedIntent,
)
from scripts.chatbot.record_metric_map import load_default_record_metric_map
from scripts.chatbot.record_planner import (
    _pick_record_years,
    _record_supported_years,
    plan_record_query,
    record_metric_recipe_for_analysis,
)


def _denial_rate_intent(**updates) -> ExtractedIntent:
    values = {
        "analyses": [
            ExtractedAnalysis(
                operation="percentage",
                measure=ExtractedConcept(
                    text="denial rate",
                    canonical_hint="mortgage denial rate",
                    dataset_hint="hmda",
                ),
            ),
        ],
        "temporal_intent": "latest",
    }
    values.update(updates)
    return ExtractedIntent(**values)


ACTION_STATUS = "906bb78b0f70"
COMPLETED_STATUSES = [
    "Loan originated",
    "Application approved but not accepted",
    "Application denied",
]
APPLICATION_STATUSES = [
    *COMPLETED_STATUSES,
    "Application withdrawn by applicant",
    "File closed for incompleteness",
]
PREAPPROVAL_STATUSES = [
    "Preapproval request denied",
    "Preapproval request approved but not accepted",
]


@pytest.mark.parametrize(
    ("metric", "denominator_statuses", "numerator_status"),
    [
        (
            "mortgage origination rate among completed applications",
            COMPLETED_STATUSES,
            "Loan originated",
        ),
        (
            "mortgage withdrawal rate",
            APPLICATION_STATUSES,
            "Application withdrawn by applicant",
        ),
        (
            "incomplete mortgage application rate",
            APPLICATION_STATUSES,
            "File closed for incompleteness",
        ),
        (
            "preapproval denial rate",
            PREAPPROVAL_STATUSES,
            "Preapproval request denied",
        ),
    ],
)
def test_curated_rates_define_exact_denominator_universes(
    metric, denominator_statuses, numerator_status,
):
    recipe = load_default_record_metric_map().lookup(metric)

    assert recipe is not None
    assert recipe.operation == "percentage"
    assert len(recipe.record_filters) == 1
    assert recipe.record_filters[0].variable_id == ACTION_STATUS
    assert recipe.record_filters[0].operator == "in"
    assert recipe.record_filters[0].value == denominator_statuses
    assert len(recipe.numerator_filters) == 1
    assert recipe.numerator_filters[0].variable_id == ACTION_STATUS
    assert recipe.numerator_filters[0].operator == "equals"
    assert recipe.numerator_filters[0].value == numerator_status


def test_application_count_excludes_purchases_and_preapprovals():
    recipe = load_default_record_metric_map().lookup(
        "mortgage application count",
    )

    assert recipe is not None
    assert recipe.operation == "count"
    assert len(recipe.record_filters) == 1
    assert recipe.record_filters[0].operator == "in"
    assert recipe.record_filters[0].value == APPLICATION_STATUSES
    assert "Purchased loan" not in recipe.record_filters[0].value
    assert not set(PREAPPROVAL_STATUSES) & set(recipe.record_filters[0].value)


def test_legacy_origination_alias_uses_completed_application_recipe():
    recipe = load_default_record_metric_map().lookup(
        "mortgage origination rate",
    )

    assert recipe is not None
    assert recipe.canonical == (
        "mortgage origination rate among completed applications"
    )


@pytest.mark.parametrize(
    ("metric", "operation", "denominator_statuses"),
    [
        (
            "mortgage origination rate",
            "percentage",
            COMPLETED_STATUSES,
        ),
        (
            "mortgage withdrawal rate",
            "percentage",
            APPLICATION_STATUSES,
        ),
        (
            "incomplete mortgage application rate",
            "percentage",
            APPLICATION_STATUSES,
        ),
        (
            "preapproval denial rate",
            "percentage",
            PREAPPROVAL_STATUSES,
        ),
        (
            "mortgage application count",
            "count",
            APPLICATION_STATUSES,
        ),
    ],
)
def test_planner_propagates_curated_metric_universe(
    metric, operation, denominator_statuses,
):
    intent = ExtractedIntent(
        analyses=[
            ExtractedAnalysis(
                operation=operation,
                measure=ExtractedConcept(
                    text=metric,
                    canonical_hint=metric,
                    dataset_hint="hmda",
                ),
            ),
        ],
        years=[2024],
    )

    plan = plan_record_query(
        intent,
        [],
        dataset="hmda", semantic_router=None,
        metadata_db=None,
        trend_lookback_years=0,
    )

    assert len(plan.calls) == 1
    assert len(plan.calls[0].api_call.record_filters) == 1
    denominator_filter = plan.calls[0].api_call.record_filters[0]
    assert denominator_filter.variable_id == ACTION_STATUS
    assert denominator_filter.operator == "in"
    assert denominator_filter.value == denominator_statuses


@pytest.mark.parametrize(
    ("status", "canonical"),
    [
        (
            "File closed for incompleteness",
            "incomplete mortgage application rate",
        ),
        ("Preapproval request denied", "preapproval denial rate"),
    ],
)
def test_status_shaped_rate_analysis_recovers_curated_recipe(
    status, canonical,
):
    analysis = ExtractedAnalysis(
        operation="percentage",
        measure=ExtractedConcept(
            text="application status",
            canonical_hint="application status",
            dataset_hint="hmda",
        ),
        filters=[
            ExtractedFilter(
                dimension=ExtractedConcept(
                    text="application status",
                    canonical_hint="application status",
                    dataset_hint="hmda",
                ),
                operator="equals",
                value_text=status,
                normalized_value_hint=status,
            ),
        ],
    )

    recipe = record_metric_recipe_for_analysis(analysis, dataset="hmda")

    assert recipe is not None
    assert recipe.canonical == canonical


def test_latest_record_metric_selects_supported_prior_year():
    intent = _denial_rate_intent()

    years = _pick_record_years(
        intent,
        [2024, 2023, 2022, 2021, 2020],
        lookback_years=3,
    )

    assert years == [2021, 2024]


def test_explicit_record_year_does_not_add_comparison():
    intent = _denial_rate_intent(years=[2022])

    years = _pick_record_years(
        intent,
        [2024, 2023, 2022, 2021, 2020],
        lookback_years=3,
    )

    assert years == [2022]


def test_record_prior_call_preserves_metric_recipe_and_scope():
    plan = plan_record_query(
        _denial_rate_intent(),
        [],
        dataset="hmda", semantic_router=None,
        metadata_db=None,
        trend_lookback_years=3,
    )

    assert [(call.year, call.role) for call in plan.calls] == [
        (2021, "prior_period"),
        (2024, "primary"),
    ]
    prior, latest = plan.calls
    assert prior.operation == latest.operation == "percentage"
    assert prior.api_call.geo_prefixes == latest.api_call.geo_prefixes
    assert prior.api_call.record_filters == latest.api_call.record_filters
    assert (
        prior.api_call.record_numerator_filters
        == latest.api_call.record_numerator_filters
    )
    assert prior.variables == latest.variables


def test_record_trend_expansion_can_be_disabled():
    plan = plan_record_query(
        _denial_rate_intent(),
        [],
        dataset="hmda", semantic_router=None,
        metadata_db=None,
        trend_lookback_years=0,
    )

    assert [(call.year, call.role) for call in plan.calls] == [
        (2024, "primary"),
    ]


def test_latest_record_metric_uses_years_advertised_by_metadata(metadata_db):
    intent = _denial_rate_intent()
    supported = _record_supported_years(intent, metadata_db, "hmda", "hmda")

    plan = plan_record_query(
        intent,
        [],
        dataset="hmda", semantic_router=None,
        metadata_db=metadata_db,
        trend_lookback_years=3,
    )

    planned_years = [call.year for call in plan.calls]
    assert planned_years[-1] == max(supported)
    assert set(planned_years).issubset(set(supported))
    if len(set(supported)) > 1:
        assert plan.calls[0].role == "prior_period"
