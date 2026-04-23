"""Tests for MagnitudeContextualizer."""
from __future__ import annotations

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedGeoRef, ResolvedGeography,
)
from scripts.chatbot.nodes.magnitude_contextualizer import (
    MagnitudeFraming, contextualize_magnitudes,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _geo(name: str, level: str = "neighborhood") -> ResolvedGeography:
    return ResolvedGeography(
        geo_id="x",
        geo_level=level,          # type: ignore[arg-type]
        geo_type=level,
        display_name=name,
        api_for_clause="for=x",
        confidence=1.0,
        data_level_available="tract",
        source_ref=ExtractedGeoRef(text=name, ref_type="administrative"),
    )


def _concept(text: str, hint: str | None = None) -> ExtractedConcept:
    return ExtractedConcept(text=text, canonical_hint=hint)


def _av(
    *, geo: ResolvedGeography, concept: ExtractedConcept, year: int,
    role: str, value: float | None = None, ratio: float | None = None,
) -> AggregatedValue:
    return AggregatedValue(
        geo=geo, concept=concept, year=year, role=role,
        value=value, ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Primary value + full comparator chain
# ---------------------------------------------------------------------------

def test_framing_computes_ratios_against_every_comparator():
    primary_geo = _geo("Buckhead")
    county = _geo("Fulton County", "county")
    msa = _geo("Atlanta MSA", "msa")
    state = _geo("Georgia", "state")
    us = _geo("United States", "us")
    c = _concept("median income", "median household income")
    values = [
        _av(geo=primary_geo, concept=c, year=2024, role="primary", value=140_000),
        _av(geo=county, concept=c, year=2024, role="comparator.county", value=95_000),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", value=90_000),
        _av(geo=state, concept=c, year=2024, role="comparator.state", value=77_000),
        _av(geo=us, concept=c, year=2024, role="comparator.us", value=81_000),
    ]
    res = AggregatedResult(values=values)
    framings = contextualize_magnitudes(res)
    assert len(framings) == 1
    f = framings[0]
    assert f.geo_display_name == "Buckhead"
    assert f.value == 140_000
    assert f.vs_county is not None
    assert abs(f.vs_county - 140_000 / 95_000) < 1e-6
    assert f.vs_msa is not None
    assert abs(f.vs_msa - 140_000 / 90_000) < 1e-6
    assert f.vs_state is not None
    assert f.vs_us is not None
    # Buckhead >>> all comparators → very_high
    assert f.framing_label in ("high", "very_high")
    # framing_phrase should mention the metro ratio (strongest
    # comparator preference)
    assert "metro" in f.framing_phrase


def test_framing_trend_from_prior_period():
    g = _geo("Buckhead")
    c = _concept("median income", "median household income")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=140_000),
        _av(geo=g, concept=c, year=2021, role="prior_period", value=130_000),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.prior_period_year == 2021
    assert f.prior_period_value == 130_000
    assert f.trend_delta == 10_000
    assert f.trend_pct is not None
    assert abs(f.trend_pct - (10_000 / 130_000)) < 1e-6
    assert f.trend_label in ("rising_mild", "rising_strong")
    assert "up" in f.framing_phrase and "2021" in f.framing_phrase


def test_framing_falling_trend():
    g = _geo("Buckhead")
    c = _concept("poverty rate", "poverty rate")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=0.18),
        _av(geo=g, concept=c, year=2021, role="prior_period", value=0.24),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.trend_pct is not None and f.trend_pct < 0
    assert f.trend_label in ("falling_mild", "falling_strong")
    assert "down" in f.framing_phrase


def test_framing_missing_comparators_ok():
    g = _geo("Buckhead")
    c = _concept("odd metric")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=42),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.vs_msa is None and f.vs_state is None
    assert f.framing_label == "unknown"
    assert "no comparator" in f.framing_phrase


def test_framing_typical_bucket_when_close_to_comparator():
    g = _geo("Buckhead")
    msa = _geo("Atlanta MSA", "msa")
    c = _concept("poverty rate", "poverty rate")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=0.13),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", value=0.12),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.vs_msa is not None
    assert 1.0 < f.vs_msa < 1.15
    assert f.framing_label == "typical"
    assert "in line" in f.framing_phrase


def test_framing_uses_ratio_field_when_value_is_none():
    """poverty rate returned as ratio field — should still pick up."""
    g = _geo("Buckhead")
    c = _concept("poverty rate", "poverty rate")
    msa = _geo("Atlanta MSA", "msa")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", ratio=0.234),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", ratio=0.109),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.value == 0.234
    assert f.value_kind == "ratio"
    assert f.vs_msa is not None
    assert abs(f.vs_msa - 0.234 / 0.109) < 1e-6


def test_framing_multi_concept_grouping():
    """Two primaries, two concepts — each gets its own framing with
    its own comparators."""
    g = _geo("Buckhead")
    msa = _geo("Atlanta MSA", "msa")
    c1 = _concept("median income", "median household income")
    c2 = _concept("poverty rate", "poverty rate")
    values = [
        _av(geo=g, concept=c1, year=2024, role="primary", value=140_000),
        _av(geo=msa, concept=c1, year=2024, role="comparator.msa", value=90_000),
        _av(geo=g, concept=c2, year=2024, role="primary", value=0.08),
        _av(geo=msa, concept=c2, year=2024, role="comparator.msa", value=0.12),
    ]
    framings = contextualize_magnitudes(AggregatedResult(values=values))
    assert len(framings) == 2
    by_concept = {f.concept_text: f for f in framings}
    # Income primary high vs MSA
    assert by_concept["median income"].vs_msa is not None
    assert by_concept["median income"].vs_msa > 1.4
    # Poverty primary low vs MSA (0.08 / 0.12 ≈ 0.67)
    assert by_concept["poverty rate"].vs_msa is not None
    assert by_concept["poverty rate"].vs_msa < 0.85


def test_framing_empty_aggregated_returns_empty():
    framings = contextualize_magnitudes(AggregatedResult(values=[]))
    assert framings == []


def test_framing_comparison_summary_populated():
    g = _geo("Buckhead")
    msa = _geo("Atlanta MSA", "msa")
    c = _concept("median income", "median household income")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=140_000),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", value=90_000),
        _av(geo=g, concept=c, year=2021, role="prior_period", value=130_000),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    joined = " | ".join(f.comparison_summary)
    assert "metro: 90,000" in joined
    assert "prior (2021): 130,000" in joined


def test_framing_prior_with_zero_value_skips_trend():
    """Division by zero safety — just don't populate the ratio."""
    g = _geo("Buckhead")
    c = _concept("foo")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=42),
        _av(geo=g, concept=c, year=2021, role="prior_period", value=0),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.trend_pct is None
    assert f.trend_label == "unknown"


# ---------------------------------------------------------------------------
# Count-vs-rate scale guard — suppresses obvious population-scale ratios
# so the synthesizer doesn't emit nonsense like "0.00× the national average"
# when the primary is a count at a small geo and the comparator is a count
# at a much larger geo.
# ---------------------------------------------------------------------------

def test_count_vs_rate_guard_suppresses_tiny_ratio():
    """SW Atlanta SNAP count (~4k) vs US SNAP count (~18M) gives
    a ratio of ~2e-4. That's a population-scale artifact, not a rate
    signal — guard must drop it."""
    primary_geo = _geo("sw side of Atlanta city")
    msa = _geo("Atlanta MSA", "msa")
    state = _geo("Georgia", "state")
    us = _geo("United States", "us")
    c = _concept("SNAP households")
    values = [
        _av(geo=primary_geo, concept=c, year=2024, role="primary", value=4_200),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", value=250_000),
        _av(geo=state, concept=c, year=2024, role="comparator.state", value=620_000),
        _av(geo=us, concept=c, year=2024, role="comparator.us", value=18_000_000),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    # All three comparator ratios land outside [0.10, 10.0] → suppressed
    assert f.vs_msa is None
    assert f.vs_state is None
    assert f.vs_us is None
    # Raw comparator values are still retained (so the synthesizer can
    # mention them; they just don't get a "0.00×" framing).
    assert f.msa_value == 250_000
    assert f.state_value == 620_000
    assert f.us_value == 18_000_000
    # No plausible ratio → label degrades to unknown, phrase degrades
    # to the no-context message.
    assert f.framing_label == "unknown"
    assert f.framing_phrase == "no comparator or trend context available"


def test_count_vs_rate_guard_preserves_plausible_ratios():
    """A meaningful rate ratio (0.5×, 0.7×, 0.9×) must NOT be suppressed."""
    primary_geo = _geo("Buckhead")
    msa = _geo("Atlanta MSA", "msa")
    state = _geo("Georgia", "state")
    us = _geo("United States", "us")
    c = _concept("poverty rate")
    values = [
        _av(geo=primary_geo, concept=c, year=2024, role="primary", ratio=0.076),
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", ratio=0.110),
        _av(geo=state, concept=c, year=2024, role="comparator.state", ratio=0.134),
        _av(geo=us, concept=c, year=2024, role="comparator.us", ratio=0.125),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    # 0.076/0.110 ≈ 0.69 — in the plausibility window, kept.
    assert f.vs_msa is not None
    assert 0.6 < f.vs_msa < 0.8
    assert f.vs_state is not None
    assert f.vs_us is not None
    assert f.framing_label in ("low", "very_low")


def test_count_vs_rate_guard_trend_untouched():
    """Trend compares same geo across years — the plausibility window
    doesn't apply. Big year-over-year value swings still emit trend_pct
    normally (the trend_break AnomalyDetector threshold is the right
    place to flag those)."""
    g = _geo("Test County", "county")
    c = _concept("veterans")
    values = [
        # A 10× jump between years is implausible as a rate-delta but
        # is what raw-count queries can show on small geos; the trend
        # guard must not silently drop it.
        _av(geo=g, concept=c, year=2024, role="primary", value=5_000),
        _av(geo=g, concept=c, year=2021, role="prior_period", value=500),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.trend_pct == 9.0           # (5000 - 500) / 500
    assert f.prior_period_year == 2021


def test_count_vs_rate_guard_boundary_values():
    """Ratios exactly at the window boundary (0.10 and 10.0) are kept;
    anything outside is dropped."""
    g = _geo("Test")
    msa = _geo("MSA", "msa")
    state = _geo("ST", "state")
    c = _concept("x")
    values = [
        _av(geo=g, concept=c, year=2024, role="primary", value=10),
        # 10 / 100 = 0.10 exactly → keep
        _av(geo=msa, concept=c, year=2024, role="comparator.msa", value=100),
        # 10 / 101 ≈ 0.099 → drop (below 0.10)
        _av(geo=state, concept=c, year=2024, role="comparator.state", value=101),
    ]
    f = contextualize_magnitudes(AggregatedResult(values=values))[0]
    assert f.vs_msa is not None and abs(f.vs_msa - 0.10) < 1e-9
    assert f.vs_state is None
