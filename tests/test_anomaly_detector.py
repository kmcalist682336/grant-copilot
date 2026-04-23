"""Tests for AnomalyDetector."""
from __future__ import annotations

from scripts.chatbot.nodes.anomaly_detector import (
    AnomalyFlag, detect_anomalies,
)
from scripts.chatbot.nodes.magnitude_contextualizer import (
    MagnitudeFraming,
)


def _framing(**overrides) -> MagnitudeFraming:
    """Construct a MagnitudeFraming with sensible defaults; overrides
    tweak individual fields for each test."""
    base = dict(
        geo_display_name="Buckhead",
        concept_text="median income",
        year=2024,
        value=100_000.0,
        value_kind="scalar",
    )
    base.update(overrides)
    return MagnitudeFraming(**base)


# ---------------------------------------------------------------------------
# Comparator-based flags
# ---------------------------------------------------------------------------

def test_far_above_metro_triggers_flag():
    f = _framing(vs_msa=2.5, msa_value=40_000)
    flags = detect_anomalies([f])
    assert len(flags) == 1
    assert flags[0].anomaly_kind == "far_above_comparators"
    assert flags[0].triggering_comparator == "metro"
    assert flags[0].magnitude == 2.5
    assert "far above" in flags[0].context


def test_far_below_metro_triggers_flag():
    f = _framing(vs_msa=0.4, msa_value=250_000)
    flags = detect_anomalies([f])
    assert len(flags) == 1
    assert flags[0].anomaly_kind == "far_below_comparators"
    assert "far below" in flags[0].context


def test_within_range_no_flag():
    """Ratios between 0.67 and 1.5 don't trigger a flag."""
    f = _framing(vs_msa=1.2, msa_value=83_000)
    flags = detect_anomalies([f])
    assert flags == []


def test_picks_strongest_comparator():
    """Flag magnitude reflects the comparator that deviates most."""
    # US ratio is the most extreme; should win
    f = _framing(
        vs_county=1.1,   # within range
        vs_msa=1.3,      # within range
        vs_state=1.6,    # above threshold
        vs_us=2.5,       # far above
    )
    flags = detect_anomalies([f])
    assert len(flags) == 1
    assert flags[0].triggering_comparator == "national"
    assert flags[0].magnitude == 2.5


# ---------------------------------------------------------------------------
# Trend-break flags
# ---------------------------------------------------------------------------

def test_trend_up_25_pct_flags():
    f = _framing(
        trend_pct=0.30, prior_period_year=2021,
    )
    flags = detect_anomalies([f])
    trend_flags = [x for x in flags if x.anomaly_kind == "trend_break_up"]
    assert len(trend_flags) == 1
    assert abs(trend_flags[0].magnitude - 0.30) < 1e-9
    assert "2021" in trend_flags[0].context


def test_trend_down_25_pct_flags():
    f = _framing(
        trend_pct=-0.35, prior_period_year=2020,
    )
    flags = detect_anomalies([f])
    trend_flags = [x for x in flags if x.anomaly_kind == "trend_break_down"]
    assert len(trend_flags) == 1
    assert "2020" in trend_flags[0].context


def test_mild_trend_no_flag():
    f = _framing(trend_pct=0.10, prior_period_year=2021)
    flags = detect_anomalies([f])
    assert flags == []


# ---------------------------------------------------------------------------
# Combined: one framing can produce multiple flags
# ---------------------------------------------------------------------------

def test_far_above_AND_trend_break_both_emit():
    f = _framing(
        vs_msa=2.0,                 # comparator anomaly
        trend_pct=0.40,             # trend anomaly
        prior_period_year=2021,
    )
    flags = detect_anomalies([f])
    kinds = {fl.anomaly_kind for fl in flags}
    assert kinds == {"far_above_comparators", "trend_break_up"}


# ---------------------------------------------------------------------------
# Empty + minimal inputs
# ---------------------------------------------------------------------------

def test_no_framings_returns_empty():
    assert detect_anomalies([]) == []


def test_framing_without_ratios_or_trend_returns_empty():
    f = _framing()   # no comparator ratios, no trend
    assert detect_anomalies([f]) == []


def test_zero_ratio_ignored():
    """Defensive: a comparator ratio of 0 (division edge) shouldn't
    trigger a false flag."""
    f = _framing(vs_msa=0.0, msa_value=0)
    flags = detect_anomalies([f])
    assert flags == []


# ---------------------------------------------------------------------------
# Multiple framings each get independently evaluated
# ---------------------------------------------------------------------------

def test_multiple_framings_each_independent():
    fs = [
        _framing(geo_display_name="A", vs_msa=2.0),   # flagged
        _framing(geo_display_name="B", vs_msa=1.1),   # not flagged
        _framing(geo_display_name="C", vs_msa=0.3),   # flagged below
    ]
    flags = detect_anomalies(fs)
    assert len(flags) == 2
    names = {f.geo_display_name for f in flags}
    assert names == {"A", "C"}
