"""MagnitudeContextualizer — turn comparator fetches into narrative context.

Phase 4 post-aggregate node. Reads ``AggregatedResult``, groups its
values by (concept, year), and for each primary value computes:

  * ratios vs each comparator geo (county / MSA / state / US)
  * trend delta vs prior_period
  * a bucketed ``framing_label`` (very_low .. very_high)
  * a natural-language ``framing_phrase`` the synthesizer can weave
    directly into prose

No LLM call — all deterministic arithmetic + curated phrasing. This
is the node that makes J.1 comparators actually pay off: without it
the synthesizer saw five naked numbers and had to invent the
relationships between them. After it, the synthesizer sees typed
framings and just renders.

Public API:
    MagnitudeFraming      — one framing per primary value
    contextualize_magnitudes(aggregated) → list[MagnitudeFraming]
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

FramingLabel = Literal[
    "very_low", "low", "typical", "elevated", "high", "very_high",
    "unknown",
]

TrendLabel = Literal[
    "rising_strong", "rising_mild", "flat", "falling_mild",
    "falling_strong", "unknown",
]


class MagnitudeFraming(BaseModel):
    """One primary value, placed in context."""

    model_config = ConfigDict(extra="ignore")

    geo_display_name: str
    concept_text: str
    year: int
    value: float
    value_kind: Literal["scalar", "ratio", "components"] = "scalar"

    # Ratios to comparator geos. None when the comparator wasn't
    # fetched or came back empty.
    vs_county: Optional[float] = None
    vs_msa: Optional[float] = None
    vs_state: Optional[float] = None
    vs_us: Optional[float] = None

    # Raw comparator values for inline rendering.
    county_value: Optional[float] = None
    msa_value: Optional[float] = None
    state_value: Optional[float] = None
    us_value: Optional[float] = None

    # Trend
    prior_period_year: Optional[int] = None
    prior_period_value: Optional[float] = None
    trend_delta: Optional[float] = None
    trend_pct: Optional[float] = None
    trend_label: TrendLabel = "unknown"

    # Narrative
    framing_label: FramingLabel = "unknown"
    framing_phrase: str = ""           # one-sentence, synthesizer-ready
    comparison_summary: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Bucketing thresholds (ratio of primary / comparator)
# ---------------------------------------------------------------------------

_FRAMING_BUCKETS = [
    (0.60, "very_low"),
    (0.85, "low"),
    (1.15, "typical"),
    (1.40, "elevated"),
    (1.80, "high"),
    (float("inf"), "very_high"),
]


def _bucket_label(ratio: float) -> FramingLabel:
    for threshold, label in _FRAMING_BUCKETS:
        if ratio < threshold:
            return label      # type: ignore[return-value]
    return "very_high"


_TREND_BUCKETS = [
    (-0.10, "falling_strong"),
    (-0.03, "falling_mild"),
    (0.03, "flat"),
    (0.10, "rising_mild"),
    (float("inf"), "rising_strong"),
]


# Scale-plausibility window for comparator ratios. When the primary is
# a raw count (not a population-normalized ratio) and the comparator is
# a much larger geo, the "ratio" collapses to a pure population ratio
# rather than a meaningful per-capita comparison — e.g., Cobb County's
# 600k veterans against the US's 18M veterans gives "0.002× the
# national" which reads as "far below" but is just population scale.
#
# When a comparator ratio falls outside [0.10, 10.0], we suppress it
# rather than emit a nonsense framing. The synthesizer then renders the
# raw value without a ratio clause, and the framing_label degrades to
# the next comparator that does land inside the window.
_RATIO_PLAUSIBLE_MIN = 0.10
_RATIO_PLAUSIBLE_MAX = 10.0


def _ratio_is_plausible(ratio: Optional[float]) -> bool:
    """True iff the ratio lands in the plausibility window. None, 0,
    or extreme values (>10× or <0.1×) return False — those almost
    always indicate a count-vs-rate scale mismatch rather than a real
    'far above/below' signal."""
    if ratio is None or ratio <= 0:
        return False
    return _RATIO_PLAUSIBLE_MIN <= ratio <= _RATIO_PLAUSIBLE_MAX


def _trend_label(trend_pct: float) -> TrendLabel:
    for threshold, label in _TREND_BUCKETS:
        if trend_pct < threshold:
            return label      # type: ignore[return-value]
    return "rising_strong"


# ---------------------------------------------------------------------------
# Phrasing (deterministic; synthesizer may paraphrase)
# ---------------------------------------------------------------------------

def _ratio_phrase(ratio: float, comparator_label: str) -> str:
    """One-clause natural description of the ratio."""
    if ratio < 0.60:
        return f"far below the {comparator_label} level ({ratio:.2f}×)"
    if ratio < 0.85:
        return f"below the {comparator_label} level ({ratio:.2f}×)"
    if ratio < 1.15:
        return f"roughly in line with the {comparator_label} level ({ratio:.2f}×)"
    if ratio < 1.40:
        return f"somewhat above the {comparator_label} level ({ratio:.2f}×)"
    if ratio < 1.80:
        return f"substantially above the {comparator_label} level ({ratio:.2f}×)"
    return f"far above the {comparator_label} level ({ratio:.2f}×)"


def _strongest_label(
    vs_county: Optional[float], vs_msa: Optional[float],
    vs_state: Optional[float], vs_us: Optional[float],
) -> FramingLabel:
    """Pick the most-informative comparator's label. Prefer MSA, then
    state, then county, then US — reflecting which comparator best
    represents 'peer context' for a sub-state geo."""
    for ratio in (vs_msa, vs_state, vs_county, vs_us):
        if ratio is not None and ratio > 0:
            return _bucket_label(ratio)
    return "unknown"


def _build_framing_phrase(framing: MagnitudeFraming) -> str:
    """One-sentence summary combining the strongest comparator with
    a trend clause. Readable standalone, composable into prose."""
    parts: list[str] = []
    label = framing.framing_label
    strong_ratio: Optional[tuple[float, str]] = None
    for lvl, ratio in (
        ("metro", framing.vs_msa),
        ("state", framing.vs_state),
        ("county", framing.vs_county),
        ("national", framing.vs_us),
    ):
        if ratio is not None and ratio > 0:
            strong_ratio = (ratio, lvl)
            break
    if strong_ratio:
        parts.append(_ratio_phrase(*strong_ratio))
    if framing.trend_pct is not None and framing.prior_period_year is not None:
        pct = framing.trend_pct * 100
        direction = (
            "up" if pct >= 0 else "down"
        )
        parts.append(
            f"{direction} {abs(pct):.1f}% from {framing.prior_period_year}"
        )
    if not parts:
        return "no comparator or trend context available"
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _primary_value(v: AggregatedValue) -> Optional[float]:
    """A single scalar we can compare. Ratio → multiply by 100 for
    percentage readability? No — keep it as the raw ratio 0..1 so
    the math stays consistent. Synthesizer renders percentages."""
    if v.value is not None:
        return float(v.value)
    if v.ratio is not None:
        return float(v.ratio)
    return None


def _value_kind(v: AggregatedValue) -> str:
    if v.ratio is not None:
        return "ratio"
    if v.components is not None:
        return "components"
    return "scalar"


def _concept_key(v: AggregatedValue) -> str:
    """Match values across primary/comparator/prior_period via concept text."""
    return (v.concept.canonical_hint or v.concept.text or "").strip().lower()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def contextualize_magnitudes(
    aggregated: AggregatedResult,
) -> list[MagnitudeFraming]:
    """Build one MagnitudeFraming per primary-role AggregatedValue.

    Comparator and prior_period values are consumed but not emitted
    as their own framings — they're inputs to the primary's framing.

    Values with ``components`` (race-breakdown multi-variable pulls)
    get a skeletal framing without ratios, since there's no scalar
    to compare. Future J.5 disparity work can enrich these.
    """
    primaries: list[AggregatedValue] = []
    # concept_key → list of (role_family, role_detail, value)
    by_concept: dict[str, list[AggregatedValue]] = defaultdict(list)

    for v in aggregated.values:
        by_concept[_concept_key(v)].append(v)
        if v.role == "primary":
            primaries.append(v)

    out: list[MagnitudeFraming] = []
    for primary in primaries:
        key = _concept_key(primary)
        neighbors = by_concept[key]
        framing = _framing_for(primary, neighbors)
        out.append(framing)
    return out


def _framing_for(
    primary: AggregatedValue,
    neighbors: list[AggregatedValue],
) -> MagnitudeFraming:
    primary_scalar = _primary_value(primary)
    f = MagnitudeFraming(
        geo_display_name=primary.geo.display_name,
        concept_text=primary.concept.text,
        year=primary.year,
        value=primary_scalar if primary_scalar is not None else 0.0,
        value_kind=_value_kind(primary),   # type: ignore[arg-type]
    )

    def _safe_ratio(p: Optional[float], q: float) -> Optional[float]:
        """Guarded ratio: suppresses the output when the primary is a
        count and the comparator is at a population scale that makes
        the ratio meaningless. Plausibility window [0.10, 10.0]."""
        if p is None or q == 0:
            return None
        r = p / q
        return r if _ratio_is_plausible(r) else None

    # Walk neighbors, pick off comparator values + prior_period.
    for n in neighbors:
        if n is primary:
            continue
        n_scalar = _primary_value(n)
        if n_scalar is None:
            continue
        role = n.role
        if role == "comparator.county":
            f.county_value = n_scalar
            f.vs_county = _safe_ratio(primary_scalar, n_scalar)
        elif role == "comparator.msa":
            f.msa_value = n_scalar
            f.vs_msa = _safe_ratio(primary_scalar, n_scalar)
        elif role == "comparator.state":
            f.state_value = n_scalar
            f.vs_state = _safe_ratio(primary_scalar, n_scalar)
        elif role == "comparator.us":
            f.us_value = n_scalar
            f.vs_us = _safe_ratio(primary_scalar, n_scalar)
        elif role == "prior_period":
            # Pick the most recent prior if multiple exist.
            # Trend uses the SAME geo across years, so the plausibility
            # window doesn't apply — the value is comparing a place
            # against its own earlier self.
            if (f.prior_period_year is None
                    or n.year > f.prior_period_year):
                f.prior_period_year = n.year
                f.prior_period_value = n_scalar
                if primary_scalar is not None and n_scalar != 0:
                    f.trend_delta = primary_scalar - n_scalar
                    f.trend_pct = (primary_scalar - n_scalar) / n_scalar
                    f.trend_label = _trend_label(f.trend_pct)

    f.framing_label = _strongest_label(
        f.vs_county, f.vs_msa, f.vs_state, f.vs_us,
    )
    f.comparison_summary = _build_comparison_summary(f)
    f.framing_phrase = _build_framing_phrase(f)
    return f


def _build_comparison_summary(f: MagnitudeFraming) -> list[str]:
    """Short bullet-ready strings: each comparator as its own line."""
    out: list[str] = []
    if f.county_value is not None:
        out.append(f"county: {f.county_value:,.2f}")
    if f.msa_value is not None:
        out.append(f"metro: {f.msa_value:,.2f}")
    if f.state_value is not None:
        out.append(f"state: {f.state_value:,.2f}")
    if f.us_value is not None:
        out.append(f"national: {f.us_value:,.2f}")
    if f.prior_period_year is not None and f.prior_period_value is not None:
        out.append(
            f"prior ({f.prior_period_year}): {f.prior_period_value:,.2f}"
        )
    return out
