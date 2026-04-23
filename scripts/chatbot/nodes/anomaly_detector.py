"""AnomalyDetector — surface statistical outliers for the synthesizer.

Phase 4 post-aggregate node. Flags values that deviate materially
from their comparator context or from their own prior period. Pure
mechanical — no LLM, no network. Runs in ~1ms over a typical
response's value set.

When PeerRetriever lands in Phase 3, this node gains peer-distribution
z-scores. For now it's ratio-based against comparator geos.

Public API:
    AnomalyFlag              — one flag per detected anomaly
    detect_anomalies(agg, framings)  -> list[AnomalyFlag]
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from scripts.chatbot.nodes.magnitude_contextualizer import (
    MagnitudeFraming,
)

logger = logging.getLogger(__name__)


AnomalyKind = Literal[
    "far_above_comparators",
    "far_below_comparators",
    "trend_break_up",
    "trend_break_down",
]


class AnomalyFlag(BaseModel):
    """One outlier worth surfacing in the prose / caveats."""

    model_config = ConfigDict(extra="ignore")

    geo_display_name: str
    concept_text: str
    year: int
    value: float
    anomaly_kind: AnomalyKind
    magnitude: float              # ratio or trend_pct that triggered
    triggering_comparator: Optional[str] = None   # "metro", "state", etc.
    context: str = ""             # short human-readable summary


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# A ratio above this or below 1/this flags "far above / below" vs. a
# specific comparator. 1.5 ≈ 50% above; this is looser than the
# framing_label buckets in MagnitudeContextualizer because we want
# AnomalyDetector to fire on fewer, more extreme cases — the framing
# already communicates "elevated"/"high". Anomaly is for the "whoa"
# outliers.
_FAR_ABOVE_RATIO = 1.50
_FAR_BELOW_RATIO = 0.67           # ≈ 1 / 1.5

# Year-over-year absolute change this large is notable regardless of
# direction (25%+). Catches "rent doubled" and "poverty halved"-type
# events the synthesizer shouldn't understate.
_TREND_BREAK_PCT = 0.25


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _strongest_ratio(
    f: MagnitudeFraming,
) -> tuple[Optional[float], Optional[str]]:
    """Pick the comparator ratio that deviates most from 1.0. Ties
    are broken by prefer-metro > state > county > US (narrative
    relevance order)."""
    candidates = [
        (f.vs_msa, "metro"),
        (f.vs_state, "state"),
        (f.vs_county, "county"),
        (f.vs_us, "national"),
    ]
    best: tuple[Optional[float], Optional[str]] = (None, None)
    best_dev = 0.0
    for ratio, label in candidates:
        if ratio is None or ratio <= 0:
            continue
        dev = abs(ratio - 1.0)
        if dev > best_dev:
            best = (ratio, label)
            best_dev = dev
    return best


def detect_anomalies(
    framings: list[MagnitudeFraming],
) -> list[AnomalyFlag]:
    """Produce zero-or-more AnomalyFlag records across all primary
    values. One framing can emit up to two flags (comparator-based
    and trend-based)."""
    flags: list[AnomalyFlag] = []
    for f in framings:
        # Comparator-based anomaly
        ratio, label = _strongest_ratio(f)
        if ratio is not None and label is not None:
            if ratio >= _FAR_ABOVE_RATIO:
                flags.append(AnomalyFlag(
                    geo_display_name=f.geo_display_name,
                    concept_text=f.concept_text,
                    year=f.year,
                    value=f.value,
                    anomaly_kind="far_above_comparators",
                    magnitude=ratio,
                    triggering_comparator=label,
                    context=(
                        f"{ratio:.2f}× the {label} average — far above "
                        f"the expected range"
                    ),
                ))
            elif ratio <= _FAR_BELOW_RATIO:
                flags.append(AnomalyFlag(
                    geo_display_name=f.geo_display_name,
                    concept_text=f.concept_text,
                    year=f.year,
                    value=f.value,
                    anomaly_kind="far_below_comparators",
                    magnitude=ratio,
                    triggering_comparator=label,
                    context=(
                        f"{ratio:.2f}× the {label} average — far below "
                        f"the expected range"
                    ),
                ))

        # Trend-break anomaly
        if (f.trend_pct is not None
                and f.prior_period_year is not None):
            if f.trend_pct >= _TREND_BREAK_PCT:
                flags.append(AnomalyFlag(
                    geo_display_name=f.geo_display_name,
                    concept_text=f.concept_text,
                    year=f.year,
                    value=f.value,
                    anomaly_kind="trend_break_up",
                    magnitude=f.trend_pct,
                    triggering_comparator=None,
                    context=(
                        f"up {f.trend_pct * 100:.1f}% since "
                        f"{f.prior_period_year} — notable shift"
                    ),
                ))
            elif f.trend_pct <= -_TREND_BREAK_PCT:
                flags.append(AnomalyFlag(
                    geo_display_name=f.geo_display_name,
                    concept_text=f.concept_text,
                    year=f.year,
                    value=f.value,
                    anomaly_kind="trend_break_down",
                    magnitude=f.trend_pct,
                    triggering_comparator=None,
                    context=(
                        f"down {abs(f.trend_pct) * 100:.1f}% since "
                        f"{f.prior_period_year} — notable shift"
                    ),
                ))
    return flags
