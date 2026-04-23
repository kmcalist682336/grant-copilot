"""Confidence scoring for resolver candidates.

Pure functions — no DB, no LLM. Kept separate from resolver.py so tuning
happens in one place and calibration in Phase 5 doesn't need to touch
the strategy logic.

Formula:
    score = base * state_boost * type_match * alias_exact * margin
    confidence = min(1.0, score)

where:
    base         = 1 / (1 + rank_position)
    state_boost  = 1.15 if candidate.state_fips == '13' and no non-GA scope hint, else 1.0
    type_match   = 1.0 if target_type aligns with ref_type, else 0.80
    alias_exact  = 1.20 if alias matches the ref text exactly (ci), else 1.0
    margin       = factor in [0.70, 1.0] that drops as second-best gets close

Thresholds for the clarification tiers live in config/chatbot.yaml.
"""
from __future__ import annotations

from typing import Optional

from scripts.chatbot.candidates import ResolutionCandidate
from scripts.chatbot.models import ExtractedGeoRef

# Non-Georgia scope markers. If any of these appear in the ref.scope_hint
# (case-insensitive), we skip the GA bias for this candidate.
_NON_GA_MARKERS = {
    "tennessee", "tn", "alabama", "al", "florida", "fl", "south carolina",
    "sc", "north carolina", "nc", "ohio", "oh", "texas", "tx", "new york",
    "ny", "mississippi", "ms", "illinois", "il", "arkansas", "ar",
    "louisiana", "la", "kentucky", "ky",
}

# target_type alignments to ref_type. A candidate whose target_type matches
# the ref's expected family gets a full 1.0 multiplier; otherwise 0.80.
_TYPE_ALIGN: dict[str, set[str]] = {
    "administrative": {"admin"},
    "neighborhood":   {"named_place"},
    "corridor":       {"road"},
    "region":         {"named_place"},
    "landmark_area":  {"named_place"},
    "relative":       {"admin", "named_place"},   # anchors can be either
    "national":       set(),                       # handled outside FTS
}

_STATE_BOOST = 1.15
_TYPE_MISMATCH_PENALTY = 0.80
_ALIAS_EXACT_BOOST = 1.20
_MARGIN_FLOOR = 0.70


def _has_non_ga_scope(scope_hint: Optional[str]) -> bool:
    if not scope_hint:
        return False
    lower = scope_hint.lower()
    return any(m in lower for m in _NON_GA_MARKERS)


def _type_aligns(target_type: str, ref_type: str) -> bool:
    return target_type in _TYPE_ALIGN.get(ref_type, set())


def _margin_factor(
    candidate: ResolutionCandidate,
    all_candidates: list[ResolutionCandidate],
) -> float:
    """Drop confidence when the next-best candidate is close.

    Only meaningful for the best (rank_position == 0) candidate.
    Otherwise returns 1.0 (scoring isn't weakened for alternatives).
    """
    if candidate.rank_position != 0:
        return 1.0
    if len(all_candidates) < 2:
        return 1.0
    best = all_candidates[0]
    runner_up = all_candidates[1]
    # BM25 is negative; closer numbers = closer quality. Compute relative
    # gap as a fraction of the best's magnitude. Ratios < 10% → tight tie.
    if best.bm25 == 0:
        return 1.0
    gap = (runner_up.bm25 - best.bm25) / abs(best.bm25)
    # gap 0 → very close → factor ~ floor; gap >= 0.25 → factor 1.0
    clamped = max(0.0, min(gap, 0.25))
    return _MARGIN_FLOOR + (1.0 - _MARGIN_FLOOR) * (clamped / 0.25)


def compute_confidence(
    candidate: ResolutionCandidate,
    ref: ExtractedGeoRef,
    all_candidates: list[ResolutionCandidate],
) -> float:
    """Return the [0, 1] confidence score for a candidate."""
    base = 1.0 / (1 + candidate.rank_position)

    state_boost = 1.0
    if candidate.state_fips == "13" and not _has_non_ga_scope(ref.scope_hint):
        state_boost = _STATE_BOOST

    type_match = 1.0
    if not _type_aligns(candidate.target_type, ref.ref_type):
        type_match = _TYPE_MISMATCH_PENALTY

    alias_exact = 1.0
    if candidate.alias_matched.strip().lower() == ref.text.strip().lower():
        alias_exact = _ALIAS_EXACT_BOOST

    margin = _margin_factor(candidate, all_candidates)

    return min(1.0, base * state_boost * type_match * alias_exact * margin)
