"""Deterministic checks on synthesizer output.

Lints never modify the answer — they produce warnings the UI renders
next to it. The point is to make model misbehavior *visible* during
prompt iteration, so a student can tell whether an edit helped or just
moved the problem.

The headline check is ``forbid_unsourced_numbers``. Because citations
are derived from the plan rather than the model, every figure the model
is allowed to state must trace back to a value the pipeline actually
retrieved. Anything else was invented. That makes hallucination
mechanically detectable here in a way it usually isn't.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel


class Lint(BaseModel):
    """One lint finding."""

    rule: str
    severity: str          # "error" | "warning" | "info"
    message: str
    detail: Optional[str] = None


# Numbers as they appear in prose: 1,234,567 · $72,500 · 13.5% · 0.135
#
# The comma-grouped form comes first and requires commas to be followed
# by exactly three digits. A looser [\d,]* swallows sentence punctuation
# ("In 2023, median…" tokenizes as "2023,"), which then misses the
# year exemption and reports every date as a hallucinated figure.
_NUM_RE = re.compile(
    r"\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"      # 1,234  ·  $72,500
    r"|\$?\d+(?:\.\d+)?%?"                     # 2023  ·  13.5%  ·  0.135
)

# Four-digit years, 1900-2099. Excluded from the unsourced-number check
# (they're vintages, not measurements) and used by the year lint.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Small integers are almost always prose ("three counties", "top 5"),
# ordinals, or list counts rather than data claims. Checking them
# produces false positives that train students to ignore the lint.
_TRIVIAL_MAX = 100.0


def _parse_number(token: str) -> Optional[float]:
    """'$72,500' -> 72500.0 · '13.5%' -> 13.5 · '1,234' -> 1234.0"""
    cleaned = token.strip().lstrip("$").rstrip("%").replace(",", "")
    if not cleaned or cleaned == ".":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sourced_values(aggregated: Any) -> set[float]:
    """Every number the pipeline actually retrieved, in every form the
    synthesizer is licensed to render it in.

    A ratio of 0.135 may legitimately appear as '13.5%', so both the raw
    and the percent-scaled form count as sourced.
    """
    out: set[float] = set()

    def add(x: Optional[float]) -> None:
        if x is None:
            return
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        if math.isnan(f) or math.isinf(f):
            return
        out.add(f)

    for v in getattr(aggregated, "values", []) or []:
        add(getattr(v, "value", None))
        add(getattr(v, "sample_size", None))

        ratio = getattr(v, "ratio", None)
        if ratio is not None:
            add(ratio)
            add(ratio * 100.0)        # rendered as a percent

        components = getattr(v, "components", None) or {}
        if isinstance(components, dict):
            total = 0.0
            for cv in components.values():
                add(cv)
                try:
                    total += float(cv)
                except (TypeError, ValueError):
                    pass
            add(total)                # component sums are fair to state

    return out


def _is_sourced(n: float, sourced: Iterable[float]) -> bool:
    """Match with tolerance, because the prompt explicitly tells the
    model to round ('$72,483' -> '$72,500', '13.47%' -> '13.5%').

    Tolerance is proportional: 1.5% of magnitude, which covers rounding
    to 2-3 significant figures without waving through a wrong number.
    """
    for s in sourced:
        if s == n:
            return True
        scale = max(abs(s), abs(n), 1.0)
        if abs(s - n) / scale <= 0.015:
            return True
    return False


def check_unsourced_numbers(prose: str, aggregated: Any) -> list[Lint]:
    sourced = _sourced_values(aggregated)
    years = {m.group(0) for m in _YEAR_RE.finditer(prose)}

    unsourced: list[str] = []
    for m in _NUM_RE.finditer(prose):
        token = m.group(0)
        if token in years:
            continue
        n = _parse_number(token)
        if n is None:
            continue
        # A bare integer in year range is a vintage, not a measurement.
        # Checked on the parsed value so "2023" is exempt whether or not
        # the year regex matched it in context.
        if (
            "$" not in token and "%" not in token and "." not in token
            and 1900 <= n <= 2099
        ):
            continue
        if abs(n) <= _TRIVIAL_MAX and "%" not in token and "$" not in token:
            continue
        if not _is_sourced(n, sourced):
            unsourced.append(token)

    if not unsourced:
        return []
    return [Lint(
        rule="forbid_unsourced_numbers",
        severity="error",
        message=(
            f"{len(unsourced)} number(s) in the prose don't match any "
            f"retrieved value."
        ),
        detail=(
            "Unmatched: " + ", ".join(unsourced[:12])
            + ("…" if len(unsourced) > 12 else "")
            + ". These may be hallucinated, or derived (a difference or "
              "percentage the model computed itself). Either way the "
              "prose is stating a figure the pipeline can't cite."
        ),
    )]


def check_year_mentioned(prose: str) -> list[Lint]:
    if _YEAR_RE.search(prose):
        return []
    return [Lint(
        rule="require_year_mentioned",
        severity="warning",
        message="The prose never states which year the data is from.",
        detail="ACS vintages differ materially; an undated figure is "
               "ambiguous to a grant reviewer.",
    )]


def check_peer_names(
    prose: str, key_findings: list[str], peer_contexts: Any,
) -> list[Lint]:
    """Catch peer geographies leaking into prose that's supposed to stay
    focused on the anchor geography."""
    names: set[str] = set()
    for ctx in peer_contexts or []:
        peers = getattr(ctx, "peers", None)
        if peers is None and isinstance(ctx, dict):
            peers = ctx.get("peers")
        for p in peers or []:
            nm = getattr(p, "geo_name", None)
            if nm is None and isinstance(p, dict):
                nm = p.get("geo_name")
            if nm:
                names.add(str(nm))

    if not names:
        return []

    haystack = " ".join([prose] + list(key_findings or []))
    leaked = sorted(n for n in names if n and n in haystack)
    if not leaked:
        return []
    return [Lint(
        rule="forbid_peer_names",
        severity="warning",
        message=f"{len(leaked)} peer geography name(s) appear in the prose.",
        detail=(
            "Leaked: " + ", ".join(leaked[:8])
            + ". The peers section renders these separately. Consider "
              "setting payload.include_peer_contexts: false in "
              "presentation.yaml rather than instructing the model to "
              "avoid them."
        ),
    )]


def check_prose_length(prose: str, word_range: Any) -> list[Lint]:
    if not isinstance(word_range, (list, tuple)) or len(word_range) != 2:
        return []
    try:
        lo, hi = int(word_range[0]), int(word_range[1])
    except (TypeError, ValueError):
        return []
    n = len(prose.split())
    if lo <= n <= hi:
        return []
    direction = "short" if n < lo else "long"
    return [Lint(
        rule="prose_word_range",
        severity="info",
        message=f"Prose is {n} words — {direction} (target {lo}-{hi}).",
    )]


def run_all(
    answer: Any,
    aggregated: Any,
    peer_contexts: Any,
    config: dict,
) -> list[Lint]:
    """Run every enabled lint. Never raises — a broken lint must not take
    down an otherwise good answer."""
    if answer is None:
        return []

    prose = getattr(answer, "prose", "") or ""
    key_findings = list(getattr(answer, "key_findings", []) or [])
    out: list[Lint] = []

    try:
        if config.get("forbid_unsourced_numbers", True) and aggregated is not None:
            out += check_unsourced_numbers(prose, aggregated)
        if config.get("require_year_mentioned", True):
            out += check_year_mentioned(prose)
        if config.get("forbid_peer_names", True):
            out += check_peer_names(prose, key_findings, peer_contexts)
        if "prose_word_range" in config:
            out += check_prose_length(prose, config.get("prose_word_range"))
    except Exception as e:                              # pragma: no cover
        out.append(Lint(
            rule="lint_error",
            severity="info",
            message=f"A lint failed to run: {e}",
        ))
    return out
