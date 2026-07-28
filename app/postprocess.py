"""Deterministic cleanup of synthesizer output — STUDENTS EDIT THIS FILE.

Runs after the LLM returns and before the answer is rendered. No LLM
calls, no network, no randomness: the same answer in must produce the
same answer out. That property is what makes "Re-render" instant and
free.

Where to make a change:

  * "the wording should be different"        -> prompts/v1/synthesizer.yaml
  * "it shouldn't be told about X at all"    -> presentation.yaml, bundle.send
  * "the format is wrong / strip that thing" -> HERE
  * "don't show that section"                -> presentation.yaml, sections

``custom_clean`` receives the synthesis bundle, so cleanup can be
conditional on what the pipeline actually produced for this query rather
than guessing. ``"peer_contexts" in bundle`` is a real test: bundle keys
are omitted when empty.
"""
from __future__ import annotations

import re
from typing import Any

from app.pipeline_adapter import SynthesizedAnswer

# ```json ... ``` wrappers the model occasionally emits around prose
# despite being asked for a bare string.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _normalize_ws(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _truncate_words(text: str, max_words: int) -> str:
    """Cut to max_words, then back off to the last sentence boundary so
    the prose doesn't end mid-clause."""
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words])
    for end in (". ", "! ", "? "):
        idx = clipped.rfind(end)
        if idx > len(clipped) * 0.5:
            return clipped[: idx + 1]
    return clipped.rstrip(" ,;:") + "…"


def clean(
    answer: SynthesizedAnswer,
    bundle: dict[str, Any],
    formatting: dict[str, Any],
) -> SynthesizedAnswer:
    """Apply the config-driven steps, then ``custom_clean``.

    Returns a new answer; the input is not mutated, so the trace pane can
    still show the model's raw output.
    """
    if answer is None:
        return answer

    prose = answer.prose or ""
    key_findings = list(answer.key_findings or [])
    caveats = list(answer.caveats or [])

    if formatting.get("strip_markdown_fences", True):
        prose = _strip_fences(prose)
        key_findings = [_strip_fences(k) for k in key_findings]
        caveats = [_strip_fences(c) for c in caveats]

    if formatting.get("normalize_whitespace", True):
        prose = _normalize_ws(prose)
        key_findings = [_normalize_ws(k) for k in key_findings]
        caveats = [_normalize_ws(c) for c in caveats]

    max_words = formatting.get("prose_max_words")
    if isinstance(max_words, int) and max_words > 0:
        prose = _truncate_words(prose, max_words)

    max_kf = formatting.get("max_key_findings")
    if isinstance(max_kf, int) and max_kf > 0:
        key_findings = key_findings[:max_kf]

    max_cv = formatting.get("max_caveats")
    if isinstance(max_cv, int) and max_cv > 0:
        caveats = caveats[:max_cv]

    out = SynthesizedAnswer(
        prose=prose,
        key_findings=[k for k in key_findings if k],
        caveats=[c for c in caveats if c],
        citations=answer.citations,
    )
    return custom_clean(out, bundle, formatting)


def custom_clean(
    answer: SynthesizedAnswer,
    bundle: dict[str, Any],
    formatting: dict[str, Any],
) -> SynthesizedAnswer:
    """Your cleanup goes here. Called last. Starts as a no-op.

    ``bundle`` is the synthesis bundle — every upstream artifact that was
    available for this query. The Upstream panel in the app lists its
    keys; anything shown there can be branched on here.

    Deterministic only: no LLM calls, no network, no randomness. If this
    raises, the app falls back to the uncleaned answer and shows a
    warning rather than breaking.

    Examples:

        # Only strip peer names when peers were actually computed.
        if "peer_contexts" in bundle:
            for group in bundle["peer_contexts"]:
                for peer in getattr(group, "peers", []) or []:
                    answer.prose = answer.prose.replace(
                        getattr(peer, "geo_name", ""), "a comparable metro",
                    )

        # Aggregated tract medians are statistically soft — make sure the
        # caveat survives, but only when a value actually carries one.
        if any(getattr(v, "aggregation_caveat", None)
               for v in bundle.get("aggregated_values", [])):
            if not any("aggregat" in c.lower() for c in answer.caveats):
                answer.caveats.append(
                    "Neighborhood figures average tract-level estimates."
                )

        # House style.
        answer.key_findings = [
            k if k.endswith(".") else k + "." for k in answer.key_findings
        ]
    """
    return answer
