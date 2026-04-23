"""ScopeGate — one LLM call at the very top of the pipeline that
decides whether a user query is within the scope of US Census data.

Purpose
-------
The rest of the pipeline will confidently produce an answer to any
query that routes to SOMETHING in the Census catalog — even when the
nearest table is only topically adjacent. A user asking "how many
jet skis are in Northeast Atlanta" currently gets back a vehicle-
ownership ratio dressed up as an answer, because B25044 (Tenure by
Vehicles Available) is the closest cosine match. That's wrong: the
user's question can't be answered from Census data at all, and the
correct behavior is to say so.

This gate runs BEFORE extraction / resolution / routing and short-
circuits the pipeline with a polite "Census doesn't track that"
message when the query is out of scope. Deterministic checks
(obvious-bad substrings, trivially-good phrasings like "population
of X") decide without LLM cost; only ambiguous queries consult the
LLM.

Public API
----------
    ScopeVerdict              - result model (answerable + message)
    is_in_scope(query, llm)   - entry point
    ScopeGateError            - raised on LLM failure (caller should
                                fail-open: proceed with the query)

Design notes
------------
The gate is fail-open by convention: if the LLM call fails or times
out, we let the query through. A missed-rejection is recoverable
downstream (you get a weak answer); a false-rejection feels broken
(we refuse to answer a legitimate question). Also keeps the gate's
latency ceiling from becoming a hard dependency.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.prompt_loader import (
    load_prompt_template, render_system_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class ScopeVerdict(BaseModel):
    """Whether the query is answerable from US Census data."""

    model_config = ConfigDict(extra="ignore")

    answerable: bool = Field(
        ...,
        description=(
            "True when the query is about demographic, economic, "
            "housing, transportation, or similar Census-covered "
            "topics for a US geography. False for consumer goods, "
            "recreational items, brand/business names, entertainment, "
            "weather, sports scores, etc."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "One short sentence explaining the decision. Used for "
            "logs and — when answerable is False — as the user-"
            "facing message."
        ),
    )


class ScopeGateError(RuntimeError):
    """Raised when the gate LLM call fails or produces unparseable
    output. Callers should fail-open (proceed with the query)."""


# ---------------------------------------------------------------------------
# Fast deterministic shortcuts
# ---------------------------------------------------------------------------

# Queries that are almost always in-scope. One of these as a substring
# in the query skips the LLM entirely.
_ALWAYS_IN_SCOPE_TOKENS: tuple[str, ...] = (
    "population", "median income", "median household",
    "poverty rate", "poverty", "unemployment", "employment",
    "snap", "food stamp", "food pantry", "food insecurity",
    "food bank", "food desert",
    "rent", "housing", "home value", "homeowner", "renter",
    "cost burden", "housing stress",
    "commute", "vehicle", "broadband", "internet",
    "education", "bachelor", "degree", "school enrollment",
    "insurance", "uninsured", "medicare", "medicaid",
    "race", "ethnicity", "hispanic", "black", "white",
    "asian", "language", "foreign-born", "immigration",
    "veteran", "disability", "age", "senior", "elderly",
    "child", "family", "household",
    "displacement", "gentrification", "neighborhood change",
)

# Queries that almost certainly aren't Census-answerable — consumer
# goods, recreation, entertainment, specific brands/businesses,
# weather, sports, etc. Substring hit → rejected without LLM.
_OBVIOUS_OUT_OF_SCOPE_TOKENS: tuple[str, ...] = (
    "jet ski", "snowmobile", "yacht", "boat ownership",
    "boats ",
    "guitar", "piano", "violin",
    "tv show", "netflix", "youtube", "tiktok", "instagram",
    "restaurant", "bar ", "coffee shop", "menu",
    "concert", "band", "album", "movie", "film",
    "sports team", "football team", "nba", "nfl", "mlb",
    "weather", "temperature", "rainfall",
    "stock price", "crypto", "bitcoin",
    "recipe", "cooking",
    "video game", "playstation", "xbox", "nintendo",
    "pet ownership", "dog ownership", "cat ownership",
)


_DEFAULT_REFUSAL = (
    "I don't have Census data on that topic. I can help with "
    "demographic, economic, housing, transportation, health-"
    "insurance, and education indicators for US geographies. "
    "Try rephrasing around one of those."
)


def _deterministic_check(query: str) -> Optional[ScopeVerdict]:
    """Return a verdict if the query is trivially in/out of scope;
    otherwise None (meaning: ask the LLM).

    Uses word-boundary regex matching rather than raw substring ``in``
    to avoid false-positives like "broadband" tripping the "band"
    out-of-scope token, or "concerts" colliding with "concerts" the
    single-word in-scope token — the earlier version of this check
    silently rejected rural-broadband grant queries because of that
    exact bug."""
    import re as _re
    q = query.lower()
    for bad in _OBVIOUS_OUT_OF_SCOPE_TOKENS:
        pattern = r"\b" + _re.escape(bad) + r"s?\b"
        if _re.search(pattern, q):
            return ScopeVerdict(
                answerable=False,
                reason=(
                    f"contains out-of-scope token {bad!r} — Census "
                    f"does not track this subject"
                ),
            )
    for good in _ALWAYS_IN_SCOPE_TOKENS:
        pattern = r"\b" + _re.escape(good) + r"s?\b"
        if _re.search(pattern, q):
            return ScopeVerdict(
                answerable=True,
                reason=f"matches in-scope token {good!r}",
            )
    return None


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "prompts" / "v1" / "scope_gate.yaml"
)


@functools.lru_cache(maxsize=1)
def _get_prompt_and_schema() -> tuple[str, dict]:
    template = load_prompt_template(_PROMPT_PATH)
    schema = ScopeVerdict.model_json_schema()
    return render_system_prompt(template, schema), schema


def invalidate_prompt_cache() -> None:
    _get_prompt_and_schema.cache_clear()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def is_in_scope(
    query: str,
    llm: LLMClient,
    *,
    temperature: float = 0.0,
    hard_timeout_s: float = 20.0,
) -> ScopeVerdict:
    """Return a ScopeVerdict for ``query``.

    Priority order:
      1. Deterministic in/out substring match (no LLM).
      2. LLM classifier, wrapped in a daemon-thread timeout so a
         stuck Vertex call can't stall the pipeline.
      3. On any LLM failure → fail-open ``answerable=True``.
    """
    det = _deterministic_check(query)
    if det is not None:
        return det

    try:
        system_prompt, schema = _get_prompt_and_schema()
    except FileNotFoundError as e:
        logger.warning(
            "scope_gate prompt missing (%s); failing open", e,
        )
        return ScopeVerdict(
            answerable=True, reason="scope_gate prompt missing",
        )

    user_payload = json.dumps({"query": query}, ensure_ascii=False)

    holder: dict = {"raw": None, "error": None}

    def _invoke():
        try:
            holder["raw"] = llm.extract(
                system_prompt=system_prompt,
                user_text=user_payload,
                schema=schema,
                temperature=temperature,
            )
        except BaseException as e:                     # noqa: BLE001
            holder["error"] = e

    worker = threading.Thread(
        target=_invoke, name="scope-gate", daemon=True,
    )
    worker.start()
    worker.join(timeout=hard_timeout_s)
    if worker.is_alive():
        logger.warning(
            "scope_gate timed out after %.1fs; failing open", hard_timeout_s,
        )
        return ScopeVerdict(
            answerable=True,
            reason=f"scope_gate timed out after {hard_timeout_s}s",
        )

    err = holder["error"]
    if err is not None:
        if isinstance(err, LLMCallError):
            logger.warning("scope_gate LLM call failed: %s", err)
        else:
            logger.warning("scope_gate unexpected error: %s", err)
        return ScopeVerdict(
            answerable=True,
            reason=f"scope_gate LLM failed: {err}",
        )
    raw = holder["raw"]

    if not isinstance(raw, dict):
        logger.warning(
            "scope_gate non-dict output: %s", type(raw).__name__,
        )
        return ScopeVerdict(
            answerable=True, reason="scope_gate returned non-dict",
        )

    try:
        verdict = ScopeVerdict.model_validate(raw)
    except ValidationError as e:
        logger.warning("scope_gate schema mismatch: %s", e)
        return ScopeVerdict(
            answerable=True, reason="scope_gate schema mismatch",
        )

    return verdict


def user_message_for_refusal(verdict: ScopeVerdict) -> str:
    """Render the user-facing message when the gate rejects a query."""
    if verdict.answerable:
        return ""
    # Use the LLM's reason when it's substantive, else the canned
    # message. "one sentence" is a loose contract, so guard against
    # empty / trivial reasons.
    reason = (verdict.reason or "").strip()
    if len(reason) < 15:
        return _DEFAULT_REFUSAL
    return (
        f"I don't have Census data on that topic. {reason}\n\n"
        f"I can help with demographic, economic, housing, "
        f"transportation, health-insurance, and education indicators "
        f"for US geographies."
    )
