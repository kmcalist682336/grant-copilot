"""Phase I: AggregatedResult → human-readable prose with citations.

The synthesizer is the last LLM hop in the chatbot pipeline. Inputs:

  - the user's original query (for tone + ensuring we answer it)
  - the planner's metadata (so we can emit accurate citations)
  - the aggregator's structured outputs (the actual numbers)

Outputs (Pydantic, schema-validated):

  - prose: 1-3 paragraphs the user reads
  - key_findings: 3-5 short bullets the UI can render as highlights
  - caveats: explicit data-limitation warnings for the user

The orchestrator builds the citation list deterministically from the
plan rather than trusting the LLM — citations must be exact.

Numbers are formatted by the synthesizer prompt's instructions, but
we also embed them into the user-message JSON in their raw form so
the LLM can echo correctly. We never let the LLM invent values.

Public API:
    SynthesizedAnswer
    Citation
    synthesize_answer(query, intent, plan, aggregated, llm) → SynthesizedAnswer
    SynthesisError
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.llm_client import LLMCallError, LLMClient
from scripts.chatbot.models import ExtractedIntent
from scripts.chatbot.planner import PlanResult

logger = logging.getLogger(__name__)


class SynthesisError(RuntimeError):
    """Raised when the LLM fails to produce a valid SynthesizedAnswer."""


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class CitedVariable(BaseModel):
    """One variable under a Citation, with its plain-English label.

    Populated after synthesis by the orchestrator via a metadata-DB
    lookup — keeping it on the Citation (rather than elsewhere) lets
    the REPL render each source with the specific variable_ids that
    were actually fetched, not just the table id."""

    model_config = ConfigDict(extra="ignore")

    variable_id: str
    label: Optional[str] = None


class Citation(BaseModel):
    """One Census source the answer relies on."""

    model_config = ConfigDict(extra="ignore")

    table_id: str
    dataset: str
    year: int
    table_title: Optional[str] = None
    universe: Optional[str] = None
    variables: list[CitedVariable] = Field(
        default_factory=list,
        description=(
            "Variables pulled from this table, with plain-English "
            "labels when the metadata DB has them. Rendered under "
            "the source line so you can see exactly which columns "
            "fed the answer."
        ),
    )


class SynthesizedAnswer(BaseModel):
    """Full synthesizer output."""

    model_config = ConfigDict(extra="ignore")

    prose: str = Field(
        ..., description="1-3 paragraph natural-language answer.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 short bullets for UI highlight rendering.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Data limitations the user should know about.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Filled by the orchestrator from the plan, not the LLM.",
    )


class _SynthesisLLMOutput(BaseModel):
    """What the LLM emits — citations are added later by the orchestrator."""

    model_config = ConfigDict(extra="ignore")

    prose: str
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Citation building
# ---------------------------------------------------------------------------

def _build_citations(
    plan: PlanResult, aggregated: AggregatedResult,
) -> list[Citation]:
    """Derive a deduped, ordered list of Citations from the plan.

    One Citation per unique ``(table_id, dataset, year)`` actually
    called by the planner — iterating ``plan.calls`` directly so
    Tier-3 (semantic-routed) tables get cited alongside Tier-1
    (curated concept map) and Tier-2 (FTS) picks. The previous
    implementation keyed off ``plan.concept_resolutions`` and
    silently dropped every tier_semantic resolution, leaving the
    Sources list incomplete when the router — not the curated map
    — picked the table.

    Order is stable (sorted) so bibliography numbers are
    reproducible across runs.
    """
    # Collect table titles / universes from concept_resolutions
    # indexed by the TABLE the resolution pointed at. This lets us
    # attach a title even when the plan picked a different table
    # than the concept_resolution suggests (e.g. a tier_semantic
    # override of a tier_1 hit).
    title_by_table: dict[tuple[str, str], dict[str, Any]] = {}
    for cr in plan.concept_resolutions:
        if cr.concept_entry is not None:
            key = (cr.concept_entry.table_id, cr.concept_entry.dataset)
            title_by_table.setdefault(key, {
                "title": cr.concept_entry.canonical,
                "universe": cr.concept_entry.universe,
            })
        if cr.table_candidate is not None:
            tc = cr.table_candidate
            key = (tc.table_id, tc.dataset)
            title_by_table.setdefault(key, {
                "title": tc.title,
                "universe": getattr(tc, "universe", None),
            })
        # tier_semantic: routed_result.top_tables[0] carries table_id
        # + dataset but no title string. We still want a citation —
        # just without a human title on that row.
        rr = getattr(cr, "routed_result", None)
        if rr is not None:
            top_tables = getattr(rr, "top_tables", None) or []
            for tt in top_tables:
                tid = getattr(tt, "target_table_id", None)
                ds = getattr(tt, "target_dataset", None)
                if tid and ds:
                    title_by_table.setdefault((tid, ds), {
                        "title": None, "universe": None,
                    })

    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    vars_by_key: dict[tuple[str, str, int], list[str]] = {}
    for c in plan.calls:
        key = (c.api_call.table_id, c.api_call.dataset, c.year)
        if key not in by_key:
            by_key[key] = title_by_table.get(
                (c.api_call.table_id, c.api_call.dataset),
                {"title": None, "universe": None},
            )
        # Collect every variable_id cited under this (table, dataset,
        # year). Dedupe and preserve first-seen order so rendering
        # is stable across runs.
        bucket = vars_by_key.setdefault(key, [])
        for vid in c.api_call.variables:
            if vid not in bucket:
                bucket.append(vid)

    return [
        Citation(
            table_id=tid, dataset=ds, year=year,
            table_title=meta.get("title"),
            universe=meta.get("universe"),
            variables=[
                CitedVariable(variable_id=v)
                for v in vars_by_key.get((tid, ds, year), [])
            ],
        )
        for (tid, ds, year), meta in sorted(by_key.items())
    ]


def enrich_citation_variables(
    citations: list[Citation], metadata_db,
) -> list[Citation]:
    """Look up plain-English labels for every variable under each
    citation from ``search_index.db.variables.label_natural``. Runs
    AFTER synthesis in the orchestrator (the synthesizer itself
    doesn't take a DB handle). Returns new Citation objects with
    the ``variables`` list populated.

    If the metadata DB is missing a variable entry, its label stays
    None — the renderer falls back to showing the variable_id alone
    rather than dropping it, so the Sources list stays complete even
    on partial catalog coverage.
    """
    if metadata_db is None or not citations:
        return citations
    out: list[Citation] = []
    for cit in citations:
        enriched: list[CitedVariable] = []
        for cv in cit.variables:
            try:
                row = metadata_db.execute(
                    "SELECT label_natural FROM variables "
                    "WHERE table_id = ? AND dataset = ? "
                    "  AND variable_id = ? "
                    "  AND label_natural IS NOT NULL "
                    "  AND label_natural != '' "
                    "ORDER BY year DESC LIMIT 1",
                    (cit.table_id, cit.dataset, cv.variable_id),
                ).fetchone()
            except Exception:                          # pragma: no cover
                row = None
            label = None
            if row is not None:
                label = (
                    row[0] if isinstance(row, tuple)
                    else row["label_natural"]
                )
            enriched.append(CitedVariable(
                variable_id=cv.variable_id, label=label,
            ))
        out.append(cit.model_copy(update={"variables": enriched}))
    return out


# ---------------------------------------------------------------------------
# LLM input construction
# ---------------------------------------------------------------------------

def _format_value(v: AggregatedValue) -> dict[str, Any]:
    """Compact, JSON-friendly representation of one AggregatedValue
    for the LLM. Surfaces every relevant signal but avoids any
    prose-style framing — that's the LLM's job."""
    out: dict[str, Any] = {
        "geo": v.geo.display_name,
        "geo_level": v.geo.geo_level,
        "concept": v.concept.text,
        "year": v.year,
        "role": v.role,
    }
    if v.value is not None:
        out["value"] = v.value
    if v.ratio is not None:
        out["ratio"] = v.ratio
    if v.components is not None:
        out["components"] = v.components
    if v.sample_size is not None:
        out["sample_size"] = v.sample_size
    if v.rows_aggregated:
        out["rows_aggregated"] = v.rows_aggregated
    if v.aggregation_caveat:
        out["aggregation_caveat"] = v.aggregation_caveat
    if v.notes:
        out["notes"] = v.notes
    return out


_SYSTEM_PROMPT = """\
You are a US Census data synthesizer. Given a user query, a structured
aggregated dataset, and optional Phase 4 realism-agent outputs
(magnitude framings, anomaly flags, followups, and the matched
grant-narrative frame), write a SHORT, FACTUAL answer.

Output STRICT JSON. No prose outside JSON.

OUTPUT SHAPE
------------
{
  "prose": "1-3 short paragraphs answering the user's question",
  "key_findings": ["short bullet 1", "short bullet 2", ...],
  "caveats": ["data limitation 1", ...]
}

CORE RULES
----------
- NEVER invent numbers. Every figure in `prose` must come from the
  `aggregated_values` payload — quote them as-is or apply the
  formatting rules below. If a value is missing, say so explicitly
  ("data unavailable for this geography").
- Format numbers cleanly:
    counts:    use commas (1,234,567)
    dollars:   prefix '$' and use commas ($72,500)
    percents:  if `ratio` ∈ [0,1] AND the concept is rate-like,
               render as percent (e.g., 0.135 → "13.5%")
    medians:   round to a sensible scale; mention "median"
- Always mention the year of the data in the prose.
- When `aggregation_caveat` is set on a value, surface it in `caveats`
  in your own words (don't dump the raw caveat verbatim).
- When the original query implies a comparison
  (`comparison_implied=true`), structure the prose around comparing
  the geos. Otherwise summarize per-geo.
- If `fetch_failures` is non-empty, mention partial-data conditions in
  `caveats`. Each failure string is shaped:
    "<geo> × '<concept>' @ <geo_level> [<role>] (<year>, <dataset>/<table>): <error>"
  Phrase each one in `caveats` as:
    "Couldn't retrieve <concept> for <geo> at the <geo_level> level —
     this could be a transient API problem or the variable may not
     exist at that geography."
  When the `error` is `abandoned_after_20s` or any `*_timeout*`
  variant, lean toward the API-problem framing. When the error is
  `http_400` / `http_404`, lean toward "may not exist at that level."
  Group by concept if several geos failed for the same concept so the
  caveats stay short.
- If `aggregated_values` is empty, return prose explaining that no
  data was returned and why (use the failures list as a guide).
- Do NOT include citations in JSON output — the system appends them
  separately from the planner's metadata.

GRANT-FRAME HANDLING
--------------------
If the payload includes a `frame` object with a `rhetorical_target`,
shape the prose to that target. For instance, a "food_access" frame
wants a grant-case paragraph (poverty + SNAP + grocery access) rather
than a naked-statistics list. A "specific_lookup" frame wants one
paragraph stating the value with comparator context and nothing else.

MAGNITUDE FRAMINGS (when present)
---------------------------------
When `magnitude_framings` is non-empty, use its typed fields instead
of inventing comparator phrasings from raw ratios:
  - `framing_label` (very_low / low / typical / elevated / high / very_high)
    tells you the strongest comparator bucket.
  - `framing_phrase` is a synthesizer-ready summary — weave it in,
    don't quote verbatim.
  - `vs_county / vs_msa / vs_state / vs_us` are ratios (1.0 = parity).
    Only cite them when they materially exceed the "typical" bucket.
  - `trend_label` + `prior_period_year` give trend direction; cite
    when not "flat" or "unknown".
Prefer concrete wording: "42% — nearly double the metro rate of 23%"
over abstract "above average".

ANOMALY FLAGS (when present)
----------------------------
When `anomaly_flags` is non-empty, surface them prominently. An
`anomaly_kind == "far_above_comparators"` with `magnitude > 2.0` is
headline-worthy for a grant case — open the prose with it. Use the
`context` field as phrasing guidance.

FOLLOWUPS (when present)
------------------------
When `suggested_followups` is non-empty, append them to `key_findings`
as a final "Try next:" bullet group (up to 4). Format as
"Try next: {question}" — keep them short; don't include the rationale.

PEER CONTEXTS (when present)
----------------------------
``peer_contexts`` is passed in so the downstream UI can render a
dedicated "Peer comparisons" section beneath your prose. That
section names each peer with its own short justification — so the
prose itself should NOT mention peers, peer cities/counties, peer
feature values, or "nationwide peers of comparable size" phrasing.

Rules:
- DO NOT name any peer geography (e.g. "Kansas City, Kansas",
  "Columbus, Ohio", "DeKalb County") in the prose or key_findings.
- DO NOT cite peer feature values in the prose. Keep every number in
  the prose tied to the anchor geography and its comparators
  (county/MSA/state/national) only.
- DO NOT reference the existence of peer comparisons ("similar to
  other cities", "peers suggest…") — the peers section below the
  prose handles that entirely.
- Focus the prose on the anchor's own numbers, trends, and how it
  compares to its administrative comparators (county/MSA/state/US).
  Shorter is better — aim for 80-150 words of tight prose that
  directly supports the grant narrative.

Be concise. 100-250 words for `prose` is plenty for most queries;
grant-frame queries can go to 300.
"""


def _build_user_payload(
    query: str,
    intent: ExtractedIntent,
    aggregated: AggregatedResult,
    *,
    frame: Optional[Any] = None,
    magnitude_framings: Optional[list] = None,
    anomaly_flags: Optional[list] = None,
    followups: Optional[list] = None,
    peer_contexts: Optional[list] = None,
) -> str:
    """Render the user-side JSON payload the LLM sees.

    Phase 4 extras (frame, magnitude_framings, anomaly_flags,
    followups) and Phase 3 extras (peer_contexts) are only included
    when non-empty so legacy callers without them see the original
    payload shape.
    """
    payload: dict[str, Any] = {
        "user_query": query,
        "intent_summary": {
            "intent_type": intent.intent_type,
            "comparison_implied": intent.comparison_implied,
            "national_comparison_implied": intent.national_comparison_implied,
            "temporal_intent": intent.temporal_intent,
            "explicit_years": intent.years,
        },
        "aggregated_values": [_format_value(v) for v in aggregated.values],
        "fetch_failures": aggregated.fetch_failures,
    }
    if frame is not None:
        payload["frame"] = {
            "name": getattr(frame, "name", ""),
            "rhetorical_target": getattr(frame, "rhetorical_target", ""),
            "standard_caveats": list(
                getattr(frame, "standard_caveats", []) or []
            ),
        }
    if magnitude_framings:
        payload["magnitude_framings"] = [
            f.model_dump() if hasattr(f, "model_dump") else f
            for f in magnitude_framings
        ]
    if anomaly_flags:
        payload["anomaly_flags"] = [
            f.model_dump() if hasattr(f, "model_dump") else f
            for f in anomaly_flags
        ]
    if followups:
        payload["suggested_followups"] = [
            f.model_dump() if hasattr(f, "model_dump") else f
            for f in followups
        ]
    if peer_contexts:
        # Compact each peer context so the LLM sees what it needs to
        # cite peers with concrete numbers — axis, scope, anchor's
        # own feature values, and each peer's feature values. Raw
        # distance scores are omitted on purpose; they're not
        # narratively useful.
        def _attr(obj, name, default=None):
            if hasattr(obj, name):
                return getattr(obj, name)
            if isinstance(obj, dict):
                return obj.get(name, default)
            return default

        payload["peer_contexts"] = [
            {
                "axis": _attr(c, "axis", ""),
                "axis_description": _attr(c, "axis_description", ""),
                "pool_scope": _attr(c, "pool_scope", ""),
                "anchor_geo_name": _attr(c, "anchor_geo_name", ""),
                "anchor_feature_values": _attr(
                    c, "anchor_feature_values", {}
                ) or {},
                "peers": [
                    {
                        "geo_name": _attr(p, "geo_name", ""),
                        "population": _attr(p, "population", None),
                        "match_explanation": _attr(
                            p, "match_explanation", "",
                        ),
                        "feature_values": _attr(
                            p, "feature_values", {}
                        ) or {},
                    }
                    for p in (_attr(c, "peers", []) or [])
                ],
            }
            for c in peer_contexts
        ]
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_answer(
    query: str,
    intent: ExtractedIntent,
    plan: PlanResult,
    aggregated: AggregatedResult,
    llm: LLMClient,
    *,
    temperature: float = 0.2,
    frame: Optional[Any] = None,
    magnitude_framings: Optional[list] = None,
    anomaly_flags: Optional[list] = None,
    followups: Optional[list] = None,
    peer_contexts: Optional[list] = None,
) -> SynthesizedAnswer:
    """Run the synthesizer LLM call. Returns a fully-populated
    SynthesizedAnswer (prose + key_findings + caveats + citations).

    Citations are derived from ``plan`` deterministically — the LLM is
    not allowed to mint them.

    Phase 4 extras (frame, magnitude_framings, anomaly_flags,
    followups) and Phase 3 extras (peer_contexts) are optional. When
    provided, the system prompt teaches the LLM to weave them into the
    prose instead of inventing comparator phrasings, trend language,
    or peer-metro claims.
    """
    user_payload = _build_user_payload(
        query, intent, aggregated,
        frame=frame,
        magnitude_framings=magnitude_framings,
        anomaly_flags=anomaly_flags,
        followups=followups,
        peer_contexts=peer_contexts,
    )
    schema = _SynthesisLLMOutput.model_json_schema()

    try:
        raw = llm.extract(
            system_prompt=_SYSTEM_PROMPT,
            user_text=user_payload,
            schema=schema,
            temperature=temperature,
        )
    except LLMCallError as e:
        raise SynthesisError(f"synthesizer LLM call failed: {e}") from e

    if not isinstance(raw, dict):
        raise SynthesisError(
            f"synthesizer returned non-dict: {type(raw).__name__}"
        )

    try:
        out = _SynthesisLLMOutput.model_validate(raw)
    except ValidationError as e:
        raise SynthesisError(
            f"synthesizer output failed validation: {e}"
        ) from e

    citations = _build_citations(plan, aggregated)
    return SynthesizedAnswer(
        prose=out.prose,
        key_findings=out.key_findings,
        caveats=out.caveats,
        citations=citations,
    )
