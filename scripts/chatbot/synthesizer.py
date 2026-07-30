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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
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


_BUILTIN_SYSTEM_PROMPT = """\
You are a multi-dataset data synthesizer. Given a user query, a structured
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


# config/chatbot.yaml has always declared paths.synthesizer_prompt as this
# file. The constant above is the fallback when it's missing or broken —
# the pipeline must never fail to answer because someone saved bad YAML.
SYNTHESIZER_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "v1" / "synthesizer.yaml"
)


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Return the synthesizer system prompt, read fresh from disk.

    Re-read on every call so an edit takes effect on the next synthesis
    with no process restart — that's what makes the app's prompt-
    iteration loop work.
    """
    p = path or SYNTHESIZER_PROMPT_PATH
    try:
        doc = yaml.safe_load(p.read_text())
        text = (doc or {}).get("system")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing or empty 'system' key")
        return text
    except FileNotFoundError:
        return _BUILTIN_SYSTEM_PROMPT
    except Exception as e:
        logger.warning(
            "synthesizer prompt at %s is unusable (%s); using the built-in "
            "prompt", p, e,
        )
        return _BUILTIN_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# The synthesis bundle
# ---------------------------------------------------------------------------
#
# Everything the synthesizer might use is collected into one dict by
# ``build_synthesis_bundle``. ``synthesize`` takes that dict and nothing
# else. The point of the split is that upstream dependencies change often
# (a new dataset adds artifacts; a new node adds context) while the
# write-up step changes for entirely different reasons. Keeping them on
# opposite sides of a dict means:
#
#   * adding an upstream artifact is adding a key — no signature change,
#     no caller change, and it shows up in the app's inventory panel
#     automatically;
#   * changing how answers read means editing ``synthesize`` and the
#     prompt, with one well-known input to reason about.
#
# A key with no BUNDLE_SPECS entry still works: it is inspectable and
# renderable, just not sent to the model unless explicitly enabled. New
# artifacts therefore never silently change what the LLM sees (or what a
# turn costs) — someone has to opt in.


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key access, so bundle entries may be Pydantic models
    or plain dicts interchangeably."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _dump(obj: Any) -> Any:
    """Best-effort JSON-able rendering of an arbitrary bundle value."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    return str(obj)


@dataclass(frozen=True)
class BundleSpec:
    """How one bundle key is described, and how it reaches the LLM."""

    description: str
    default_send: bool
    # (value, bundle, options) -> dict merged into the LLM payload.
    render: Callable[[Any, dict, dict], dict]
    # False for structural entries used by code (citations, lints) but
    # never sent to the model.
    sendable: bool = True


def _render_query(v, b, o):
    return {"user_query": v}


def _render_intent(v, b, o):
    return {"intent_summary": {
        "intent_type": _attr(v, "intent_type"),
        "comparison_implied": _attr(v, "comparison_implied"),
        "national_comparison_implied": _attr(
            v, "national_comparison_implied",
        ),
        "temporal_intent": _attr(v, "temporal_intent"),
        "explicit_years": _attr(v, "years", []),
        "analyses": _dump(_attr(v, "analyses", [])),
    }}


def _render_values(v, b, o):
    values = list(v or [])
    cap = o.get("max_values_sent")
    out: dict[str, Any] = {}
    if isinstance(cap, int) and cap > 0 and len(values) > cap:
        # Tell the model it was capped rather than letting it assume it
        # saw everything and write a false "across all geographies".
        out["values_truncated"] = len(values) - cap
        values = values[:cap]
    out["aggregated_values"] = [_format_value(x) for x in values]
    return out


def _render_failures(v, b, o):
    return {"fetch_failures": list(v or [])}


def _render_frame(v, b, o):
    return {"frame": {
        "name": _attr(v, "name", ""),
        "rhetorical_target": _attr(v, "rhetorical_target", ""),
        "standard_caveats": list(_attr(v, "standard_caveats", []) or []),
    }}


def _render_list(payload_key: str):
    def render(v, b, o):
        return {payload_key: [_dump(x) for x in (v or [])]}
    return render


def _render_peers(v, b, o):
    # Compacted on purpose: the LLM needs axis, scope, and concrete
    # feature values to cite peers. Raw distance scores are omitted —
    # they aren't narratively useful.
    return {"peer_contexts": [
        {
            "axis": _attr(c, "axis", ""),
            "axis_description": _attr(c, "axis_description", ""),
            "pool_scope": _attr(c, "pool_scope", ""),
            "anchor_geo_name": _attr(c, "anchor_geo_name", ""),
            "anchor_feature_values": _attr(c, "anchor_feature_values", {}) or {},
            "peers": [
                {
                    "geo_name": _attr(p, "geo_name", ""),
                    "population": _attr(p, "population", None),
                    "match_explanation": _attr(p, "match_explanation", ""),
                    "feature_values": _attr(p, "feature_values", {}) or {},
                }
                for p in (_attr(c, "peers", []) or [])
            ],
        }
        for c in (v or [])
    ]}


def _render_concept_resolutions(v, b, o):
    return {"concept_resolutions": [
        {
            "concept": _attr(_attr(r, "concept"), "text", ""),
            "tier": _attr(r, "tier", ""),
            "notes": list(_attr(r, "notes", []) or []),
        }
        for r in (v or [])
    ]}


# Order matters: it's the order keys appear in the LLM payload and in the
# inventory panel. Defaults reproduce the pre-bundle payload exactly.
BUNDLE_SPECS: dict[str, BundleSpec] = {
    "query": BundleSpec(
        "The user's question, verbatim.", True, _render_query),
    "intent": BundleSpec(
        "Parsed intent: comparison/temporal flags and explicit years.",
        True, _render_intent),
    "aggregated_values": BundleSpec(
        "The retrieved numbers. Every figure the answer may state.",
        True, _render_values),
    "fetch_failures": BundleSpec(
        "Calls that failed, so the answer can flag partial data.",
        True, _render_failures),
    "frame": BundleSpec(
        "Matched grant frame — sets the rhetorical target of the prose.",
        True, _render_frame),
    "magnitude_framings": BundleSpec(
        "Comparator ratios vs county/MSA/state/US, plus trend labels.",
        True, _render_list("magnitude_framings")),
    "anomaly_flags": BundleSpec(
        "Values far outside their comparators — headline material.",
        True, _render_list("anomaly_flags")),
    "followups": BundleSpec(
        "Suggested next questions.", True, _render_list("suggested_followups")),
    "peer_contexts": BundleSpec(
        "Peer geographies with feature values. The UI renders these in "
        "their own section, so sending them to the model risks duplicate "
        "or conflicting peer talk in the prose.",
        True, _render_peers),
    "concept_resolutions": BundleSpec(
        "Which tier resolved each concept, and any routing notes. Useful "
        "for teaching the model to hedge on low-confidence routing.",
        False, _render_concept_resolutions),
    # Structural — used for citations and output checks, never sent.
    "plan": BundleSpec(
        "The query plan. Citations are derived from it deterministically.",
        False, lambda v, b, o: {}, sendable=False),
    "aggregated": BundleSpec(
        "The full AggregatedResult, for citation building and lints.",
        False, lambda v, b, o: {}, sendable=False),
    "resolved_geos": BundleSpec(
        "Resolved geographies with tract lists and confidence.",
        False, lambda v, b, o: {}, sendable=False),
}


def build_synthesis_bundle(
    query: str,
    intent: ExtractedIntent,
    plan: PlanResult,
    aggregated: AggregatedResult,
    *,
    resolved_geos: Optional[list] = None,
    frame: Optional[Any] = None,
    magnitude_framings: Optional[list] = None,
    anomaly_flags: Optional[list] = None,
    followups: Optional[list] = None,
    peer_contexts: Optional[list] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Collect everything the synthesizer might use into one dict.

    Keys whose value is empty are omitted, so ``"peer_contexts" in
    bundle`` is a meaningful test of "did the pipeline actually produce
    peers for this query".

    ``**extra`` accepts artifacts this function doesn't know about — a
    new dataset can contribute context without editing this signature.
    Unknown keys are inspectable and renderable but are not sent to the
    model unless enabled.
    """
    bundle: dict[str, Any] = {
        "query": query,
        "intent": intent,
        "plan": plan,
        "aggregated": aggregated,
        "aggregated_values": list(getattr(aggregated, "values", []) or []),
        "fetch_failures": list(getattr(aggregated, "fetch_failures", []) or []),
    }
    optional = {
        "resolved_geos": resolved_geos,
        "frame": frame,
        "magnitude_framings": magnitude_framings,
        "anomaly_flags": anomaly_flags,
        "followups": followups,
        "peer_contexts": peer_contexts,
    }
    for key, value in optional.items():
        if value:
            bundle[key] = value
    for key, value in extra.items():
        if value:
            bundle[key] = value
    return bundle


def bundle_inventory(
    bundle: dict[str, Any],
    options: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Describe every key in the bundle: what it is, whether it's being
    sent to the model, how big it is, and a preview.

    This is what the app's upstream panel renders. Keys with no spec are
    included and marked unknown, so a newly added artifact shows up
    without anyone updating a list.
    """
    opts = options or {}
    sends = opts.get("send", {}) or {}
    out: list[dict[str, Any]] = []

    ordered = [k for k in BUNDLE_SPECS if k in bundle]
    ordered += [k for k in bundle if k not in BUNDLE_SPECS]

    for key in ordered:
        value = bundle[key]
        spec = BUNDLE_SPECS.get(key)
        size = len(value) if isinstance(value, (list, tuple, dict)) else None
        preview = _dump(value)
        if isinstance(preview, list):
            preview = preview[:2]
        text = json.dumps(preview, ensure_ascii=False, default=str)
        out.append({
            "key": key,
            "known": spec is not None,
            "description": spec.description if spec else
                           "Not registered in BUNDLE_SPECS — available to "
                           "the pipeline but off by default.",
            "sendable": spec.sendable if spec else True,
            "sent": _should_send(key, spec, sends),
            "type": type(value).__name__,
            "count": size,
            "preview": text[:600] + ("…" if len(text) > 600 else ""),
        })
    return out


def _should_send(key: str, spec: Optional[BundleSpec], sends: dict) -> bool:
    if spec is not None and not spec.sendable:
        return False
    if key in sends:
        return bool(sends[key])
    if spec is not None:
        return spec.default_send
    return False        # unknown keys are opt-in


def render_bundle_payload(
    bundle: dict[str, Any],
    options: Optional[dict] = None,
) -> str:
    """Render the JSON payload the LLM sees from the bundle.

    ``options`` may carry ``send`` (per-key overrides of the spec
    defaults) and ``max_values_sent``. With no options the output is
    byte-identical to the pre-bundle payload builder.
    """
    opts = options or {}
    sends = opts.get("send", {}) or {}
    payload: dict[str, Any] = {}

    ordered = [k for k in BUNDLE_SPECS if k in bundle]
    ordered += [k for k in bundle if k not in BUNDLE_SPECS]

    for key in ordered:
        spec = BUNDLE_SPECS.get(key)
        if not _should_send(key, spec, sends):
            continue
        value = bundle[key]
        try:
            if spec is not None:
                payload.update(spec.render(value, bundle, opts))
            else:
                payload[key] = _dump(value)
        except Exception as e:                       # pragma: no cover
            logger.warning("bundle key %r failed to render: %s", key, e)
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(
    bundle: dict[str, Any],
    llm: LLMClient,
    *,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
) -> SynthesizedAnswer:
    """Turn a synthesis bundle into a written answer.

    One input carries every upstream artifact, so changing how answers
    read means editing this function and the prompt — not chasing a
    parameter list that grows with every new pipeline stage.

    Citations are derived from ``bundle['plan']`` deterministically; the
    LLM is not allowed to mint them.

    ``system_prompt`` overrides the on-disk prompt for one call (the
    app's "try without saving"). ``options`` gates which bundle keys
    reach the model; with none, the payload matches the pre-bundle
    behavior exactly.
    """
    user_payload = render_bundle_payload(bundle, options)
    schema = _SynthesisLLMOutput.model_json_schema()
    active_prompt = system_prompt or load_system_prompt()

    try:
        raw = llm.extract(
            system_prompt=active_prompt,
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

    citations = _build_citations(bundle.get("plan"), bundle.get("aggregated"))
    return SynthesizedAnswer(
        prose=out.prose,
        key_findings=out.key_findings,
        caveats=out.caveats,
        citations=citations,
    )


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
    **kwargs: Any,
) -> SynthesizedAnswer:
    """Backward-compatible wrapper over ``build_synthesis_bundle`` +
    ``synthesize``.

    Kept so existing callers and tests keep working unchanged. New code
    should build a bundle and call ``synthesize`` directly, which is the
    path that supports per-key send control and the app's inventory.
    """
    bundle = build_synthesis_bundle(
        query, intent, plan, aggregated,
        frame=frame,
        magnitude_framings=magnitude_framings,
        anomaly_flags=anomaly_flags,
        followups=followups,
        peer_contexts=peer_contexts,
    )
    return synthesize(
        bundle, llm,
        temperature=temperature,
        system_prompt=kwargs.get("system_prompt"),
        options=kwargs.get("options"),
    )
