"""The public API contract — what the frontend binds to.

Deliberately NOT ``orchestrator.QueryResponse``. That object exposes
``ExtractedIntent``, ``PlanResult``, ``AggregatedResult``, ``FrameMatch``
and friends — internal shapes that will churn as datasets are added. If
the UI binds to those, backend work breaks the frontend.

``to_public()`` is the firewall. When an internal shape changes, this one
function changes and the contract holds.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.lints import Lint


class Citation(BaseModel):
    """A source the answer draws on. Derived from the plan, never from
    the model — citations here cannot be hallucinated.

    ``title`` and ``variables`` are plain-English when the metadata DB
    has them (table title, per-variable label) — never a client-side
    guess at what a table code means, only what the pipeline actually
    looked up. ``variables`` falls back to the raw variable id when a
    label isn't in the catalog, so the list is never empty."""

    table_id: str = ""
    dataset: str = ""
    year: Optional[int] = None
    title: str = ""
    universe: str = ""
    variables: list[str] = Field(default_factory=list)
    url: str = ""


class GeoOut(BaseModel):
    name: str = ""
    geo_level: str = ""
    geoid: str = ""
    confidence: Optional[float] = None
    tract_count: Optional[int] = None


class ValueOut(BaseModel):
    """One retrieved number, flattened for display.

    Dataset-agnostic on purpose: ``value`` carries medians and counts,
    ``ratio`` carries rates. HMDA loan counts and denial rates fit the
    same two fields as ACS medians and poverty rates.
    """

    geo: str = ""
    concept: str = ""
    year: Optional[int] = None
    role: str = "primary"
    value: Optional[float] = None
    ratio: Optional[float] = None
    components: Optional[dict[str, float]] = None
    sample_size: Optional[float] = None
    units: str = ""
    caveat: Optional[str] = None


class PeerOut(BaseModel):
    geo_name: str = ""
    population: Optional[int] = None
    match_explanation: str = ""
    # The real numbers behind match_explanation's prose — whichever of
    # the axis's features PeerRef actually populated for this peer, so
    # the UI can show e.g. "Median family income: $65,200" instead of
    # just naming the feature. Never a guess: same dict the pipeline
    # itself scored the match on.
    feature_values: dict[str, float] = Field(default_factory=dict)


class PeerGroupOut(BaseModel):
    axis: str = ""
    axis_description: str = ""
    anchor_geo_name: str = ""
    peers: list[PeerOut] = Field(default_factory=list)
    # The anchor's own values for this axis's features — the benchmark
    # every peer in `peers` is actually being compared against.
    anchor_feature_values: dict[str, float] = Field(default_factory=dict)
    # The ACS vintage this comparison actually ran against — reported by
    # PeerContext.anchor_vintage, which the pipeline resolves against
    # data/metadata/peer_features.sqlite's own `vintage` column at query
    # time. Read straight through, never inferred or hardcoded here;
    # None when the pipeline itself couldn't resolve one.
    year: Optional[int] = None


class AnomalyOut(BaseModel):
    concept: str = ""
    geo: str = ""
    kind: str = ""
    magnitude: Optional[float] = None
    context: str = ""


class FollowupOut(BaseModel):
    question: str = ""
    rationale: str = ""


class UpstreamItem(BaseModel):
    """One artifact available to the synthesizer for this query.

    Derived from the synthesis bundle's keys, so a new artifact appears
    here without anyone updating a list — including artifacts a future
    dataset contributes.
    """

    key: str
    known: bool = True
    description: str = ""
    sendable: bool = True
    sent: bool = False
    type: str = ""
    count: Optional[int] = None
    preview: str = ""


class LLMCallOut(BaseModel):
    """One LLM call made anywhere in the pipeline."""

    node: str = ""
    seconds: float = 0.0
    system_prompt: str = ""
    user_text: str = ""
    response: Optional[Any] = None
    error: Optional[str] = None


class TraceOut(BaseModel):
    """What the model was actually sent and returned.

    Captured at the client boundary, so ``calls`` covers every node, not
    only the synthesizer.
    """

    system_prompt: str = ""
    user_payload: str = ""
    raw_response: Optional[Any] = None
    bundle_options: dict[str, Any] = Field(default_factory=dict)
    calls: list[LLMCallOut] = Field(default_factory=list)


class MetricsOut(BaseModel):
    total_s: float = 0.0
    extract_s: float = 0.0
    resolve_s: float = 0.0
    plan_s: float = 0.0
    fetch_s: float = 0.0
    synthesize_s: float = 0.0
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    census_calls_total: int = 0
    census_cache_hits: int = 0
    census_failures: int = 0
    record_calls_total: int = 0
    record_cache_hits: int = 0
    record_failures: int = 0


class ChatResponse(BaseModel):
    """What the frontend renders. Stable across backend churn."""

    # "ok" | "no_data" | "error"
    # Reserved so adding them later isn't breaking: "needs_clarification",
    # "out_of_scope".
    status: str = "ok"
    query: str = ""

    prose: str = ""
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    geos: list[GeoOut] = Field(default_factory=list)
    values: list[ValueOut] = Field(default_factory=list)
    peer_groups: list[PeerGroupOut] = Field(default_factory=list)
    anomalies: list[AnomalyOut] = Field(default_factory=list)
    followups: list[FollowupOut] = Field(default_factory=list)

    lints: list[Lint] = Field(default_factory=list)
    upstream: list[UpstreamItem] = Field(default_factory=list)
    trace: Optional[TraceOut] = None
    metrics: Optional[MetricsOut] = None

    error: Optional[str] = None
    config_warning: Optional[str] = None
    run_id: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    # Prior-turn context (geo/concept the previous answer resolved), so a
    # follow-up like "what does this mean for housing affordability here"
    # can resolve "here" without restating the geography. Optional and
    # additive — omitting it is exactly today's stateless behavior.
    session_ctx: Optional[dict] = None


class ResynthesizeRequest(BaseModel):
    """Re-run only the synthesizer against a cached run's bundle."""

    run_id: str
    system_prompt: Optional[str] = None      # unsaved draft


class RerenderRequest(BaseModel):
    """Re-apply layers 3+4 to a cached run. No LLM call."""

    run_id: str


# ---------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------

def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _geo_name(geo: Any) -> str:
    for key in ("display_name", "name", "matched_name", "user_text"):
        v = _attr(geo, key)
        if v:
            return str(v)
    return ""


def _concept_name(concept: Any) -> str:
    for key in ("canonical_hint", "text", "name", "raw_text"):
        v = _attr(concept, key)
        if v:
            return str(v)
    return ""


def to_public(
    resp: Any,
    *,
    sections: dict[str, Any],
    lint_results: Optional[list[Lint]] = None,
    upstream: Optional[list[dict]] = None,
    trace: Optional[dict] = None,
    bundle_options: Optional[dict] = None,
    answer_override: Any = None,
    config_warning: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ChatResponse:
    """Map a pipeline QueryResponse onto the public contract.

    ``sections`` gates what is included — a section switched off is
    omitted from the payload entirely, not merely hidden by CSS.

    ``answer_override`` carries the post-processed answer, so layer-3
    cleanup is what reaches the user while the trace still shows the
    model's raw output.
    """
    def shown(name: str) -> bool:
        return bool(sections.get(name, True))

    answer = answer_override if answer_override is not None else _attr(resp, "answer")
    aggregated = _attr(resp, "aggregated")
    error = _attr(resp, "error")

    status = "error" if error else ("ok" if answer is not None else "no_data")

    out = ChatResponse(
        status=status,
        query=_attr(resp, "query", "") or "",
        error=error,
        config_warning=config_warning,
        run_id=run_id,
    )

    if answer is not None:
        if shown("prose"):
            out.prose = _attr(answer, "prose", "") or ""
        if shown("key_findings"):
            out.key_findings = list(_attr(answer, "key_findings", []) or [])
        if shown("caveats"):
            out.caveats = list(_attr(answer, "caveats", []) or [])
        if shown("citations"):
            for c in _attr(answer, "citations", []) or []:
                out.citations.append(Citation(
                    table_id=str(_attr(c, "table_id", "") or ""),
                    dataset=str(_attr(c, "dataset", "") or ""),
                    year=_attr(c, "year"),
                    # The internal Citation calls this `table_title`, not
                    # `title` — reading the wrong name here silently sent
                    # an empty string to the frontend on every citation,
                    # even when the pipeline had the real table title.
                    title=str(_attr(c, "table_title", "") or ""),
                    universe=str(_attr(c, "universe", "") or ""),
                    # Each entry is a CitedVariable(variable_id, label).
                    # Prefer the plain-English label the metadata DB
                    # looked up; fall back to the raw variable id rather
                    # than `str(obj)`, which stringified the whole
                    # pydantic object instead of a name.
                    variables=[
                        str(_attr(v, "label") or _attr(v, "variable_id", "") or "")
                        for v in (_attr(c, "variables", []) or [])
                    ],
                    url=str(_attr(c, "url", "") or ""),
                ))

    for g in _attr(resp, "resolved_geos", []) or []:
        tracts = _attr(g, "tract_geoids") or []
        out.geos.append(GeoOut(
            name=_geo_name(g),
            geo_level=str(_attr(g, "geo_level", "") or ""),
            geoid=str(_attr(g, "geo_id", "") or _attr(g, "geoid", "") or ""),
            confidence=_attr(g, "confidence"),
            tract_count=len(tracts) if tracts else None,
        ))

    if shown("values_table") and aggregated is not None:
        for v in _attr(aggregated, "values", []) or []:
            concept = _attr(v, "concept")
            out.values.append(ValueOut(
                geo=_geo_name(_attr(v, "geo")),
                concept=_concept_name(concept),
                year=_attr(v, "year"),
                role=str(_attr(v, "role", "primary") or "primary"),
                value=_attr(v, "value"),
                ratio=_attr(v, "ratio"),
                components=_attr(v, "components"),
                sample_size=_attr(v, "sample_size"),
                units=str(_attr(concept, "units", "") or ""),
                caveat=_attr(v, "aggregation_caveat"),
            ))

    if shown("peer_contexts"):
        for ctx in _attr(resp, "peer_contexts", []) or []:
            ctx_peers = _attr(ctx, "peers", []) or []
            out.peer_groups.append(PeerGroupOut(
                axis=str(_attr(ctx, "axis", "") or ""),
                axis_description=str(_attr(ctx, "axis_description", "") or ""),
                anchor_geo_name=str(_attr(ctx, "anchor_geo_name", "") or ""),
                anchor_feature_values=dict(
                    _attr(ctx, "anchor_feature_values", {}) or {}),
                # The pipeline now resolves and reports the ACS vintage
                # it actually queried against (PeerContext.anchor_vintage)
                # — read straight through, never re-derived here.
                year=_attr(ctx, "anchor_vintage"),
                peers=[
                    PeerOut(
                        geo_name=str(_attr(p, "geo_name", "") or ""),
                        population=_attr(p, "population"),
                        match_explanation=str(
                            _attr(p, "match_explanation", "") or ""),
                        feature_values=dict(
                            _attr(p, "feature_values", {}) or {}),
                    )
                    for p in ctx_peers
                ],
            ))

    if shown("anomaly_flags"):
        for a in _attr(resp, "anomaly_flags", []) or []:
            out.anomalies.append(AnomalyOut(
                concept=_concept_name(_attr(a, "concept")),
                geo=_geo_name(_attr(a, "geo")),
                kind=str(_attr(a, "anomaly_kind", "") or ""),
                magnitude=_attr(a, "magnitude"),
                context=str(_attr(a, "context", "") or ""),
            ))

    if shown("followups"):
        for f in _attr(resp, "followups", []) or []:
            out.followups.append(FollowupOut(
                question=str(_attr(f, "question", "") or ""),
                rationale=str(_attr(f, "rationale", "") or ""),
            ))

    out.lints = list(lint_results or [])

    if shown("upstream") and upstream is not None:
        out.upstream = [UpstreamItem(**row) for row in upstream]

    if shown("trace") and trace is not None:
        out.trace = TraceOut(
            system_prompt=trace.get("system_prompt", "") or "",
            user_payload=trace.get("user_payload", "") or "",
            raw_response=trace.get("raw_response"),
            bundle_options=bundle_options or {},
            calls=[LLMCallOut(**c) for c in (trace.get("calls") or [])],
        )

    if shown("metrics"):
        m = _attr(resp, "metrics")
        if m is not None:
            out.metrics = MetricsOut(
                total_s=_attr(m, "total_s", 0.0) or 0.0,
                extract_s=_attr(m, "extract_s", 0.0) or 0.0,
                resolve_s=_attr(m, "resolve_s", 0.0) or 0.0,
                plan_s=_attr(m, "plan_s", 0.0) or 0.0,
                fetch_s=_attr(m, "fetch_s", 0.0) or 0.0,
                synthesize_s=_attr(m, "synthesize_s", 0.0) or 0.0,
                llm_calls=_attr(m, "llm_calls", 0) or 0,
                llm_cost_usd=_attr(m, "llm_cost_usd", 0.0) or 0.0,
                census_calls_total=_attr(m, "census_calls_total", 0) or 0,
                census_cache_hits=_attr(m, "census_cache_hits", 0) or 0,
                census_failures=_attr(m, "census_failures", 0) or 0,
                record_calls_total=_attr(m, "record_calls_total", 0) or 0,
                record_cache_hits=_attr(m, "record_cache_hits", 0) or 0,
                record_failures=_attr(m, "record_failures", 0) or 0,
            )

    return out
