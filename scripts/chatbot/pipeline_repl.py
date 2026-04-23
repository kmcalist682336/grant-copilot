"""End-to-end chatbot REPL — Phases C through I.

Three modes per turn (pick with flags):
  default       extract → resolve → concept-lookup tier preview
  --plan        + planner output (no HTTP)
  --execute     + decompose-on-miss → fetch → aggregate → synthesize

Per-turn rendering:
  - EXTRACTION   geo_refs, compounds, concepts, temporal intent
  - GEO RESOL.   per-operand ResolvedGeography
  - CONCEPT      tier-cascade preview (Tier 1 hit / Tier 2 candidates)
  - PLAN         (when --plan or --execute) APIPlanCalls grouped
  - DATA         (when --execute) AggregatedValue per (geo, concept, year)
  - ANSWER       (when --execute) synthesized prose + key findings + caveats
  - METRICS      latency, LLM tokens, Census calls/cache hits, cost

Usage:
    python -m scripts.chatbot.pipeline_repl                          # live Vertex
    python -m scripts.chatbot.pipeline_repl --mock-llm               # offline
    python -m scripts.chatbot.pipeline_repl --query "..."            # one-shot
    python -m scripts.chatbot.pipeline_repl --raw                    # dump JSON
    python -m scripts.chatbot.pipeline_repl --plan                   # show plan
    python -m scripts.chatbot.pipeline_repl --execute                # full pipeline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Enable GNU readline for input() — gives us arrow-key history, word
# nav, and — critically — correct cursor handling when a long query
# wraps to a second terminal row. Without this, backspace across a
# line break can delete from the wrong column (or appear to do
# nothing) because Python's default input() uses cooked-mode tty.
try:
    import readline  # noqa: F401
except ImportError:                                    # pragma: no cover
    pass

import yaml

# Readline ignore-markers for ANSI escape sequences embedded in
# input() prompts. Without these, readline counts the color-code
# bytes as visible characters, mis-estimates prompt width, and
# botches backspace / cursor position after line wraps. \001 and
# \002 bracket spans that should contribute zero width.
_RL_START = "\001"
_RL_END = "\002"

from scripts.chatbot.concept_map import ConceptEntry, ConceptMap
from scripts.chatbot.extractor import ExtractorError, extract_intent
from scripts.chatbot.gazetteer_db import open_spatialite
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.llm_client import (
    LLMClient, MockLLMClient, build_default_client, estimate_cost_usd,
)
from scripts.chatbot.metadata_search import (
    open_search_index, search_tables, has_successful_coverage,
)
from scripts.chatbot.models import (
    ExtractedConcept, ExtractedIntent, ResolvedGeography,
)
from scripts.chatbot.planner import (
    ConceptResolution, PlanResult, PlannedCall, plan_query,
)
from scripts.chatbot.api_cache import APICache
from scripts.chatbot.decomposition_cache import DecompositionCache
from scripts.chatbot.orchestrator import QueryResponse, answer_query_sync
from scripts.chatbot.synthesizer import SynthesizedAnswer


REPO_ROOT = Path(__file__).resolve().parents[2]

# ANSI colors
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _load_config() -> dict:
    with (REPO_ROOT / "config" / "chatbot.yaml").open() as f:
        return yaml.safe_load(f)


def _build_mock_llm() -> MockLLMClient:
    """Mock seeded from gold_queries.yaml — same pattern as geo_repl."""
    gold_path = REPO_ROOT / "tests" / "fixtures" / "gold_queries.yaml"
    with gold_path.open() as f:
        gold = yaml.safe_load(f)
    responses: dict[str, dict] = {}
    for entry in gold.get("entries", []):
        exp = entry.get("expected", {})
        responses[entry["query"].lower()] = {
            "geo_refs": exp.get("geo_refs", []),
            "compounds": exp.get("compounds", []),
            "concepts": exp.get("concepts", []),
            "temporal_intent": exp.get("temporal_intent", "latest"),
            "years": exp.get("years", []),
            "intent_type": exp.get("intent_type", "exact"),
            "comparison_implied": exp.get("comparison_implied", False),
            "national_comparison_implied":
                exp.get("national_comparison_implied", False),
        }
    return MockLLMClient(responses=responses)


def _build_live_llm(config: dict) -> LLMClient:
    from scripts.chatbot.auth_check import AuthCheckError, check_auth
    try:
        auth = check_auth(verbose=True)
    except AuthCheckError as e:
        print(f"\nLive LLM unavailable:\n{e}\n", file=sys.stderr)
        sys.exit(1)
    config.setdefault("vertex_ai", {})["project_id"] = auth["project_id"]
    config["vertex_ai"].setdefault(
        "location", auth.get("location", "us-central1"),
    )
    return build_default_client(config)


# ---------------------------------------------------------------------------
# Geo level inference (which level to require coverage at, per resolved geo)
# ---------------------------------------------------------------------------

# Map ResolvedGeography.geo_level → the Census-API geo_level we'd query.
# neighborhoods/regions/corridors/landmark_areas all aggregate tracts,
# so the underlying data demand is "tract".
_GEO_LEVEL_TO_DATA_LEVEL = {
    "tract": "tract",
    "neighborhood": "tract",
    "region": "tract",
    "corridor": "tract",
    "landmark_area": "tract",
    "county": "county",
    "place": "place",
    "msa": "msa",
    "state": "state",
    "us": "us",
    "school_district_unified": "school_district_unified",
    "zcta": "zcta",
}


def _data_levels_for(resolved: list[ResolvedGeography]) -> list[str]:
    """Distinct Census-API geo_levels implied by the resolved geographies."""
    out: list[str] = []
    seen: set[str] = set()
    for r in resolved:
        lvl = _GEO_LEVEL_TO_DATA_LEVEL.get(r.geo_level, r.geo_level)
        if lvl not in seen:
            seen.add(lvl)
            out.append(lvl)
    return out


# ---------------------------------------------------------------------------
# Rendering — extraction
# ---------------------------------------------------------------------------

def _render_extraction(intent: ExtractedIntent) -> None:
    print(f"  {_DIM}──── EXTRACTION ────{_RESET}")
    print(f"    intent_type: {_BOLD}{intent.intent_type}{_RESET}", end="")
    flags = []
    if intent.comparison_implied:
        flags.append("comparison_implied")
    if intent.national_comparison_implied:
        flags.append("national_comparison_implied")
    if flags:
        print(f"  {_DIM}({', '.join(flags)}){_RESET}", end="")
    print()

    if not intent.geo_refs:
        print(f"    {_DIM}(no geo_refs extracted){_RESET}")
    for i, ref in enumerate(intent.geo_refs):
        bits = [
            f"text={_BOLD}{ref.text!r}{_RESET}",
            f"ref_type={_CYAN}{ref.ref_type}{_RESET}",
        ]
        if ref.qualifier:
            bits.append(f"qualifier={ref.qualifier}")
        if ref.proximity_tier:
            bits.append(f"tier={ref.proximity_tier}")
        if ref.scope_hint:
            bits.append(f"scope={ref.scope_hint!r}")
        print(f"    geo_refs[{i}]: " + "  ".join(bits))

    for j, c in enumerate(intent.compounds):
        extra = f", direction={c.direction}" if c.direction else ""
        print(
            f"    {_MAGENTA}compounds[{j}]{_RESET}: op={c.op}, "
            f"primary={c.primary_ref_idx}, filter={c.filter_ref_idx}{extra}"
        )

    if not intent.concepts:
        print(f"    {_DIM}(no concepts extracted){_RESET}")
    for k, con in enumerate(intent.concepts):
        bits = [f"text={_BOLD}{con.text!r}{_RESET}"]
        if con.canonical_hint:
            bits.append(f"hint={_CYAN}{con.canonical_hint!r}{_RESET}")
        if con.is_composite:
            bits.append(f"{_YELLOW}composite{_RESET}")
        print(f"    concepts[{k}]: " + "  ".join(bits))

    if intent.temporal_intent != "latest" or intent.years:
        ys = f" years={intent.years}" if intent.years else ""
        print(
            f"    {_MAGENTA}temporal{_RESET}: "
            f"{intent.temporal_intent}{ys}"
        )


# ---------------------------------------------------------------------------
# Rendering — geo resolution
# ---------------------------------------------------------------------------

def _render_geo_resolution(resolved: list[ResolvedGeography]) -> None:
    print(f"\n  {_DIM}──── GEO RESOLUTION ────{_RESET}")
    if not resolved:
        print(f"    {_RED}(no geographies resolved){_RESET}")
        return
    for r in resolved:
        tract_blurb = (
            f"tracts={len(r.tract_geoids)}" if r.tract_geoids else "tracts=0"
        )
        conf_color = (
            _GREEN if r.confidence >= 0.95
            else _CYAN if r.confidence >= 0.80
            else _YELLOW
        )
        print(
            f"    ✓ {_BOLD}{r.display_name}{_RESET} "
            f"{_DIM}({r.geo_level}){_RESET}  "
            f"{tract_blurb}   "
            f"{conf_color}conf={r.confidence:.2f}{_RESET}   "
            f"{_DIM}data={r.data_level_available}{_RESET}"
        )


# ---------------------------------------------------------------------------
# Rendering — concept lookup
# ---------------------------------------------------------------------------

def _lookup_concept(
    cmap: ConceptMap, concept: ExtractedConcept,
) -> tuple[Optional[ConceptEntry], Optional[str]]:
    """Try canonical_hint first, then verbatim text. Return (entry, key_used)."""
    if concept.canonical_hint:
        hit = cmap.lookup(concept.canonical_hint)
        if hit is not None:
            return hit, concept.canonical_hint
    hit = cmap.lookup(concept.text)
    if hit is not None:
        return hit, concept.text
    return None, None


def _render_tier1_hit(
    entry: ConceptEntry,
    matched_key: str,
    data_levels: list[str],
    metadata_db,
) -> None:
    print(
        f"      {_GREEN}TIER 1 HIT{_RESET} via {matched_key!r} → "
        f"{_BOLD}{entry.canonical}{_RESET}"
    )
    if entry.units:
        print(f"        units: {entry.units}")
    base_vars = ", ".join(entry.variables.all_variable_ids())
    print(
        f"        base: {entry.dataset}/{entry.table_id}  "
        f"vars=[{base_vars}]"
    )
    # Show effective resolution per data level, with override deltas
    for lvl in data_levels:
        eff = entry.resolve_for(lvl)
        ovr = entry.by_geo_level.get(lvl)
        cov_ok = has_successful_coverage(
            metadata_db, eff.table_id, eff.dataset,
            year=2023, geo_levels=[lvl],
        )
        # Try a couple of years if 2023 misses.
        if not cov_ok:
            for y in (2024, 2022, 2021, 2020):
                if has_successful_coverage(
                    metadata_db, eff.table_id, eff.dataset,
                    year=y, geo_levels=[lvl],
                ):
                    cov_ok = True
                    break
        marker = (
            f"{_GREEN}✓{_RESET}" if cov_ok else f"{_RED}✗{_RESET}"
        )
        if ovr is not None:
            ovr_bits = []
            if ovr.dataset and ovr.dataset != entry.dataset:
                ovr_bits.append(f"dataset={ovr.dataset}")
            if ovr.table_id and ovr.table_id != entry.table_id:
                ovr_bits.append(f"table_id={ovr.table_id}")
            if ovr.variables:
                ovr_bits.append("variables=overridden")
            note = f" {_MAGENTA}override{_RESET}: {', '.join(ovr_bits)}"
        else:
            note = ""
        print(
            f"        {marker} {lvl:8s} → {eff.dataset}/{eff.table_id}{note}"
        )


def _render_tier2_fallback(
    concept: ExtractedConcept,
    data_levels: list[str],
    metadata_db,
) -> None:
    print(
        f"      {_YELLOW}TIER 1 MISS{_RESET} → falling back to FTS metadata search"
    )
    if concept.is_composite:
        print(
            f"        {_DIM}(concept flagged composite — Phase H would consult "
            f"decomposition cache before FTS){_RESET}"
        )
    candidates = search_tables(
        metadata_db, concept.text,
        require_coverage_at=data_levels if data_levels else None,
        top_k=3,
    )
    if not candidates:
        # Try unfiltered
        candidates = search_tables(metadata_db, concept.text, top_k=3)
        if candidates:
            print(
                f"        {_DIM}(no candidates with coverage at "
                f"{data_levels}; showing unfiltered top 3){_RESET}"
            )
        else:
            print(f"        {_DIM}(no FTS hits at all){_RESET}")
            return
    for c in candidates:
        title = c.title[:80] + ("…" if len(c.title) > 80 else "")
        print(
            f"        {c.bm25_score:>6.2f}  {c.dataset}/{c.year}/{c.table_id}  "
            f"{title!r}"
        )


def _render_planned_calls(plan: PlanResult) -> None:
    """Render the planner's APIPlanCall list grouped by (geo, concept)."""
    print(f"\n  {_DIM}──── PLAN ────{_RESET}")
    if not plan.calls:
        print(f"    {_DIM}(no API calls planned){_RESET}")
        for n in plan.notes:
            print(f"    {_DIM}· {n}{_RESET}")
        # Surface unresolved tier outcomes so users see why we got
        # nothing.
        for ci, res in enumerate(plan.concept_resolutions):
            if res.tier in (
                "tier_3_llm_fallback", "needs_llm_decomposition",
                "unresolved",
            ):
                print(
                    f"    {_YELLOW}concept[{ci}] {res.concept.text!r} → "
                    f"{res.tier}{_RESET}"
                )
                for n in res.notes:
                    print(f"      {_DIM}· {n}{_RESET}")
        return

    by_pair: dict[tuple[int, int], list[PlannedCall]] = {}
    for c in plan.calls:
        by_pair.setdefault((c.geo_idx, c.concept_idx), []).append(c)

    for (gi, ci), calls in sorted(by_pair.items()):
        geo = plan.resolved_geos[gi]
        concept = plan.intent.concepts[ci]
        resolution = plan.concept_resolutions[ci]
        # Header per pair.
        print(
            f"\n    {_BOLD}{geo.display_name}{_RESET} "
            f"{_DIM}({geo.geo_level}){_RESET}  ×  "
            f"{_BOLD}{concept.text!r}{_RESET}  "
            f"{_DIM}[{resolution.tier}]{_RESET}"
        )
        # Group by (year, role) so trend queries show vintages cleanly.
        by_yr_role: dict[tuple[int, str], list[PlannedCall]] = {}
        for c in calls:
            by_yr_role.setdefault((c.year, c.role), []).append(c)
        for (year, role), grp in sorted(by_yr_role.items()):
            tail = f" role={role}" if role != "primary" else ""
            api = grp[0].api_call
            n_counties = len(grp)
            n_tracts = sum(len(c.tract_filter) for c in grp)
            tract_blurb = (
                f"  tracts={n_tracts} across {n_counties} counties"
                if n_tracts else ""
            )
            print(
                f"      {year}{tail}  →  {_CYAN}{api.dataset}/{api.table_id}"
                f"{_RESET}  vars={api.variables}{tract_blurb}"
            )

    if plan.notes:
        print()
        for n in plan.notes:
            print(f"    {_DIM}· {n}{_RESET}")


def _fmt_number(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) < 1 and x != 0:
        return f"{x:.4f}"
    return f"{x:,.2f}"


def _render_aggregated(response: QueryResponse) -> None:
    print(f"\n  {_DIM}──── DATA ────{_RESET}")
    if response.aggregated is None or not response.aggregated.values:
        print(f"    {_DIM}(no aggregated values){_RESET}")
        if response.aggregated and response.aggregated.fetch_failures:
            for f in response.aggregated.fetch_failures[:5]:
                print(f"    {_RED}fetch failure:{_RESET} {f}")
        return
    for v in response.aggregated.values:
        bits: list[str] = []
        if v.value is not None:
            bits.append(f"value={_BOLD}{_fmt_number(v.value)}{_RESET}")
        if v.ratio is not None:
            bits.append(f"ratio={_BOLD}{v.ratio*100:.2f}%{_RESET}")
        if v.components is not None:
            comps = ", ".join(
                f"{k}={_fmt_number(val)}"
                for k, val in list(v.components.items())[:3]
            )
            bits.append(f"components=[{comps}]")
        if v.sample_size is not None:
            bits.append(f"{_DIM}n={_fmt_number(v.sample_size)}{_RESET}")
        bits.append(f"{_DIM}rows={v.rows_aggregated}{_RESET}")
        role_tail = f" [{v.role}]" if v.role != "primary" else ""
        print(
            f"    ✓ {_BOLD}{v.geo.display_name}{_RESET}  "
            f"× {_CYAN}{v.concept.text!r}{_RESET}  "
            f"@ {v.year}{role_tail}  →  " + "  ".join(bits)
        )
        if v.aggregation_caveat:
            print(f"        {_YELLOW}caveat:{_RESET} {v.aggregation_caveat}")
    if response.aggregated.fetch_failures:
        print(f"\n    {_RED}{len(response.aggregated.fetch_failures)} "
              f"fetch failure(s):{_RESET}")
        for f in response.aggregated.fetch_failures[:5]:
            print(f"      · {f}")


def _render_peer_contexts(peer_contexts) -> None:
    """Print the 'Different geographic units…' block — one section per
    axis, each peer annotated with the human-readable explanation the
    PeerRetriever composed (proximity tier + population scale + top
    matching features).

    Skipped silently when no peers were attached to the query."""
    if not peer_contexts:
        return
    print(f"\n  {_DIM}──── Peer comparisons ────{_RESET}")
    print(
        f"\n    {_BOLD}Different geographic units that might be useful "
        f"comparison points:{_RESET}"
    )
    for ctx in peer_contexts:
        axis_label = getattr(ctx, "axis_description", "") or getattr(
            ctx, "axis", "",
        )
        scope = getattr(ctx, "pool_scope", "") or ""
        scope_tail = f" ({_DIM}{scope}{_RESET})" if scope else ""
        print(
            f"\n    {_CYAN}{axis_label}{_RESET}{scope_tail}"
        )
        peers = getattr(ctx, "peers", []) or []
        if not peers:
            print(f"      {_DIM}(no peers found){_RESET}")
            continue
        for p in peers:
            rank = getattr(p, "rank", 0)
            name = getattr(p, "geo_name", "")
            explanation = getattr(p, "match_explanation", "") or ""
            explan = f" — {explanation}" if explanation else ""
            print(f"      {rank:>2}. {_BOLD}{name}{_RESET}{explan}")


def _render_answer(answer: Optional[SynthesizedAnswer]) -> None:
    print(f"\n  {_DIM}──── ANSWER ────{_RESET}")
    if answer is None:
        print(f"    {_RED}(no synthesized answer){_RESET}")
        return
    # Prose, indented for readability
    for para in answer.prose.split("\n\n"):
        print()
        for line in _wrap_indent(para.strip(), width=78, indent="    "):
            print(line)
    if answer.key_findings:
        print(f"\n    {_BOLD}Key findings:{_RESET}")
        for kf in answer.key_findings:
            print(f"      • {kf}")
    if answer.caveats:
        print(f"\n    {_YELLOW}Caveats:{_RESET}")
        for cv in answer.caveats:
            print(f"      · {cv}")
    if answer.citations:
        print(f"\n    {_DIM}Sources:{_RESET}")
        for c in answer.citations:
            title = f" — {c.table_title}" if c.table_title else ""
            print(
                f"      [{c.dataset}/{c.year}/{c.table_id}]{title}"
            )
            # Show each variable_id with its plain-English label so
            # a reader can audit exactly which columns fed the
            # answer. Missing labels render as the variable_id only.
            for cv in getattr(c, "variables", []) or []:
                lbl = cv.label if cv.label else _DIM + "(no label)" + _RESET
                print(f"        · {cv.variable_id}  {lbl}")


def _wrap_indent(text: str, *, width: int, indent: str) -> list[str]:
    """Cheap word-wrapping for the prose block."""
    if not text:
        return [indent]
    words = text.split()
    lines: list[str] = []
    cur = indent
    for w in words:
        if len(cur) + 1 + len(w) > width and cur != indent:
            lines.append(cur)
            cur = indent + w
        else:
            cur = w if cur == indent else cur + " " + w
    if cur != indent:
        lines.append(cur)
    return lines


def _render_execute_metrics(response: QueryResponse) -> None:
    """When --execute, show full pipeline metrics (LLM + Census + timing)."""
    m = response.metrics
    print(f"\n  {_DIM}──── METRICS ────{_RESET}")
    print(
        f"    {_DIM}total {m.total_s:.2f}s  "
        f"= extract {m.extract_s:.2f} + resolve {m.resolve_s:.2f} + "
        f"plan {m.plan_s:.2f} + decompose {m.decompose_s:.2f} + "
        f"fetch {m.fetch_s:.2f} + aggregate {m.aggregate_s:.2f} + "
        f"synth {m.synthesize_s:.2f}{_RESET}"
    )
    print(
        f"    {_DIM}LLM: {m.llm_calls} calls, "
        f"{m.llm_input_tokens - m.llm_cached_tokens}→"
        f"{m.llm_output_tokens} tokens "
        f"({m.llm_cached_tokens} cached)  "
        f"cost ~${m.llm_cost_usd:.5f}{_RESET}"
    )
    print(
        f"    {_DIM}Census: {m.census_calls_total} calls, "
        f"{m.census_cache_hits} cache hits, "
        f"{m.census_failures} failures{_RESET}"
    )
    if response.error:
        print(f"    {_RED}error:{_RESET} {response.error}")


def _render_concept_lookup(
    intent: ExtractedIntent,
    resolved: list[ResolvedGeography],
    cmap: ConceptMap,
    metadata_db,
) -> None:
    print(f"\n  {_DIM}──── CONCEPT LOOKUP ────{_RESET}")
    if not intent.concepts:
        print(f"    {_DIM}(no concepts to look up){_RESET}")
        return
    data_levels = _data_levels_for(resolved)
    levels_blurb = (
        f" {_DIM}(coverage required at: {data_levels}){_RESET}"
        if data_levels else ""
    )
    print(f"    {levels_blurb.strip() or ''}")
    for k, con in enumerate(intent.concepts):
        print(f"\n    [{k}] {_BOLD}{con.text!r}{_RESET}")
        entry, key_used = _lookup_concept(cmap, con)
        if entry is not None:
            _render_tier1_hit(entry, key_used, data_levels, metadata_db)
        else:
            _render_tier2_fallback(con, data_levels, metadata_db)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _render_metrics(
    latency_s: float,
    usage: Optional[dict],
    session_cost_total: float,
    *,
    no_call_reason: str = "mock",
) -> None:
    print(f"\n  {_DIM}──── METRICS ────{_RESET}")
    if usage is None:
        if no_call_reason == "no_call":
            print(f"    {_DIM}latency {latency_s:.2f}s  "
                  f"(no LLM call — short-circuited){_RESET}")
        else:
            print(f"    {_DIM}latency {latency_s:.2f}s  "
                  f"(mock LLM — no token / cost data){_RESET}")
        return
    cost = estimate_cost_usd(usage)
    cached = usage.get("cached_tokens", 0)
    fresh_in = usage.get("input_tokens", 0) - cached
    cache_blurb = f" ({cached} cached)" if cached else ""
    calls = usage.get("calls", 0)
    call_blurb = f"  LLM calls: {calls}" if calls != 1 else ""
    print(
        f"    {_DIM}latency {latency_s:.2f}s  "
        f"tokens {fresh_in}→{usage.get('output_tokens', 0)}{cache_blurb}  "
        f"cost ~${cost:.5f}  session total ~${session_cost_total:.5f}"
        f"{call_blurb}{_RESET}"
    )


# ---------------------------------------------------------------------------
# Main per-turn driver
# ---------------------------------------------------------------------------

def _interactive_ask_user(rendered: str) -> str:
    """Default ask_user callback for the REPL: print the question,
    then read one line from stdin. The user may type a letter (a/b/c)
    or a freeform response."""
    print()
    print(f"{_YELLOW}{rendered}{_RESET}")
    print()
    try:
        return input(
            f"{_RL_START}{_BOLD}{_RL_END}  clarify> {_RL_START}{_RESET}{_RL_END}",
        ).strip()
    except EOFError:
        return ""


_progress_last_tick: dict = {"t": None}


def _progress_printer(stage: str, detail: Optional[str]) -> None:
    """Stage-progress callback for the REPL. Prints a dim/cyan line
    under the user's prompt so they can see what the pipeline is
    doing while Vertex/Census calls are in flight. Each line also
    shows the elapsed time since the PREVIOUS tick — so you can
    instantly see whether a stage is genuinely running (numbers
    ticking up) or truly hung (one line with a growing timestamp)."""
    now = time.monotonic()
    prev = _progress_last_tick["t"]
    delta = "" if prev is None else f" {_YELLOW}(+{now - prev:.1f}s){_RESET}"
    _progress_last_tick["t"] = now
    t = time.strftime("%H:%M:%S")
    tail = f" {_DIM}({detail}){_RESET}" if detail else ""
    print(
        f"  {_DIM}[{t}]{_RESET} {_CYAN}› {stage}{_RESET}{tail}{delta}",
        flush=True,
    )


def _run_execute(
    text: str,
    llm: LLMClient,
    db, metadata_db,
    cmap: ConceptMap,
    decomp_cache: DecompositionCache,
    api_cache: APICache,
    api_key: Optional[str],
    config: dict,
    *,
    show_raw: bool = False,
    semantic_router: Optional[object] = None,
    peer_retriever: Optional[object] = None,
    universe_picker: Optional[object] = None,
    ask_user: Optional[object] = None,
) -> tuple[Optional[dict], float, QueryResponse]:
    """Full end-to-end pipeline via the orchestrator."""
    if hasattr(llm, "reset_usage_counters"):
        llm.reset_usage_counters()
    _progress_last_tick["t"] = None       # reset per-query inter-stage deltas
    response = answer_query_sync(
        text, llm, db, metadata_db, cmap,
        decomp_cache=decomp_cache, api_cache=api_cache,
        api_key=api_key, config=config,
        semantic_router=semantic_router,
        peer_retriever=peer_retriever,
        universe_picker=universe_picker,
        ask_user=ask_user,
        progress_cb=_progress_printer,
    )
    if show_raw:
        print(f"  {_DIM}raw intent JSON:{_RESET}")
        print(f"  {json.dumps(response.intent.model_dump(), indent=2)}")
    _render_extraction(response.intent)
    _render_geo_resolution(response.resolved_geos)
    _render_concept_lookup(
        response.intent, response.resolved_geos, cmap, metadata_db,
    )
    _render_planned_calls(response.plan)
    _render_aggregated(response)
    _render_answer(response.answer)
    _render_peer_contexts(getattr(response, "peer_contexts", None))
    _render_execute_metrics(response)
    # Return shape matches run_one's, so the main loop's metrics path
    # can keep working — but execute prints its own metrics block.
    usage_dict = {
        "calls": response.metrics.llm_calls,
        "input_tokens": response.metrics.llm_input_tokens,
        "cached_tokens": response.metrics.llm_cached_tokens,
        "output_tokens": response.metrics.llm_output_tokens,
    } if response.metrics.llm_calls else None
    return usage_dict, response.metrics.total_s, response


def run_one(
    text: str,
    llm: LLMClient,
    db,
    metadata_db,
    cmap: ConceptMap,
    config: dict,
    *,
    show_raw: bool = False,
    show_plan: bool = False,
    semantic_router: Optional[object] = None,
) -> tuple[Optional[dict], float]:
    if hasattr(llm, "reset_usage_counters"):
        llm.reset_usage_counters()

    t0 = time.time()
    try:
        intent = extract_intent(
            text, llm,
            temperature=config.get("vertex_ai", {}).get("temperature", 0.1),
        )
    except ExtractorError as e:
        print(f"  {_RED}extractor error: {e}{_RESET}")
        return None, time.time() - t0
    except KeyError:
        print(
            f"  {_YELLOW}mock LLM has no scripted response for this query"
            f"{_RESET}"
        )
        return None, time.time() - t0

    if show_raw:
        print(f"  {_DIM}raw intent JSON:{_RESET}")
        print(f"  {json.dumps(intent.model_dump(), indent=2)}")

    _render_extraction(intent)
    resolved = resolve_intent(intent, db)
    _render_geo_resolution(resolved)
    _render_concept_lookup(intent, resolved, cmap, metadata_db)
    if show_plan:
        plan = plan_query(
            intent, resolved, cmap, metadata_db,
            semantic_router=semantic_router,
        )
        _render_planned_calls(plan)
    latency = time.time() - t0

    usage = None
    if hasattr(llm, "get_usage_counters"):
        raw_usage = llm.get_usage_counters()
        if raw_usage.get("calls", 0) > 0:
            usage = raw_usage
    return usage, latency


# ---------------------------------------------------------------------------
# REPL loop
# ---------------------------------------------------------------------------

def _handle_command(cmd: str) -> Optional[bool]:
    if cmd in ("/quit", "/exit"):
        return False
    if cmd == "/help":
        print(
            f"  {_DIM}/quit  /help  /raw <query>  /plan <query>  "
            f"/execute <query>  (any other text → preview only){_RESET}"
        )
        return True
    if cmd.startswith("/"):
        print(f"  unknown command: {cmd}")
        return True
    return None


def run(args: argparse.Namespace) -> int:
    config = _load_config()
    if args.mock_llm:
        print(f"{_DIM}LLM: MockLLMClient (gold queries only){_RESET}")
        llm = _build_mock_llm()
        no_call_reason = "mock"
    else:
        llm = _build_live_llm(config)
        no_call_reason = "no_call"

    db = open_spatialite(REPO_ROOT / config["paths"]["gazetteer_db"])
    metadata_db = open_search_index(
        REPO_ROOT / config["paths"]["metadata_db"]
    )
    cmap = ConceptMap.from_yaml()
    decomp_cache = DecompositionCache(
        REPO_ROOT / config["paths"]["decomposition_cache_db"]
    )
    api_cache = APICache(REPO_ROOT / config["paths"]["api_cache_db"])
    api_key = os.environ.get(
        config.get("census_api", {}).get("api_key_env_var", "CENSUS_API_KEY")
    )
    print(
        f"{_DIM}Loaded {len(cmap)} curated concepts ({len(cmap.all_keys())} "
        f"lookup keys){_RESET}"
    )

    # K.5 — load semantic router unless --no-router was passed.
    semantic_router = None
    if not args.no_router and not args.mock_llm:
        try:
            from scripts.chatbot.auth_check import check_auth
            from scripts.chatbot.semantic_router import (
                SemanticRouter, VertexEmbedder,
            )
            auth = check_auth(verbose=False)
            embedder = VertexEmbedder(
                project=auth["project_id"],
                location=auth.get("location", "us-central1"),
            )
            index_path = REPO_ROOT / config["paths"].get(
                "embedding_index", "data/metadata/embeddings.faiss",
            )
            metadata_path = REPO_ROOT / config["paths"]["metadata_db"]
            if index_path.exists():
                semantic_router = SemanticRouter(
                    index_path, metadata_path, embedder,
                )
                print(
                    f"{_DIM}Semantic router: "
                    f"{semantic_router.index.ntotal:,} vectors "
                    f"(dim={semantic_router.index.d}){_RESET}"
                )
            else:
                print(
                    f"{_YELLOW}embeddings.faiss missing — falling back to "
                    f"legacy tier cascade. Run: "
                    f"python -m tools.build_embedding_index{_RESET}"
                )
        except Exception as e:                  # pragma: no cover
            print(f"{_YELLOW}Could not load semantic router: {e}; "
                  f"falling back to legacy tier cascade.{_RESET}")
            semantic_router = None

    # Universe-aware variable picker (task #35). Replaces the legacy
    # weight-stacked composer with a two-stage semantic flow. Loads
    # lazily from data/metadata/universe_embeddings.npz; when absent
    # the planner falls back to the legacy composer automatically.
    universe_picker = None
    if semantic_router is not None:
        try:
            from scripts.chatbot.nodes.universe_picker import (
                UniverseEmbeddings, UniversePicker,
            )
            ue_path = (
                REPO_ROOT / "data" / "metadata" / "universe_embeddings.npz"
            )
            if ue_path.exists():
                universe_picker = UniversePicker(
                    universe_emb=UniverseEmbeddings.load(ue_path),
                    faiss_index=semantic_router.index,
                    metadata_db=semantic_router.metadata_db,
                    embedder=semantic_router.embedder,
                )
                print(
                    f"{_DIM}Universe picker: enabled "
                    f"({ue_path.name}){_RESET}"
                )
            else:
                print(
                    f"{_DIM}Universe picker: universe_embeddings.npz "
                    f"missing — using legacy composer.{_RESET}"
                )
        except Exception as e:                    # pragma: no cover
            print(
                f"{_YELLOW}Could not load universe picker: {e}; "
                f"using legacy composer.{_RESET}"
            )
            universe_picker = None

    # Phase 3 — load PeerRetriever when peer_features.sqlite exists.
    # Silent no-op when missing; the orchestrator treats
    # peer_retriever=None as "skip peer context enrichment".
    peer_retriever = None
    if not args.mock_llm:
        try:
            from scripts.chatbot.nodes.peer_retriever import PeerRetriever
            peer_db_path = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
            if peer_db_path.exists():
                peer_retriever = PeerRetriever(peer_db_path)
                print(
                    f"{_DIM}Peer retriever: loaded "
                    f"{peer_db_path.name}{_RESET}"
                )
            else:
                print(
                    f"{_DIM}Peer retriever: peer_features.sqlite not found; "
                    f"skipping peer context.{_RESET}"
                )
        except Exception as e:                  # pragma: no cover
            print(f"{_YELLOW}Could not load peer retriever: {e}{_RESET}")
            peer_retriever = None

    if args.execute and not api_key:
        print(f"{_YELLOW}warning: CENSUS_API_KEY not set; rate-limited "
              f"to 500/day{_RESET}")

    session_id = uuid.uuid4().hex[:8]
    print(
        f"{_BOLD}pipeline REPL{_RESET}  session={_CYAN}{session_id}{_RESET}  "
        f"{_DIM}(/quit, /help){_RESET}"
    )

    session_cost_total = 0.0
    try:
        if args.query:
            if args.execute:
                # One-shot --query mode: no ask_user wired → clarifier
                # auto-picks (a). Users who want interactive clarify
                # should drop into the REPL.
                _, _, _ = _run_execute(
                    args.query, llm, db, metadata_db, cmap,
                    decomp_cache, api_cache, api_key, config,
                    show_raw=args.raw, semantic_router=semantic_router,
                    peer_retriever=peer_retriever,
                    universe_picker=universe_picker,
                    ask_user=None,
                )
                return 0
            usage, latency = run_one(
                args.query, llm, db, metadata_db, cmap, config,
                show_raw=args.raw, show_plan=args.plan,
                semantic_router=semantic_router,
            )
            if usage is not None:
                session_cost_total += estimate_cost_usd(usage)
            _render_metrics(latency, usage, session_cost_total,
                            no_call_reason=no_call_reason)
            return 0

        while True:
            try:
                text = input(
                    f"\n{_RL_START}{_BOLD}{_RL_END}> {_RL_START}{_RESET}{_RL_END}",
                ).strip()
            except EOFError:
                print()
                break
            if not text:
                continue
            cmd_result = _handle_command(text)
            if cmd_result is False:
                break
            if cmd_result is True:
                continue

            show_raw = False
            show_plan = args.plan
            run_execute = args.execute
            if text.startswith("/raw "):
                text = text[len("/raw "):].strip()
                show_raw = True
            if text.startswith("/plan "):
                text = text[len("/plan "):].strip()
                show_plan = True
            if text.startswith("/execute "):
                text = text[len("/execute "):].strip()
                run_execute = True

            if run_execute:
                _, _, response = _run_execute(
                    text, llm, db, metadata_db, cmap,
                    decomp_cache, api_cache, api_key, config,
                    show_raw=show_raw, semantic_router=semantic_router,
                    peer_retriever=peer_retriever,
                    universe_picker=universe_picker,
                    ask_user=_interactive_ask_user,
                )
                if response.metrics.llm_cost_usd:
                    session_cost_total += response.metrics.llm_cost_usd
                continue

            usage, latency = run_one(
                text, llm, db, metadata_db, cmap, config,
                show_raw=show_raw, show_plan=show_plan,
                semantic_router=semantic_router,
            )
            if usage is not None:
                session_cost_total += estimate_cost_usd(usage)
            _render_metrics(latency, usage, session_cost_total,
                            no_call_reason=no_call_reason)
    finally:
        db.close()
        metadata_db.close()
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mock-llm", action="store_true",
                   help="Use MockLLMClient seeded from gold_queries.yaml")
    p.add_argument("--query", default=None,
                   help="One-shot mode: run a single query, then exit")
    p.add_argument("--raw", action="store_true",
                   help="Also print raw extracted-intent JSON")
    p.add_argument("--plan", action="store_true",
                   help="Also render the planner's API-call plan "
                        "(no calls made)")
    p.add_argument("--execute", action="store_true",
                   help="Run the full pipeline end-to-end (extract → "
                        "resolve → plan → decompose → fetch → aggregate "
                        "→ synthesize). Issues real Census API calls.")
    p.add_argument("--no-router", action="store_true",
                   help="Skip the semantic router; fall back to legacy "
                        "Tier 1/2/3 cascade (used for A/B comparison).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
