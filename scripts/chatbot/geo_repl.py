"""Geographic-resolution REPL.

Phase C scope: tests just the geographic side of the pipeline.
  user query → LLM extraction → resolver/composer → display

No concept tagging, no API calls, no synthesis — those land in later
phases. This CLI is intended for poking at hard geographic queries
("south side of Buckhead", "near I-85 in downtown Atlanta", etc.) and
seeing exactly what the extractor + resolver produce.

Usage:
    python -m scripts.chatbot.geo_repl                 # live Vertex
    python -m scripts.chatbot.geo_repl --mock-llm      # offline; gold
                                                       # queries only
    python -m scripts.chatbot.geo_repl --query "..."   # one-shot, exits

Per-turn output:
    > south side of Buckhead
      ──── EXTRACTION ────
        intent_type: exact
        geo_refs[0]: text="Buckhead" ref_type=neighborhood
                     scope_hint="Atlanta"
        compound: directional, primary=0, filter=0, direction=s
      ──── RESOLUTION ────
        ✓ s side of Buckhead (neighborhood)
              tracts=4   conf=0.79   data=tract
              · derived 33 tracts from Buckhead before directional filter
              · filtered to tracts on the s side of Buckhead
      ──── METRICS ────
        latency 2.3s   tokens 412→58 (3851 cached)   cost ~$0.00021
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

import yaml

from scripts.chatbot.extractor import (
    ExtractorError, extract_intent,
)
from scripts.chatbot.gazetteer_db import open_spatialite
from scripts.chatbot.geo_resolver import resolve_intent
from scripts.chatbot.llm_client import (
    LLMClient, MockLLMClient, build_default_client, estimate_cost_usd,
)
from scripts.chatbot.models import ExtractedIntent, ResolvedGeography


REPO_ROOT = Path(__file__).resolve().parents[2]

# ANSI colours
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_MAGENTA = "\033[35m"
_RED    = "\033[31m"
_RESET  = "\033[0m"


def _load_config() -> dict:
    with (REPO_ROOT / "config" / "chatbot.yaml").open() as f:
        return yaml.safe_load(f)


def _build_mock_llm() -> MockLLMClient:
    """Mock LLM seeded from the v1 gold-queries fixture."""
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
    from scripts.chatbot.auth_check import check_auth, AuthCheckError
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
# Rendering
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
        bits = [f"text={_BOLD}{ref.text!r}{_RESET}",
                f"ref_type={_CYAN}{ref.ref_type}{_RESET}"]
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
    for k, con in enumerate(intent.concepts):
        bits = [f"text={_BOLD}{con.text!r}{_RESET}"]
        if con.canonical_hint:
            bits.append(f"hint={_CYAN}{con.canonical_hint!r}{_RESET}")
        if con.is_composite:
            bits.append(f"{_YELLOW}composite{_RESET}")
        print(f"    concepts[{k}]: " + "  ".join(bits))
    if intent.temporal_intent != "latest" or intent.years:
        years_blurb = (
            f" years={intent.years}" if intent.years else ""
        )
        print(
            f"    {_MAGENTA}temporal{_RESET}: "
            f"{intent.temporal_intent}{years_blurb}"
        )


def _render_resolution(resolved: list[ResolvedGeography]) -> None:
    print(f"  {_DIM}──── RESOLUTION ────{_RESET}")
    if not resolved:
        print(f"    {_RED}(no resolution){_RESET}")
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
        data_mismatch = (
            r.data_level_available != "none"
            and r.data_level_available != r.geo_level
        )
        data_color = _YELLOW if data_mismatch else _DIM
        print(
            f"    ✓ {_BOLD}{r.display_name}{_RESET} "
            f"{_DIM}({r.geo_level}){_RESET}  "
            f"{tract_blurb}   "
            f"{conf_color}conf={r.confidence:.2f}{_RESET}   "
            f"{data_color}data={r.data_level_available}{_RESET}"
        )
        if r.county_geoid:
            print(f"        {_DIM}county_geoid: {r.county_geoid}{_RESET}")
        if r.tract_geoids:
            preview = ", ".join(r.tract_geoids[:3])
            more = f" (+{len(r.tract_geoids) - 3} more)" if len(r.tract_geoids) > 3 else ""
            print(f"        {_DIM}sample tracts: {preview}{more}{_RESET}")
        for note in r.assumption_notes:
            print(f"        {_DIM}· {note}{_RESET}")


def _render_metrics(
    latency_s: float,
    usage: Optional[dict],
    session_cost_total: float,
    *,
    no_call_reason: str = "mock",
) -> None:
    print(f"  {_DIM}──── METRICS ────{_RESET}")
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
# Commands
# ---------------------------------------------------------------------------

def _handle_command(cmd: str) -> Optional[bool]:
    if cmd in ("/quit", "/exit"):
        return False
    if cmd == "/help":
        print(f"  {_DIM}/quit  /help  /raw <query>  (any other text → resolve){_RESET}")
        return True
    if cmd.startswith("/"):
        print(f"  unknown command: {cmd}")
        return True
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_one(
    text: str,
    llm: LLMClient,
    db,
    config: dict,
    *,
    show_raw: bool = False,
) -> tuple[Optional[dict], float]:
    """Resolve one query and render. Returns (usage_dict, latency_s)."""
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
        print(f"  {_YELLOW}mock LLM has no scripted response for "
              f"this query{_RESET}")
        return None, time.time() - t0

    if show_raw:
        print(f"  {_DIM}raw intent JSON:{_RESET}")
        print(f"  {json.dumps(intent.model_dump(), indent=2)}")

    _render_extraction(intent)

    resolved = resolve_intent(intent, db)
    _render_resolution(resolved)
    latency = time.time() - t0

    usage = None
    if hasattr(llm, "get_usage_counters"):
        raw_usage = llm.get_usage_counters()
        if raw_usage.get("calls", 0) > 0:
            usage = raw_usage
    return usage, latency


def run(args: argparse.Namespace) -> int:
    config = _load_config()
    if args.mock_llm:
        print(f"{_DIM}LLM: MockLLMClient (gold queries only){_RESET}")
        llm = _build_mock_llm()
        no_call_reason = "mock"
    else:
        llm = _build_live_llm(config)
        no_call_reason = "no_call"   # only for short-circuits in this mode

    db = open_spatialite(REPO_ROOT / config["paths"]["gazetteer_db"])
    session_id = uuid.uuid4().hex[:8]
    print(f"{_BOLD}geo resolver REPL{_RESET}  "
          f"session={_CYAN}{session_id}{_RESET}  "
          f"{_DIM}(/quit, /help){_RESET}")

    session_cost_total = 0.0
    try:
        if args.query:
            # one-shot
            usage, latency = run_one(args.query, llm, db, config,
                                       show_raw=args.raw)
            if usage is not None:
                session_cost_total += estimate_cost_usd(usage)
            _render_metrics(latency, usage, session_cost_total,
                             no_call_reason=no_call_reason)
            return 0

        while True:
            try:
                text = input(f"{_BOLD}> {_RESET}").strip()
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
            if text.startswith("/raw "):
                text = text[len("/raw "):].strip()
                show_raw = True

            usage, latency = run_one(text, llm, db, config, show_raw=show_raw)
            if usage is not None:
                session_cost_total += estimate_cost_usd(usage)
            _render_metrics(latency, usage, session_cost_total,
                             no_call_reason=no_call_reason)
    finally:
        db.close()
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mock-llm", action="store_true",
                   help="Use MockLLMClient seeded from gold_queries.yaml")
    p.add_argument("--query", default=None,
                   help="One-shot mode: resolve a single query, then exit")
    p.add_argument("--raw", action="store_true",
                   help="Also print raw extracted-intent JSON")
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
