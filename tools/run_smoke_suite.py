"""Sequential smoke-test runner for the 50 gold-standard queries.

Runs every query through the orchestrator, persists per-query JSON to
``reports/smoke_suite_progress.jsonl`` after each one (so crashes /
rate-limit backoffs can resume), and writes a final markdown report
to ``reports/smoke_suite_report.md`` at the end.

Rate-limit handling: if the query raises an exception whose message
looks like a Vertex throttle (``429``, ``quota``, ``resource_
exhausted``, etc.), the runner sleeps 5 minutes and retries up to
3 times before giving up on that query and moving on.

Usage:
    python -m tools.run_smoke_suite
    python -m tools.run_smoke_suite --fresh   # ignore prior progress
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


QUERIES: list[tuple[int, str, str]] = [
    # (id, level, query)
    (1, "L1 simple lookup", "What's the median household income in DeKalb County, Georgia?"),
    (2, "L1 simple lookup", "What's the poverty rate in Fulton County?"),
    (3, "L1 simple lookup", "What's the unemployment rate in Clayton County?"),
    (4, "L1 simple lookup", "How many people live in the city of Atlanta?"),
    (5, "L1 simple lookup", "What percentage of households in Gwinnett County are renters?"),
    (6, "L2 variable domain", "What's the median gross rent in Cobb County?"),
    (7, "L2 variable domain", "What's the homeownership rate in Fulton County?"),
    (8, "L2 variable domain", "What's the uninsured rate in DeKalb County?"),
    (9, "L2 variable domain", "How many veterans live in Fulton County?"),
    (10, "L2 variable domain", "What's the educational attainment in Clayton County?"),
    (11, "L3 neighborhood", "What's the median income in Buckhead?"),
    (12, "L3 neighborhood", "What's the poverty rate in Midtown Atlanta?"),
    (13, "L3 neighborhood", "How many households in East Atlanta have no vehicle?"),
    (14, "L3 neighborhood", "What's the rent burden rate in Old Fourth Ward?"),
    (15, "L3 neighborhood", "What's the child poverty rate in the Westside of Atlanta?"),
    (16, "L4 grant frame", "I'm writing a food pantry grant for Southwest Atlanta."),
    (17, "L4 grant frame", "Help me build a rent burden case for Fulton County."),
    (18, "L4 grant frame", "I need an ESL services grant narrative for Clarkston, Georgia."),
    (19, "L4 grant frame", "Workforce development grant for DeKalb County — what data should I cite?"),
    (20, "L4 grant frame", "Building a case for senior isolation programs in Cobb County."),
    (21, "L5 compare/trend", "Compare poverty rates between DeKalb and Fulton counties."),
    (22, "L5 compare/trend", "How has median household income changed in Atlanta since 2019?"),
    (23, "L5 compare/trend", "How has SNAP participation changed in Clayton County over the last 5 years?"),
    (24, "L5 compare/trend", "Compare rent burden between Atlanta city and the Atlanta MSA."),
    (25, "L5 compare/trend", "How has child poverty changed in Fulton County since 2019?"),
    (26, "L6 disparity", "What's the racial income gap in Fulton County?"),
    (27, "L6 disparity", "Compare poverty rates by race in DeKalb County."),
    (28, "L6 disparity", "Is there a gender earnings gap in Atlanta?"),
    (29, "L6 disparity", "Compare child poverty rates for Hispanic vs white children in Gwinnett."),
    (30, "L6 disparity", "What's the educational attainment gap by race in Atlanta?"),
    (31, "L7 complex narrative", "Needs assessment for a youth workforce grant in Clayton County."),
    (32, "L7 complex narrative", "First-time homebuyer gap for Black households in Atlanta."),
    (33, "L7 complex narrative", "Climate resilience case for elderly residents in South Fulton."),
    (34, "L7 complex narrative", "Kinship caregiving rates in Clayton County — grandparents raising grandchildren."),
    (35, "L7 complex narrative", "Digital divide case for rural Georgia counties."),
    (36, "L8 multi-concept", "Food access indicators across Southwest Atlanta, Clayton County, and the metro."),
    (37, "L8 multi-concept", "Housing instability comparison between Buckhead and East Atlanta."),
    (38, "L8 multi-concept", "Theory of change for a community health program in Fulton County."),
    (39, "L8 multi-concept", "Broadband expansion grant for a rural high-poverty Georgia area."),
    (40, "L8 multi-concept", "Environmental justice indicators for the Westside of Atlanta."),
    (41, "L9 nuanced edge", "Substance use prevention program — demographic risk indicators for DeKalb."),
    (42, "L9 nuanced edge", "Homelessness risk indicators for Atlanta."),
    (43, "L9 nuanced edge", "Language access case for the city of Atlanta."),
    (44, "L9 nuanced edge", "Disability services baseline for Atlanta."),
    (45, "L9 nuanced edge", "Mental health services case for DeKalb County."),
    (46, "L10 scope gate", "What's the weather forecast for Atlanta this weekend?"),
    (47, "L10 scope gate", "How many Waffle Houses are in Fulton County?"),
    (48, "L10 scope gate", "How many jet skis are in Northeast Atlanta?"),
    (49, "L10 exclusion", "I'm writing a grant for more school buses in Atlanta excluding the Buckhead area."),
    (50, "L10 relative geo", "What's the poverty rate near the Atlanta Beltline?"),
]


# Tokens in error messages that indicate a Vertex rate limit / quota.
_RATE_LIMIT_TOKENS = (
    "429", "quota", "resource_exhausted", "resource exhausted",
    "rate limit", "rate-limit", "too many requests",
    "throttl", "overloaded",
)


def _looks_rate_limited(msg: str) -> bool:
    low = msg.lower()
    return any(tok in low for tok in _RATE_LIMIT_TOKENS)


def _bootstrap_clients(config: dict):
    """Build every client/fixture the orchestrator needs. Matches
    tools/trace_orchestrator.py's setup."""
    from scripts.chatbot.auth_check import check_auth
    from scripts.chatbot.llm_client import GeminiVertexClient
    from scripts.chatbot.concept_map import ConceptMap
    from scripts.chatbot.api_cache import APICache
    from scripts.chatbot.decomposition_cache import DecompositionCache
    from scripts.chatbot.semantic_router import (
        SemanticRouter, VertexEmbedder, RouterConfig,
    )
    from scripts.chatbot.frames import load_default_frames
    from scripts.chatbot.gazetteer_db import open_spatialite
    from scripts.chatbot.nodes.peer_retriever import PeerRetriever
    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )

    auth = check_auth(verbose=False)
    project = auth["project_id"]
    location = auth.get("location", "us-central1")

    gaz_db = open_spatialite(REPO_ROOT / config["paths"]["gazetteer_db"])
    gaz_db.row_factory = sqlite3.Row
    meta_db = sqlite3.connect(REPO_ROOT / config["paths"]["metadata_db"])
    meta_db.row_factory = sqlite3.Row
    cmap = ConceptMap.from_yaml(REPO_ROOT / config["paths"]["concept_map"])
    api_cache = APICache(REPO_ROOT / config["paths"]["api_cache_db"])
    decomp_cache = DecompositionCache(
        REPO_ROOT / config["paths"]["decomposition_cache_db"]
    )

    vx = config.get("vertex_ai", {})
    llm = GeminiVertexClient(
        project_id=project, location=location,
        model=vx.get("model", "gemini-2.5-flash"),
        temperature=vx.get("temperature", 0.1),
        timeout_seconds=vx.get("timeout_seconds", 60.0),
        enable_cache=vx.get("enable_cache", False),
        cache_ttl_seconds=vx.get("cache_ttl_seconds", 3600),
    )

    embedder = VertexEmbedder(project=project, location=location)
    router = SemanticRouter(
        REPO_ROOT / "data" / "metadata" / "embeddings.faiss",
        REPO_ROOT / config["paths"]["metadata_db"],
        embedder, config=RouterConfig(),
    )

    frame_registry = load_default_frames()

    peer_db = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
    peer_retriever = PeerRetriever(peer_db) if peer_db.exists() else None

    ue_path = REPO_ROOT / "data" / "metadata" / "universe_embeddings.npz"
    universe_picker = None
    if ue_path.exists():
        universe_picker = UniversePicker(
            universe_emb=UniverseEmbeddings.load(ue_path),
            faiss_index=router.index,
            metadata_db=router.metadata_db,
            embedder=router.embedder,
        )

    return {
        "llm": llm, "gaz_db": gaz_db, "meta_db": meta_db, "cmap": cmap,
        "api_cache": api_cache, "decomp_cache": decomp_cache,
        "semantic_router": router, "frame_registry": frame_registry,
        "peer_retriever": peer_retriever, "universe_picker": universe_picker,
    }


def _serialize_response(query_id, level, query, resp, duration_s, dump_folder):
    answer = getattr(resp, "answer", None)
    plan = getattr(resp, "plan", None)
    agg = getattr(resp, "aggregated", None)
    peer_contexts = getattr(resp, "peer_contexts", []) or []
    metrics = getattr(resp, "metrics", None)
    return {
        "id": query_id,
        "level": level,
        "query": query,
        "duration_s": round(duration_s, 1),
        "error": getattr(resp, "error", None),
        "plan_calls": len(plan.calls) if plan else 0,
        "aggregated_values": len(agg.values) if agg else 0,
        "fetch_failures": (
            len(agg.fetch_failures) if agg else 0
        ),
        "answer_prose": (answer.prose if answer else None),
        "key_findings": (list(answer.key_findings) if answer else []),
        "caveats": (list(answer.caveats) if answer else []),
        "citations": (
            [
                {
                    "dataset": c.dataset, "year": c.year,
                    "table_id": c.table_id,
                    "table_title": c.table_title,
                    "variables": [
                        {"variable_id": cv.variable_id, "label": cv.label}
                        for cv in (c.variables or [])
                    ],
                }
                for c in (answer.citations if answer else [])
            ]
        ),
        "peer_contexts": [
            {
                "axis": pc.axis,
                "pool_scope": pc.pool_scope,
                "peers": [
                    {
                        "geo_name": p.geo_name,
                        "explanation": p.match_explanation,
                    }
                    for p in (pc.peers or [])[:5]
                ],
            }
            for pc in peer_contexts
        ],
        "metrics": {
            "total_s": getattr(metrics, "total_s", None),
            "llm_calls": getattr(metrics, "llm_calls", None),
            "census_calls": getattr(metrics, "census_calls_total", None),
        } if metrics else {},
        "dump_folder": str(dump_folder) if dump_folder else None,
    }


def _dump_folder_for(t_start_iso):
    """Best-effort locate the derived_data folder written by the
    orchestrator for this run."""
    base = REPO_ROOT / "derived_data"
    if not base.exists():
        return None
    # Return whichever folder was most recently modified since our
    # start time. Sorted by name is timestamp-ordered.
    newest = None
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("data_"):
            if newest is None or p.stat().st_mtime > newest.stat().st_mtime:
                newest = p
    return newest


def run_suite(fresh: bool = False):
    progress_path = REPO_ROOT / "reports" / "smoke_suite_progress.jsonl"
    progress_path.parent.mkdir(exist_ok=True)
    if fresh and progress_path.exists():
        progress_path.unlink()

    # Load prior progress so we can resume.
    done_ids: set[int] = set()
    if progress_path.exists():
        with progress_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["id"])
                except json.JSONDecodeError:
                    continue
    if done_ids:
        print(f"Resuming: {len(done_ids)} queries already done "
              f"({sorted(done_ids)})")

    # Bootstrap clients ONCE so we don't reload FAISS 50 times.
    with (REPO_ROOT / "config" / "chatbot.yaml").open() as f:
        config = yaml.safe_load(f)
    # PoC-run override: disable the clarifier / plan-reviewer
    # clarification loop. The reviewer still runs internally but its
    # clarifications won't fire; the orchestrator auto-skips in
    # non-interactive mode (ask_user=None) anyway, but this makes
    # the intent explicit in the run record.
    config.setdefault("clarification", {})["enabled"] = False
    clients = _bootstrap_clients(config)

    from scripts.chatbot.orchestrator import answer_query_sync

    api_key = os.environ.get("CENSUS_API_KEY")

    # Inter-query pause. The Census API throttles bursts of
    # neighborhood queries (each composite neighborhood expands into
    # a separate fetch per county + variable combo), so we space the
    # queries out at 10s between runs — enough to drain any per-
    # second token bucket while keeping total runtime reasonable.
    INTER_QUERY_PAUSE_S = 10.0

    start_wall = time.time()
    attempts_by_id: dict[int, int] = {}
    for q_ix, (qid, level, query) in enumerate(QUERIES):
        if qid in done_ids:
            print(f"[{qid:02d}/50] SKIP (done)")
            continue

        if q_ix > 0:
            print(f"  [pause {INTER_QUERY_PAUSE_S:.0f}s before next query]",
                  flush=True)
            time.sleep(INTER_QUERY_PAUSE_S)

        print(f"[{qid:02d}/50] {level}: {query[:70]}...", flush=True)
        max_attempts = 3
        attempt = 0
        record = None
        while attempt < max_attempts:
            attempt += 1
            attempts_by_id[qid] = attempt
            t0 = time.time()
            try:
                if hasattr(clients["llm"], "reset_usage_counters"):
                    clients["llm"].reset_usage_counters()
                resp = answer_query_sync(
                    query, clients["llm"], clients["gaz_db"],
                    clients["meta_db"], clients["cmap"],
                    decomp_cache=clients["decomp_cache"],
                    api_cache=clients["api_cache"],
                    api_key=api_key, config=config,
                    semantic_router=clients["semantic_router"],
                    frame_registry=clients["frame_registry"],
                    peer_retriever=clients["peer_retriever"],
                    universe_picker=clients["universe_picker"],
                    ask_user=None,
                )
                duration = time.time() - t0
                dump_folder = _dump_folder_for(t0)
                record = _serialize_response(
                    qid, level, query, resp, duration, dump_folder,
                )
                record["attempts"] = attempt
                record["rate_limited"] = False
                break
            except Exception as e:
                duration = time.time() - t0
                msg = f"{type(e).__name__}: {e}"
                if _looks_rate_limited(msg) and attempt < max_attempts:
                    print(
                        f"    RATE LIMIT detected (attempt {attempt}/{max_attempts}) — "
                        f"sleeping 300s before retry: {msg[:120]}"
                    )
                    time.sleep(300)
                    continue
                # Non-rate-limit, or attempts exhausted.
                tb = traceback.format_exc()
                record = {
                    "id": qid, "level": level, "query": query,
                    "duration_s": round(duration, 1),
                    "error": msg,
                    "traceback": tb[-2000:],
                    "rate_limited": _looks_rate_limited(msg),
                    "attempts": attempt,
                    "plan_calls": 0, "aggregated_values": 0,
                    "answer_prose": None, "key_findings": [],
                    "caveats": [], "citations": [], "peer_contexts": [],
                    "metrics": {},
                    "dump_folder": None,
                }
                break

        with progress_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        status = (
            "OK" if record.get("error") is None and record.get("answer_prose")
            else ("ERROR" if record.get("error") else "NO_ANSWER")
        )
        dur = record.get("duration_s", 0)
        n_calls = record.get("plan_calls", 0)
        print(
            f"    {status}  {dur:.1f}s  {n_calls} plan calls  "
            f"attempt={record.get('attempts')}"
        )

    total = time.time() - start_wall
    print(f"\nSuite complete in {total/60:.1f} minutes")
    return progress_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore prior progress and re-run every query")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    run_suite(fresh=args.fresh)


if __name__ == "__main__":
    sys.exit(main())
