"""Orchestrator-only trace: run answer_query_sync with an aggressive
watchdog so we can see exactly which orchestrator extra (comparator
expansion, trend expansion, concept expansion, agent routing) is the
one that hangs on neighborhood queries.

Usage:
    python -m tools.trace_orchestrator "What's the poverty rate in Buckhead?"
"""
from __future__ import annotations

import argparse
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

B = "\033[34m"
R = "\033[31m"
G = "\033[32m"
Y = "\033[33m"
RESET = "\033[0m"


def _watchdog(seconds: float):
    def _fire(signum, frame):
        raise TimeoutError(
            f"orchestrator exceeded {seconds}s — dumping stack"
        )
    signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("query")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="Watchdog deadline in seconds.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Turn down the noisiest HTTP logs so stage timing is readable
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    from scripts.chatbot.auth_check import check_auth
    from scripts.chatbot.llm_client import GeminiVertexClient
    from scripts.chatbot.concept_map import ConceptMap
    from scripts.chatbot.api_cache import APICache
    from scripts.chatbot.decomposition_cache import DecompositionCache
    from scripts.chatbot.semantic_router import (
        SemanticRouter, VertexEmbedder, RouterConfig,
    )
    from scripts.chatbot.frames import load_default_frames
    from scripts.chatbot.orchestrator import answer_query_sync

    with open(REPO_ROOT / "config" / "chatbot.yaml") as f:
        config = yaml.safe_load(f)

    auth = check_auth(verbose=False)
    project = auth["project_id"]
    location = auth.get("location", "us-central1")

    from scripts.chatbot.gazetteer_db import open_spatialite
    gaz_db = open_spatialite(
        REPO_ROOT / config["paths"]["gazetteer_db"]
    )
    gaz_db.row_factory = sqlite3.Row
    meta_db = sqlite3.connect(
        REPO_ROOT / config["paths"]["metadata_db"]
    )
    meta_db.row_factory = sqlite3.Row
    cmap = ConceptMap.from_yaml(
        REPO_ROOT / config["paths"]["concept_map"],
    )
    api_cache = APICache(
        REPO_ROOT / config["paths"]["api_cache_db"]
    )
    decomp_cache = DecompositionCache(
        REPO_ROOT / config["paths"]["decomposition_cache_db"]
    )

    vx = config.get("vertex_ai", {})
    llm = GeminiVertexClient(
        project_id=project, location=location,
        model=vx.get("model", "gemini-2.5-flash"),
        temperature=vx.get("temperature", 0.1),
        timeout_seconds=vx.get("timeout_seconds", 5.0),
        enable_cache=vx.get("enable_cache", True),
        cache_ttl_seconds=vx.get("cache_ttl_seconds", 3600),
    )
    faiss_path = REPO_ROOT / "data" / "metadata" / "embeddings.faiss"
    embedder = VertexEmbedder(project=project, location=location)
    router = SemanticRouter(
        faiss_path, REPO_ROOT / config["paths"]["metadata_db"],
        embedder, config=RouterConfig(),
    )
    frame_registry = load_default_frames()

    from scripts.chatbot.nodes.peer_retriever import PeerRetriever
    peer_db = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
    peer_retriever = PeerRetriever(peer_db) if peer_db.exists() else None

    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )
    ue_path = REPO_ROOT / "data" / "metadata" / "universe_embeddings.npz"
    universe_picker = None
    if ue_path.exists():
        universe_picker = UniversePicker(
            universe_emb=UniverseEmbeddings.load(ue_path),
            faiss_index=router.index,
            metadata_db=router.metadata_db,
            embedder=router.embedder,
        )

    print(f"\n{B}{'=' * 72}{RESET}")
    print(f"{B}  orchestrator trace: {args.query!r}  watchdog={args.timeout}s{RESET}")
    print(f"{B}{'=' * 72}{RESET}\n")

    _watchdog(args.timeout)
    t0 = time.time()
    try:
        resp = answer_query_sync(
            args.query, llm, gaz_db, meta_db, cmap,
            decomp_cache=decomp_cache, api_cache=api_cache,
            api_key=os.environ.get("CENSUS_API_KEY"),
            config=config, semantic_router=router,
            frame_registry=frame_registry,
            peer_retriever=peer_retriever,
            universe_picker=universe_picker,
            ask_user=None,       # auto-pick (a) on any clarifier
        )
        elapsed = time.time() - t0
    except TimeoutError as e:
        elapsed = time.time() - t0
        print(f"\n{R}TIMEOUT after {elapsed:.1f}s{RESET}")
        print(f"{R}Stack trace at hang:{RESET}\n")
        traceback.print_stack()
        return 2
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n{R}EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {e}{RESET}")
        traceback.print_exc()
        return 3
    finally:
        signal.alarm(0)

    print(f"\n{G}completed in {elapsed:.1f}s{RESET}\n")

    # Summary
    print(f"  error: {resp.error}")
    print(f"  frame: {resp.frame.name if getattr(resp, 'frame', None) else '—'}")
    print(f"  plan calls: {len(resp.plan.calls) if resp.plan else 0}")
    if resp.plan and resp.plan.calls:
        print("  calls (first 10):")
        for c in resp.plan.calls[:10]:
            print(
                f"    {c.role:22s} {c.api_call.table_id}"
                f" @ {c.api_call.geo_level} year={c.year}"
                f" vars={len(c.api_call.variables)}"
            )
        if len(resp.plan.calls) > 10:
            print(f"    ... {len(resp.plan.calls) - 10} more")
    print(f"  aggregated values: {len(resp.aggregated.values) if resp.aggregated else 0}")
    print(f"  fetch_failures: {len(resp.aggregated.fetch_failures) if resp.aggregated else 0}")
    if resp.aggregated and resp.aggregated.fetch_failures:
        for f in resp.aggregated.fetch_failures[:5]:
            print(f"    ! {f}")
    m = resp.metrics
    print(f"\n  metrics:")
    for attr in ("extract_s", "resolve_s", "clarify_s",
                 "frame_s", "decompose_s",
                 "plan_s", "plan_query_s", "plan_trend_s",
                 "plan_decomp_retry_s",
                 "fetch_s", "aggregate_s", "synthesize_s", "total_s"):
        v = getattr(m, attr, None)
        if v is not None:
            tag = (G if v < 5 else (Y if v < 20 else R))
            print(f"    {attr:14s} {tag}{v:.2f}s{RESET}")
    print(f"    llm_calls     {m.llm_calls}")
    print(f"    census_calls  {getattr(m, 'census_calls_made', 0)}")

    if getattr(resp, "peer_contexts", None):
        print(f"\n  peer_contexts ({len(resp.peer_contexts)}):")
        for pc in resp.peer_contexts:
            names = ", ".join(p.geo_name for p in pc.peers[:5])
            print(
                f"    [{pc.axis}] ({pc.pool_scope}): {names}"
            )

    if resp.answer:
        print(f"\n  answer prose (full):")
        prose = resp.answer.prose or ""
        # Print with indent; preserve paragraph breaks.
        for line in prose.split("\n"):
            print(f"    {line}")

    return 0 if resp.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
