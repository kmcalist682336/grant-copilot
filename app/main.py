"""FastAPI application.

Owns HTTP and nothing else. Pipeline work goes through
``orchestrator.answer_query``; presentation goes through
``app.presentation`` + ``app.postprocess``.

Endpoints
    GET  /                      the UI
    GET  /healthz               process alive
    GET  /readyz                artifacts loaded (separate on purpose —
                                the FAISS index takes a while)
    POST /chat                  full pipeline
    GET  /chat/stream           full pipeline, SSE stage events
    POST /resynthesize          synthesizer LLM only, cached bundle
    POST /rerender              layers 3+4 only, no LLM
    GET  /api/upstream/{run_id} the synthesis bundle inventory
    GET/PUT /api/prompt         the synthesizer prompt
    GET/PUT /api/presentation   presentation.yaml
    GET/POST/DELETE /api/variants[/{name}]
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import pipeline_adapter as pa
from app import presentation as pres_mod
from app import promptlab
from app import setup_state
from app.context import AppContext, build_context
from app.promptlab import RunStore, StoredRun, new_run_id, render_run
from app.schemas import (
    ChatRequest, ChatResponse, RerenderRequest, ResynthesizeRequest,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

CTX: Optional[AppContext] = None
STARTUP_ERROR: Optional[str] = None
LOADING = False
_INIT_LOCK = threading.Lock()
RUNS = RunStore()

# Set by run_app.py before uvicorn imports this module.
BOOT_OPTIONS: dict[str, Any] = {"no_router": False, "record_llm": True}

# Every piece of pipeline work runs on THIS ONE THREAD, including the
# initial artifact load.
#
# The gazetteer and metadata index are sqlite3 connections, and sqlite3
# refuses to let a connection be used from a thread other than the one
# that created it. Handing requests to arbitrary threadpool workers fails
# with "SQLite objects created in a thread can only be used in that same
# thread".
#
# Pinning to one thread also matches what the pipeline is: one process,
# one big in-memory index, blocking SQLite and FAISS calls. Queries
# serialize — fine for a prompt lab. Making it concurrent means
# per-thread connections and a FAISS access strategy, not a bigger pool.
PIPELINE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="pipeline")


async def _on_pipeline_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        PIPELINE_EXECUTOR, functools.partial(fn, *args, **kwargs))


def _progress(status: str, message: str) -> None:
    symbol = {"ok": "✓", "warn": "!", "skip": "-"}.get(status, "…")
    print(f"  {symbol} {message}", flush=True)


def _initialize() -> None:
    """Build the app context. Safe to call repeatedly; no-op once ready.

    Separate from startup so the server can come up unconfigured, collect
    what's missing through the UI, and then initialize without a restart.
    """
    global CTX, STARTUP_ERROR, LOADING
    with _INIT_LOCK:
        if CTX is not None or LOADING:
            return
        LOADING = True
        STARTUP_ERROR = None

    try:
        print("\nLoading artifacts — the FAISS index is ~4.4 GB, "
              "expect this to take a minute.\n", flush=True)
        ctx = build_context(
            _progress,
            no_router=BOOT_OPTIONS.get("no_router", False),
            record_llm=BOOT_OPTIONS.get("record_llm", True))
        with _INIT_LOCK:
            CTX = ctx
        print("\n  Ready.\n", flush=True)
    except Exception as e:
        with _INIT_LOCK:
            STARTUP_ERROR = str(e)
        logger.exception("initialization failed")
        print(f"\n  NOT READY: {e}\n", flush=True)
    finally:
        with _INIT_LOCK:
            LOADING = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start serving immediately, configured or not.

    `docker compose up` is non-interactive: a container that exits on a
    missing env var leaves the user with a traceback and nowhere to type.
    So we check what's missing, report it, and let the UI collect the
    fixable parts — the server is up either way.
    """
    setup_state.load_dotenv()
    rep = setup_state.report()

    if rep.ready:
        # Off the event loop so /healthz answers while the index loads.
        asyncio.get_running_loop().run_in_executor(
            PIPELINE_EXECUTOR, _initialize)
    else:
        missing = ", ".join(c.label for c in rep.blocking)
        print(
            f"\n  Not configured yet — missing: {missing}\n"
            f"  Open the UI (URL above) and finish setup there.\n",
            flush=True,
        )
    yield


app = FastAPI(title="grant-copilot", lifespan=lifespan)


def _require_ctx() -> AppContext:
    if CTX is not None:
        return CTX
    if LOADING:
        raise HTTPException(503, "Still loading artifacts — try again shortly.")
    if STARTUP_ERROR:
        raise HTTPException(503, f"Not ready: {STARTUP_ERROR}")
    rep = setup_state.report()
    missing = ", ".join(c.label for c in rep.blocking) or "configuration"
    raise HTTPException(503, f"Setup incomplete — missing: {missing}")


def app_status() -> str:
    """needs_setup | loading | failed | ready"""
    if CTX is not None:
        return "ready"
    if LOADING:
        return "loading"
    if STARTUP_ERROR:
        return "failed"
    return "needs_setup"


# ---------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------

def _run_pipeline(
    ctx: AppContext,
    query: str,
    pres: pres_mod.PresentationConfig,
    progress_cb: Any = None,
) -> StoredRun:
    """Execute the full pipeline synchronously and store the result."""
    options = dict(pres.bundle_options)

    if hasattr(ctx.llm, "reset"):
        ctx.llm.reset()          # fresh trace per query

    resp = pa.answer_query_sync(
        query, ctx.llm, ctx.db, ctx.metadata_db, ctx.cmap,
        decomp_cache=ctx.decomp_cache,
        api_cache=ctx.api_cache,
        api_key=ctx.api_key,
        config=ctx.config,
        semantic_router=ctx.semantic_router,
        frame_registry=ctx.frame_registry,
        peer_retriever=ctx.peer_retriever,
        universe_picker=ctx.universe_picker,
        ask_user=None,           # clarifier degrades to "proceed" over HTTP
        progress_cb=progress_cb,
        synth_options=options,
    )

    # Rebuild the bundle the orchestrator used, so re-synthesis and the
    # upstream panel work off the same shape. Both paths call
    # build_synthesis_bundle, so they can't drift.
    bundle = pa.bundle_from_response(resp, ctx.frame_registry)

    trace: dict[str, Any] = {}
    if hasattr(ctx.llm, "synthesis_call"):
        call = ctx.llm.synthesis_call()
        if call:
            trace = {
                "system_prompt": call.get("system_prompt", ""),
                "user_text": call.get("user_text", ""),
                "user_payload": call.get("user_text", ""),
                "raw_response": call.get("response"),
            }
        trace["calls"] = ctx.llm.calls() if hasattr(ctx.llm, "calls") else []

    run = StoredRun(
        run_id=new_run_id(), resp=resp, bundle=bundle,
        raw_answer=getattr(resp, "answer", None),
        trace=trace, bundle_options=options,
    )
    RUNS.put(run)
    return run


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "alive"}


@app.get("/readyz")
def readyz() -> dict:
    if CTX is None:
        raise HTTPException(503, {"ready": False, "status": app_status(),
                                  "error": STARTUP_ERROR or "not initialized"})
    return CTX.status_dict()


# ---------------------------------------------------------------------
# Setup — collect what's missing without a restart
# ---------------------------------------------------------------------

class SetupPayload(BaseModel):
    """Values the browser may write to .env. All optional: the form
    submits only the fields the user filled in."""

    gcp_project: Optional[str] = None
    gcp_location: Optional[str] = None
    census_key: Optional[str] = None
    hf_token: Optional[str] = None


@app.get("/api/setup")
def get_setup() -> dict:
    """What's configured, what's missing, and how each gap is fixed.

    Secrets are reported as set/not-set, never echoed back.
    """
    rep = setup_state.report()
    return {
        "status": app_status(),
        "error": STARTUP_ERROR,
        "in_docker": Path("/.dockerenv").exists(),
        "hydrate": setup_state.HYDRATE.status(),
        **rep.to_dict(),
    }


@app.post("/api/setup")
def post_setup(payload: SetupPayload) -> dict:
    """Write the supplied values to .env and re-check.

    os.environ is updated too, so the values take effect in this process
    without a restart.
    """
    updates: dict[str, str] = {}
    if payload.gcp_project and payload.gcp_project.strip():
        updates["GCP_PROJECT_ID"] = payload.gcp_project.strip()
    if payload.gcp_location and payload.gcp_location.strip():
        updates["GCP_LOCATION"] = payload.gcp_location.strip()
    if payload.census_key and payload.census_key.strip():
        updates["CENSUS_API_KEY"] = payload.census_key.strip()
    if payload.hf_token and payload.hf_token.strip():
        updates["HF_TOKEN"] = payload.hf_token.strip()

    if updates:
        try:
            setup_state.write_env(updates)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except OSError as e:
            raise HTTPException(
                500,
                f"Could not write .env: {e}. In Docker this usually means "
                f"the repo mount isn't writable by the container user — "
                f"set APP_UID/APP_GID to your host ids.",
            )

    rep = setup_state.report()
    return {"saved": sorted(updates), "status": app_status(), **rep.to_dict()}


@app.post("/api/setup/initialize")
async def post_initialize() -> dict:
    """Load the artifacts now that configuration is in place."""
    if CTX is not None:
        return {"status": "ready"}
    if LOADING:
        return {"status": "loading"}

    rep = setup_state.report()
    if not rep.ready:
        raise HTTPException(
            400,
            "Still missing: " + ", ".join(c.label for c in rep.blocking))

    asyncio.get_running_loop().run_in_executor(PIPELINE_EXECUTOR, _initialize)
    return {"status": "loading"}


@app.post("/api/setup/hydrate")
def post_hydrate() -> dict:
    """Download the ~8 GB data layer."""
    if not os.environ.get("HF_TOKEN", "").strip():
        raise HTTPException(
            400, "HF_TOKEN is required to download the data layer.")
    if not setup_state.HYDRATE.start():
        raise HTTPException(409, "A download is already running.")
    return {"started": True}


@app.get("/api/setup/hydrate")
def get_hydrate() -> dict:
    return setup_state.HYDRATE.status()


@app.delete("/api/setup/hydrate")
def cancel_hydrate() -> dict:
    return {"cancelled": setup_state.HYDRATE.cancel()}


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    ctx = _require_ctx()
    if not req.query.strip():
        raise HTTPException(400, "Query is empty.")
    pres = pres_mod.load()
    try:
        run = await _on_pipeline_thread(_run_pipeline, ctx, req.query, pres)
    except Exception as e:
        # The orchestrator normally returns a partial QueryResponse rather
        # than raising, so reaching here means something outside its own
        # error handling broke. Report it in the contract's shape.
        logger.exception("pipeline raised")
        return ChatResponse(status="error", query=req.query, error=str(e),
                            config_warning=pres.error)
    return render_run(run, pres, config_warning=pres.error)


@app.get("/chat/stream")
async def chat_stream(q: str = Query(..., description="User query")):
    """Same pipeline, streamed as SSE stage events.

    The pipeline takes 15-25 s; without progress the UI looks hung.
    Stage names come from the orchestrator's existing progress_cb hook.
    """
    ctx = _require_ctx()
    if not q.strip():
        raise HTTPException(400, "Query is empty.")
    pres = pres_mod.load()

    events: "queue.Queue[Optional[dict]]" = queue.Queue()

    def progress_cb(stage: str, detail: Optional[str] = None) -> None:
        events.put({"type": "stage", "stage": stage, "detail": detail or ""})

    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            run = _run_pipeline(ctx, q, pres, progress_cb)
            result["response"] = render_run(run, pres, config_warning=pres.error)
        except Exception as e:
            logger.exception("pipeline failed")
            result["error"] = str(e)
        finally:
            events.put(None)

    async def stream():
        # Submitted to the pipeline thread — see PIPELINE_EXECUTOR.
        PIPELINE_EXECUTOR.submit(worker)
        while True:
            try:
                evt = await asyncio.to_thread(events.get, True, 1.0)
            except queue.Empty:
                yield ": keepalive\n\n"     # stop proxies idling us out
                continue
            if evt is None:
                break
            yield f"data: {json.dumps(evt)}\n\n"

        if "error" in result:
            payload = {"type": "error", "error": result["error"]}
        else:
            payload = {"type": "result",
                       "response": result["response"].model_dump(mode="json")}
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/resynthesize", response_model=ChatResponse)
async def resynthesize(req: ResynthesizeRequest) -> ChatResponse:
    """Re-run only the synthesizer against a cached bundle. ~3 s, no data
    fetches, identical numbers."""
    ctx = _require_ctx()
    run = RUNS.get(req.run_id)
    if run is None:
        raise HTTPException(404, "That run is no longer cached — re-ask.")
    pres = pres_mod.load()
    try:
        new_run = await _on_pipeline_thread(
            promptlab.resynthesize, run, ctx.llm, pres,
            temperature=ctx.config.get("vertex_ai", {}).get(
                "synth_temperature", 0.2),
            system_prompt=req.system_prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("resynthesize failed")
        raise HTTPException(500, f"Re-synthesis failed: {e}")
    RUNS.put(new_run)
    return render_run(new_run, pres, config_warning=pres.error)


@app.post("/rerender", response_model=ChatResponse)
async def rerender(req: RerenderRequest) -> ChatResponse:
    """Re-apply presentation config + postprocess. No LLM call, no cost."""
    _require_ctx()
    run = RUNS.get(req.run_id)
    if run is None:
        raise HTTPException(404, "That run is no longer cached — re-ask.")
    pres = pres_mod.load()
    return render_run(run, pres, config_warning=pres.error)


@app.get("/api/upstream/{run_id}")
def get_upstream(run_id: str) -> dict:
    """Every artifact available to the synthesizer for this run.

    Derived from the bundle's keys, so a newly added artifact appears
    here with no code change.
    """
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "That run is no longer cached — re-ask.")
    return {
        "run_id": run_id,
        "options": run.bundle_options,
        "items": pa.bundle_inventory(run.bundle, run.bundle_options),
        "payload_sent": pa.render_bundle_payload(run.bundle, run.bundle_options),
    }


# ---------------------------------------------------------------------
# Editable config surfaces
# ---------------------------------------------------------------------

class TextPayload(BaseModel):
    text: str


class VariantPayload(BaseModel):
    name: str
    system_prompt: str


@app.get("/api/prompt")
def get_prompt() -> dict:
    return {
        "text": pa.load_system_prompt(),
        "path": str(pa.SYNTHESIZER_PROMPT_PATH.relative_to(pa.REPO_ROOT)),
    }


@app.put("/api/prompt")
def put_prompt(payload: TextPayload) -> dict:
    """Save the synthesizer prompt. Hot-reloaded — the next Ask or
    Re-synthesize picks it up with no restart."""
    if not payload.text.strip():
        raise HTTPException(400, "Prompt cannot be empty.")
    pa.SYNTHESIZER_PROMPT_PATH.write_text(yaml.safe_dump(
        {"system": payload.text}, default_style="|", sort_keys=False,
        allow_unicode=True, width=10**6))
    return {"saved": True}


@app.get("/api/presentation")
def get_presentation() -> dict:
    cfg = pres_mod.load()
    return {"text": pres_mod.read_text(),
            "path": "config/presentation.yaml", "error": cfg.error}


@app.put("/api/presentation")
def put_presentation(payload: TextPayload) -> dict:
    err = pres_mod.write_text(payload.text)
    if err:
        raise HTTPException(400, err)
    return {"saved": True}


@app.get("/api/variants")
def get_variants() -> dict:
    return {"variants": promptlab.list_variants()}


@app.get("/api/variants/{name}")
def get_variant(name: str) -> dict:
    try:
        return {"name": name, "system_prompt": promptlab.read_variant(name)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/variants")
def post_variant(payload: VariantPayload) -> dict:
    try:
        path = promptlab.save_variant(payload.name, payload.system_prompt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"saved": True, "path": path.name}


@app.delete("/api/variants/{name}")
def delete_variant(name: str) -> dict:
    try:
        return {"deleted": promptlab.delete_variant(name)}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
