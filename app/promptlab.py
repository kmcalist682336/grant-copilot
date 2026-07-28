"""The prompt iteration loop.

Three operations, three costs — this is the point of the app:

    re-render      ~0 ms, $0        layers 3+4 on cached output
    re-synthesize  ~3 s, ~$0.0005   synthesizer LLM only, cached bundle
    ask            ~15-25 s, full   the entire pipeline

Most iteration is the first two. Re-synthesize works because
``synthesize`` takes one bundle — a stored run has everything needed to
run the last stage again against identical numbers, with no extraction,
no routing, and no data fetches.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from app import lints as lint_mod
from app import postprocess
from app.pipeline_adapter import (
    REPO_ROOT, bundle_inventory, render_bundle_payload, synthesize,
)
from app.presentation import PresentationConfig
from app.schemas import ChatResponse, to_public

logger = logging.getLogger(__name__)

VARIANTS_DIR = REPO_ROOT / "prompts" / "variants"

# Cached runs hold full bundles. Bounded so a long session can't grow
# without limit.
MAX_CACHED_RUNS = 25

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass
class StoredRun:
    """One completed run, kept so its bundle can be reused."""

    run_id: str
    resp: Any                       # orchestrator.QueryResponse
    bundle: dict = field(default_factory=dict)
    raw_answer: Any = None          # pre-postprocess, for the trace
    trace: dict = field(default_factory=dict)
    bundle_options: dict = field(default_factory=dict)


class RunStore:
    """Bounded, thread-safe LRU of recent runs."""

    def __init__(self, maxsize: int = MAX_CACHED_RUNS):
        self._runs: "OrderedDict[str, StoredRun]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def put(self, run: StoredRun) -> None:
        with self._lock:
            self._runs[run.run_id] = run
            self._runs.move_to_end(run.run_id)
            while len(self._runs) > self._maxsize:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> Optional[StoredRun]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                self._runs.move_to_end(run_id)
            return run

    def latest(self) -> Optional[StoredRun]:
        with self._lock:
            if not self._runs:
                return None
            return next(reversed(self._runs.values()))


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------
# Render — layers 3 and 4, no LLM
# ---------------------------------------------------------------------

def render_run(
    run: StoredRun,
    pres: PresentationConfig,
    *,
    config_warning: Optional[str] = None,
) -> ChatResponse:
    """Apply postprocess + lints + the public mapping to a stored run.

    Pure and cheap — this is what "Re-render" calls. Re-reads the
    presentation config each time, so flipping a section off costs
    nothing.
    """
    raw_answer = run.raw_answer
    cleaned = None
    if raw_answer is not None:
        try:
            cleaned = postprocess.clean(raw_answer, run.bundle, pres.formatting)
        except Exception as e:
            # A student's custom_clean must not take the app down.
            logger.warning("postprocess.clean failed: %s", e)
            cleaned = raw_answer
            config_warning = (
                f"{config_warning + ' ' if config_warning else ''}"
                f"postprocess.clean raised {type(e).__name__}: {e}"
            )

    lint_results = lint_mod.run_all(
        cleaned,
        run.bundle.get("aggregated"),
        run.bundle.get("peer_contexts"),
        pres.lints,
    )

    try:
        inventory = bundle_inventory(run.bundle, run.bundle_options)
    except Exception as e:                              # pragma: no cover
        logger.warning("bundle_inventory failed: %s", e)
        inventory = []

    return to_public(
        run.resp,
        sections=pres.sections,
        lint_results=lint_results,
        upstream=inventory,
        trace=run.trace,
        bundle_options=run.bundle_options,
        answer_override=cleaned,
        config_warning=config_warning,
        run_id=run.run_id,
    )


# ---------------------------------------------------------------------
# Re-synthesize — the synthesizer LLM only
# ---------------------------------------------------------------------

def resynthesize(
    run: StoredRun,
    llm: Any,
    pres: PresentationConfig,
    *,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
) -> StoredRun:
    """Re-run the final LLM call against this run's cached bundle.

    Identical numbers, new writeup. Returns a NEW StoredRun so the
    previous one stays available for comparison.

    ``system_prompt`` is an unsaved draft from the editor; when omitted
    the on-disk YAML is re-read.
    """
    if not run.bundle or run.bundle.get("aggregated") is None:
        raise ValueError(
            "This run has no data to re-synthesize from — the original "
            "query failed before any numbers were retrieved."
        )

    options = dict(pres.bundle_options)
    answer = synthesize(
        run.bundle, llm,
        temperature=temperature,
        system_prompt=system_prompt,
        options=options,
    )

    trace = {
        "system_prompt": system_prompt or "",
        "user_payload": render_bundle_payload(run.bundle, options),
        "raw_response": answer.model_dump() if hasattr(answer, "model_dump") else None,
        "calls": [],
    }
    # A recording client fills in the real strings; fall back to the
    # rendered payload above when the LLM isn't wrapped.
    if hasattr(llm, "synthesis_call"):
        call = llm.synthesis_call()
        if call:
            trace = {
                "system_prompt": call.get("system_prompt", ""),
                "user_payload": call.get("user_text", ""),
                "raw_response": call.get("response"),
                "calls": llm.calls() if hasattr(llm, "calls") else [],
            }

    new_resp = run.resp.model_copy(update={"answer": answer, "error": None})
    return StoredRun(
        run_id=new_run_id(),
        resp=new_resp,
        bundle=run.bundle,
        raw_answer=answer,
        trace=trace,
        bundle_options=options,
    )


# ---------------------------------------------------------------------
# Named prompt variants
# ---------------------------------------------------------------------

def _variant_path(name: str) -> Path:
    if not _SAFE_NAME.match(name):
        raise ValueError(
            "Variant names may use letters, numbers, dot, dash and "
            "underscore only (max 64 chars)."
        )
    return VARIANTS_DIR / f"{name}.yaml"


def list_variants() -> list[str]:
    if not VARIANTS_DIR.exists():
        return []
    return sorted(p.stem for p in VARIANTS_DIR.glob("*.yaml"))


def read_variant(name: str) -> str:
    path = _variant_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No prompt variant named {name!r}.")
    doc = yaml.safe_load(path.read_text()) or {}
    text = doc.get("system")
    if not isinstance(text, str):
        raise ValueError(f"Variant {name!r} has no 'system' key.")
    return text


def save_variant(name: str, system_prompt: str) -> Path:
    path = _variant_path(name)
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        {"system": system_prompt}, default_style="|", sort_keys=False,
        allow_unicode=True, width=10**6,
    ))
    return path


def delete_variant(name: str) -> bool:
    path = _variant_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
