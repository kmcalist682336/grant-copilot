"""Loads the heavy artifacts once, at startup.

The pipeline needs ~8 GB of on-disk artifacts (gazetteer, metadata
index, FAISS vectors, peer features). Building those per request would
be absurd, so they're constructed once here and injected into every
``answer_query`` call — possible because the orchestrator already takes
all of them as parameters.

Missing artifacts are reported with the hydration command rather than a
bare stack trace: this repo ships without them on purpose (see
``scripts/setup/hydrate_data_artifacts.py`` and
``config/data_manifest.yaml``).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from app import pipeline_adapter as pa
from app.recording_llm import RecordingLLMClient

REPO_ROOT = pa.REPO_ROOT
logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]

HYDRATE_HINT = (
    "Run:  python -m scripts.setup.hydrate_data_artifacts\n"
    "(needs HF_TOKEN — see the 'Hydrate the data layer' section of the "
    "README)"
)


@dataclass
class ArtifactStatus:
    """What loaded, what didn't, and why — surfaced on /readyz and in the
    UI so a missing 4 GB index is obvious rather than mysterious."""

    name: str
    ok: bool
    detail: str = ""
    required: bool = False


@dataclass
class AppContext:
    """Everything ``answer_query`` needs, built once."""

    config: dict
    llm: Any
    db: Any
    metadata_db: Any
    cmap: Any
    decomp_cache: Any
    api_cache: Any
    api_key: Optional[str]
    record_caller: Any = None
    record_connection: Any = None
    semantic_router: Any = None
    universe_picker: Any = None
    peer_retriever: Any = None
    frame_registry: Any = None
    statuses: list[ArtifactStatus] = field(default_factory=list)
    load_seconds: float = 0.0

    def status_dict(self) -> dict[str, Any]:
        return {
            "ready": True,
            "load_seconds": round(self.load_seconds, 1),
            "artifacts": [
                {"name": s.name, "ok": s.ok, "detail": s.detail,
                 "required": s.required}
                for s in self.statuses
            ],
        }


def load_config() -> dict:
    with (REPO_ROOT / "config" / "chatbot.yaml").open() as f:
        return yaml.safe_load(f)


def _missing(path: Path, what: str) -> RuntimeError:
    return RuntimeError(
        f"{what} not found at {path.relative_to(REPO_ROOT)}.\n\n"
        f"This repo ships without the data layer.\n{HYDRATE_HINT}"
    )


def build_context(
    progress: Optional[ProgressFn] = None,
    *,
    no_router: bool = False,
    record_llm: bool = True,
) -> AppContext:
    """Construct the full application context.

    ``progress`` is called as (status, message) per step — loading 4 GB
    of vectors in silence looks like a hang.

    ``record_llm`` wraps the LLM client so the trace pane can show every
    node's real prompt and response. Costs nothing but memory.
    """
    t_start = time.time()

    def say(status: str, message: str) -> None:
        if progress:
            try:
                progress(status, message)
            except Exception:
                pass

    statuses: list[ArtifactStatus] = []
    config = load_config()

    # --- LLM ---------------------------------------------------------
    say("...", "Checking Vertex AI credentials")
    try:
        auth = pa.check_vertex_auth(verbose=False)
    except Exception as e:
        raise RuntimeError(
            f"Vertex AI auth failed: {e}\n\n"
            f"Run:  gcloud auth application-default login\n"
            f"and set GCP_PROJECT_ID in .env (see the README's "
            f"'Credentials setup' section)."
        ) from e

    config.setdefault("vertex_ai", {})["project_id"] = auth["project_id"]
    config["vertex_ai"].setdefault(
        "location", auth.get("location", "us-central1"))
    llm: Any = pa.build_default_client(config)
    if record_llm:
        llm = RecordingLLMClient(llm)
    statuses.append(ArtifactStatus(
        "LLM", True,
        f"Vertex {config['vertex_ai'].get('model')} ({auth['project_id']})",
        required=True,
    ))
    say("ok", f"Vertex AI ready — project {auth['project_id']}")

    # --- Gazetteer (required) ---------------------------------------
    say("...", "Opening gazetteer (~273 MB)")
    gaz_path = REPO_ROOT / config["paths"]["gazetteer_db"]
    if not gaz_path.exists():
        raise _missing(gaz_path, "Gazetteer")
    db = pa.open_gazetteer(gaz_path)
    statuses.append(ArtifactStatus("Gazetteer", True, gaz_path.name, True))
    say("ok", "Gazetteer loaded")

    # --- Metadata index (required) ----------------------------------
    say("...", "Opening metadata index (~3.2 GB)")
    meta_path = REPO_ROOT / config["paths"]["metadata_db"]
    if not meta_path.exists():
        raise _missing(meta_path, "Metadata index")
    metadata_db = pa.open_metadata(meta_path)
    statuses.append(ArtifactStatus("Metadata index", True, meta_path.name, True))
    say("ok", "Metadata index loaded")

    # --- Concept map (required) -------------------------------------
    cmap = pa.load_concept_map()
    statuses.append(ArtifactStatus(
        "Concept map", True, f"{len(cmap)} curated concepts", True))
    say("ok", f"Concept map: {len(cmap)} concepts")

    # --- Caches ------------------------------------------------------
    api_cache, decomp_cache = pa.open_caches(
        REPO_ROOT / config["paths"]["api_cache_db"],
        REPO_ROOT / config["paths"]["decomposition_cache_db"],
    )
    statuses.append(ArtifactStatus("Caches", True, "api + decomposition"))

    api_key = os.environ.get(
        config.get("census_api", {}).get("api_key_env_var", "CENSUS_API_KEY"))
    statuses.append(ArtifactStatus(
        "Census API key", bool(api_key),
        "set" if api_key else "missing — rate-limited to 500 calls/day"))

    # Optional record-level connector. The Census-only app remains fully
    # functional when HMDA_RECORD_ROOT is unset. Connector construction is
    # isolated here so the orchestrator can dispatch by dataset without
    # learning anything about GCS credentials or DuckDB setup.
    record_caller = None
    record_connection = None
    hmda_root = os.environ.get("HMDA_RECORD_ROOT", "").strip()
    if hmda_root:
        try:
            from scripts.chatbot.record_connector import build_record_caller
            record_caller, record_connection = build_record_caller(
                hmda_root,
                api_cache,
                record_id_column=os.environ.get(
                    "HMDA_RECORD_ID_COLUMN", "record_id"),
                layout=os.environ.get(
                    "HMDA_RECORD_LAYOUT", "variable_tree"),
                file_glob=os.environ.get(
                    "HMDA_RECORD_FILE_GLOB", "hmda_*.parquet"),
                geography_partition=(os.environ.get(
                    "HMDA_GEOGRAPHY_PARTITION", "").strip() or None),
            )
            statuses.append(ArtifactStatus(
                "HMDA record connector", True, hmda_root))
            say("ok", f"HMDA DuckDB connector: {hmda_root}")
        except Exception as e:
            logger.warning("HMDA connector unavailable: %s", e)
            statuses.append(ArtifactStatus(
                "HMDA record connector", False, f"failed: {e}"))
            say("warn", f"HMDA connector unavailable: {e}")
    else:
        statuses.append(ArtifactStatus(
            "HMDA record connector", False, "not configured (optional)"))

    # --- Semantic router (optional, the 4.4 GB one) -----------------
    semantic_router = None
    if no_router:
        statuses.append(ArtifactStatus(
            "Semantic router", False, "disabled (--no-router)"))
        say("skip", "Semantic router disabled — legacy tier cascade")
    else:
        index_path = REPO_ROOT / config["paths"].get(
            "embedding_index", "data/metadata/embeddings.faiss")
        if not index_path.exists():
            statuses.append(ArtifactStatus(
                "Semantic router", False,
                f"{index_path.name} missing — legacy tier cascade"))
            say("warn", f"embeddings.faiss missing. {HYDRATE_HINT}")
        else:
            say("...", "Loading FAISS index (~4.4 GB — the slow step)")
            try:
                semantic_router = pa.load_semantic_router(
                    index_path, meta_path,
                    auth["project_id"], auth.get("location", "us-central1"))
                detail = (f"{semantic_router.index.ntotal:,} vectors "
                          f"(dim={semantic_router.index.d})")
                statuses.append(ArtifactStatus("Semantic router", True, detail))
                say("ok", f"Semantic router: {detail}")
            except Exception as e:
                logger.warning("semantic router unavailable: %s", e)
                statuses.append(ArtifactStatus(
                    "Semantic router", False, f"failed: {e}"))
                say("warn", f"Semantic router unavailable: {e}")

    # --- Universe picker (optional) ---------------------------------
    universe_picker = None
    if semantic_router is not None:
        ue_path = REPO_ROOT / "data" / "metadata" / "universe_embeddings.npz"
        if ue_path.exists():
            try:
                universe_picker = pa.load_universe_picker(
                    ue_path, semantic_router)
                statuses.append(ArtifactStatus(
                    "Universe picker", True, ue_path.name))
                say("ok", "Universe picker loaded")
            except Exception as e:
                logger.warning("universe picker unavailable: %s", e)
                statuses.append(ArtifactStatus(
                    "Universe picker", False, f"failed: {e}"))
        else:
            statuses.append(ArtifactStatus(
                "Universe picker", False,
                "universe_embeddings.npz missing — legacy composer"))

    # --- Peer retriever (optional) ----------------------------------
    peer_retriever = None
    peer_db = REPO_ROOT / "data" / "metadata" / "peer_features.sqlite"
    if peer_db.exists():
        try:
            peer_retriever = pa.load_peer_retriever(peer_db)
            statuses.append(ArtifactStatus("Peer retriever", True, peer_db.name))
            say("ok", "Peer retriever loaded")
        except Exception as e:
            logger.warning("peer retriever unavailable: %s", e)
            statuses.append(ArtifactStatus(
                "Peer retriever", False, f"failed: {e}"))
    else:
        statuses.append(ArtifactStatus(
            "Peer retriever", False,
            "peer_features.sqlite missing — no peer context"))

    # --- Frame registry (optional) ----------------------------------
    frame_registry = None
    try:
        frame_registry = pa.load_frames()
        statuses.append(ArtifactStatus("Grant frames", True, "loaded"))
    except Exception as e:
        logger.warning("frame registry unavailable: %s", e)
        statuses.append(ArtifactStatus("Grant frames", False, f"failed: {e}"))

    elapsed = time.time() - t_start
    say("ok", f"Ready in {elapsed:.1f}s")

    return AppContext(
        config=config, llm=llm, db=db, metadata_db=metadata_db, cmap=cmap,
        decomp_cache=decomp_cache, api_cache=api_cache, api_key=api_key,
        record_caller=record_caller, record_connection=record_connection,
        semantic_router=semantic_router, universe_picker=universe_picker,
        peer_retriever=peer_retriever, frame_registry=frame_registry,
        statuses=statuses, load_seconds=elapsed,
    )
