"""The ONLY module in ``app/`` that imports from ``scripts.chatbot``.

Everything else in the app imports from here. When the pipeline changes
— a renamed module, a new dataset connector, a reshaped model — this one
file absorbs it and the rest of the app is untouched.

``tests/test_app_contract.py`` asserts every name and field below still
exists, so a breaking pipeline change fails a named test instead of
surfacing as a runtime error in front of a class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Pipeline entry point + result -----------------------------------
from scripts.chatbot.orchestrator import (            # noqa: E402
    QueryResponse, answer_query_sync,
)

# --- Synthesis seam (the bundle) -------------------------------------
from scripts.chatbot.synthesizer import (             # noqa: E402
    SYNTHESIZER_PROMPT_PATH,
    BUNDLE_SPECS,
    SynthesisError,
    SynthesizedAnswer,
    build_synthesis_bundle,
    bundle_inventory,
    load_system_prompt,
    render_bundle_payload,
    synthesize,
)

# --- LLM ---------------------------------------------------------------
from scripts.chatbot.llm_client import (              # noqa: E402
    LLMClient, build_default_client,
)

__all__ = [
    "REPO_ROOT", "QueryResponse", "answer_query_sync",
    "SYNTHESIZER_PROMPT_PATH", "BUNDLE_SPECS", "SynthesisError",
    "SynthesizedAnswer", "build_synthesis_bundle", "bundle_inventory",
    "load_system_prompt", "render_bundle_payload", "synthesize",
    "LLMClient", "build_default_client",
    "bundle_from_response", "open_gazetteer", "open_metadata",
    "load_concept_map", "open_caches", "load_frames", "check_vertex_auth",
    "load_semantic_router", "load_universe_picker", "load_peer_retriever",
]


# ---------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------
# faiss, spatialite and the peer DB are expensive or have native
# dependencies. Importing them lazily keeps `import app` cheap, which
# matters for tests that never touch the data layer.

def open_gazetteer(path: Path):
    from scripts.chatbot.gazetteer_db import open_spatialite
    return open_spatialite(path)


def open_metadata(path: Path):
    from scripts.chatbot.metadata_search import open_search_index
    return open_search_index(path)


def load_concept_map(**kwargs):
    from scripts.chatbot.concept_map import ConceptMap
    return ConceptMap.from_yaml(**kwargs)


def open_caches(api_cache_path: Path, decomp_cache_path: Path):
    from scripts.chatbot.api_cache import APICache
    from scripts.chatbot.decomposition_cache import DecompositionCache
    return APICache(api_cache_path), DecompositionCache(decomp_cache_path)


def load_frames():
    from scripts.chatbot.frames import load_default_frames
    return load_default_frames()


def check_vertex_auth(verbose: bool = False) -> dict:
    from scripts.chatbot.auth_check import check_auth
    return check_auth(verbose=verbose)


def auth_error_type():
    from scripts.chatbot.auth_check import AuthCheckError
    return AuthCheckError


def load_semantic_router(index_path: Path, metadata_path: Path,
                         project: str, location: str):
    from scripts.chatbot.semantic_router import SemanticRouter, VertexEmbedder
    embedder = VertexEmbedder(project=project, location=location)
    return SemanticRouter(index_path, metadata_path, embedder)


def load_universe_picker(npz_path: Path, router: Any):
    from scripts.chatbot.nodes.universe_picker import (
        UniverseEmbeddings, UniversePicker,
    )
    return UniversePicker(
        universe_emb=UniverseEmbeddings.load(npz_path),
        faiss_index=router.index,
        metadata_db=router.metadata_db,
        embedder=router.embedder,
    )


def load_peer_retriever(db_path: Path):
    from scripts.chatbot.nodes.peer_retriever import PeerRetriever
    return PeerRetriever(db_path)


# ---------------------------------------------------------------------
# Bundle reconstruction
# ---------------------------------------------------------------------

def bundle_from_response(
    resp: QueryResponse,
    frame_registry: Optional[Any] = None,
) -> dict[str, Any]:
    """Rebuild the synthesis bundle from a completed QueryResponse.

    The orchestrator builds a bundle internally and discards it; the app
    needs one to power re-synthesis and the upstream inventory. Rather
    than adding a side channel to the pipeline, the bundle is rebuilt
    from the response, which carries every input that went into it.

    Both paths call the same ``build_synthesis_bundle``, so the shape
    can't drift between them.

    ``frame`` is the one field not stored on QueryResponse — it holds a
    ``frame_match`` and the Frame is looked up from the registry.
    """
    frame = None
    frame_match = getattr(resp, "frame_match", None)
    if frame_match is not None and frame_registry is not None:
        try:
            frame = frame_registry.get(frame_match.frame)
        except Exception:
            frame = None

    plan = getattr(resp, "plan", None)
    return build_synthesis_bundle(
        getattr(resp, "query", ""),
        getattr(resp, "intent", None),
        plan,
        getattr(resp, "aggregated", None),
        resolved_geos=getattr(resp, "resolved_geos", None),
        frame=frame,
        magnitude_framings=getattr(resp, "magnitude_framings", None),
        anomaly_flags=getattr(resp, "anomaly_flags", None),
        followups=getattr(resp, "followups", None),
        peer_contexts=getattr(resp, "peer_contexts", None),
        concept_resolutions=getattr(plan, "concept_resolutions", None),
    )
