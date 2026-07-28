"""Contract tests between the pipeline and the app layer.

The app reads a small set of names and fields from ``scripts.chatbot``.
When the pipeline changes — a renamed field, a reshaped model, a new
dataset connector — these fail with a clear message instead of the app
breaking at runtime in front of a class.

If one of these fails, the fix is usually a few lines in
``app/pipeline_adapter.py`` (for names) or ``app/schemas.py`` (for
fields), not a change to the pipeline.
"""
from __future__ import annotations

import inspect

import pytest

from app import pipeline_adapter as pa


# ---------------------------------------------------------------------
# Names the adapter re-exports
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "QueryResponse", "answer_query_sync",
    "synthesize", "build_synthesis_bundle", "bundle_inventory",
    "render_bundle_payload", "load_system_prompt",
    "SYNTHESIZER_PROMPT_PATH", "BUNDLE_SPECS",
    "SynthesizedAnswer", "SynthesisError",
    "LLMClient", "build_default_client",
])
def test_adapter_exports_exist(name):
    assert hasattr(pa, name), (
        f"app/pipeline_adapter.py re-exports {name!r}, but it is missing "
        f"from scripts.chatbot. Update the adapter."
    )


# ---------------------------------------------------------------------
# answer_query keyword arguments the app passes
# ---------------------------------------------------------------------

@pytest.mark.parametrize("kwarg", [
    "decomp_cache", "api_cache", "api_key", "config",
    "semantic_router", "frame_registry", "peer_retriever",
    "universe_picker", "ask_user", "progress_cb",
    "synth_system_prompt", "synth_options",
])
def test_answer_query_accepts_kwarg(kwarg):
    from scripts.chatbot.orchestrator import answer_query
    params = inspect.signature(answer_query).parameters
    assert kwarg in params, (
        f"app/main.py passes {kwarg!r} to answer_query, but the "
        f"signature no longer accepts it."
    )


# ---------------------------------------------------------------------
# QueryResponse fields the app reads
# ---------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "query", "intent", "resolved_geos", "plan", "aggregated", "answer",
    "frame_match", "magnitude_framings", "anomaly_flags", "followups",
    "peer_contexts", "error", "metrics",
])
def test_query_response_has_field(field):
    assert field in pa.QueryResponse.model_fields, (
        f"app/schemas.py reads QueryResponse.{field}, which no longer "
        f"exists. Update to_public()."
    )


@pytest.mark.parametrize("field", ["prose", "key_findings", "caveats",
                                   "citations"])
def test_synthesized_answer_has_field(field):
    assert field in pa.SynthesizedAnswer.model_fields, (
        f"The app reads SynthesizedAnswer.{field}, which no longer exists."
    )


# ---------------------------------------------------------------------
# The synthesis seam
# ---------------------------------------------------------------------

def test_synthesize_takes_a_bundle_first():
    params = list(inspect.signature(pa.synthesize).parameters)
    assert params[0] == "bundle", (
        "synthesize() should take the bundle as its first argument — the "
        "app builds bundles and hands them over."
    )


@pytest.mark.parametrize("kwarg", ["temperature", "system_prompt", "options"])
def test_synthesize_accepts_kwarg(kwarg):
    assert kwarg in inspect.signature(pa.synthesize).parameters


def test_bundle_specs_cover_the_configurable_keys():
    """presentation.yaml's bundle.send names real bundle keys.

    A typo there fails silently — the key just never matches — so this
    checks the shipped config against the registry.
    """
    from app.presentation import DEFAULTS
    configured = set(DEFAULTS["bundle"]["send"])
    unknown = configured - set(pa.BUNDLE_SPECS)
    assert not unknown, (
        f"presentation defaults reference bundle keys that don't exist: "
        f"{sorted(unknown)}"
    )


def test_structural_keys_are_never_sent():
    """plan/aggregated exist for citations and lints, not for the model.

    Sending them would blow up the payload and duplicate the values the
    model already has.
    """
    for key in ("plan", "aggregated"):
        assert pa.BUNDLE_SPECS[key].sendable is False


def test_prompt_falls_back_when_file_is_broken(tmp_path):
    """A student saving invalid YAML must not break answering."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("system: [a list, not a string]\n")
    from scripts.chatbot.synthesizer import _BUILTIN_SYSTEM_PROMPT
    assert pa.load_system_prompt(bad) == _BUILTIN_SYSTEM_PROMPT


def test_shipped_prompt_matches_the_builtin_constant():
    """prompts/v1/synthesizer.yaml was extracted from the Python
    constant. If they diverge, the fallback silently changes behavior."""
    from scripts.chatbot.synthesizer import _BUILTIN_SYSTEM_PROMPT
    assert pa.load_system_prompt() == _BUILTIN_SYSTEM_PROMPT


# ---------------------------------------------------------------------
# The recording proxy must stay transparent
# ---------------------------------------------------------------------

def test_recording_proxy_matches_llm_client_signature():
    from app.recording_llm import RecordingLLMClient
    from scripts.chatbot.llm_client import LLMClient

    proto = inspect.signature(LLMClient.extract).parameters
    proxy = inspect.signature(RecordingLLMClient.extract).parameters
    for name in proto:
        if name == "self":
            continue
        assert name in proxy, (
            f"LLMClient.extract takes {name!r}; the recording proxy must "
            f"accept it too or wrapping the client will break calls."
        )


def test_recording_proxy_is_skipped_for_call_site_attribution():
    """llm_client tags slow-call logs with the calling node. The proxy
    sits in that stack and must be skipped, or every call is attributed
    to the wrapper."""
    from scripts.chatbot.llm_client import _CALL_SITE_TRANSPARENT_MODULES
    assert "app.recording_llm" in _CALL_SITE_TRANSPARENT_MODULES
