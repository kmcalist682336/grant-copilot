"""HTTP-level tests for the app.

Exercises the routes without loading the ~8 GB of artifacts, by
injecting a stub context and a pre-built run into the store. Enough to
catch route wiring, request/response schemas, and the config editors.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.models import ExtractedConcept
from scripts.chatbot.orchestrator import QueryResponse, StageMetrics
from scripts.chatbot.synthesizer import SynthesizedAnswer

from tests.test_synthesizer import _entry, _geo, _make_plan

from app import main as app_main
from app.promptlab import StoredRun


@pytest.fixture
def client(monkeypatch):
    """TestClient with a stub context and NO lifespan.

    Constructing TestClient without the `with` form deliberately skips
    startup — otherwise every test would load the real ~8 GB artifact
    set and hit Vertex auth.
    """
    monkeypatch.setattr(app_main, "CTX", object())
    monkeypatch.setattr(app_main, "STARTUP_ERROR", None)
    return TestClient(app_main.app, raise_server_exceptions=False)


@pytest.fixture
def stored_run():
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="median income")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    resp = QueryResponse(
        query="median income in DeKalb",
        intent=plan.intent,
        resolved_geos=[geo],
        plan=plan,
        aggregated=AggregatedResult(values=[
            AggregatedValue(geo=geo, concept=concept, year=2023,
                            value=72483.0, rows_aggregated=1),
        ]),
        answer=SynthesizedAnswer(
            prose="In 2023, median household income was $72,500.",
            key_findings=["Median income: $72,500"],
            caveats=[], citations=[],
        ),
        metrics=StageMetrics(total_s=3.2, llm_calls=2),
    )
    run = StoredRun(
        run_id="testrun", resp=resp, raw_answer=resp.answer,
        trace={"system_prompt": "SYS", "user_payload": "{}",
               "raw_response": {"prose": "x"}},
        bundle_options={"include_peer_contexts": False},
    )
    app_main.RUNS.put(run)
    return run


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "grant-copilot" in r.text


def test_rerender_returns_public_contract(client, stored_run):
    r = client.post("/rerender", json={"run_id": "testrun"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["run_id"] == "testrun"
    assert "$72,500" in body["prose"]
    assert body["values"][0]["geo"] == "DeKalb County"
    assert body["trace"]["system_prompt"] == "SYS"
    # The public DTO must not leak internal pipeline shapes.
    for internal in ("intent", "plan", "aggregated", "frame_match"):
        assert internal not in body


def test_rerender_unknown_run_is_404(client):
    r = client.post("/rerender", json={"run_id": "nope"})
    assert r.status_code == 404


def test_chat_rejects_empty_query(client):
    r = client.post("/chat", json={"query": "   "})
    assert r.status_code == 400


def test_get_prompt(client):
    r = client.get("/api/prompt")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "prompts/v1/synthesizer.yaml"
    assert "Census data synthesizer" in body["text"]


def test_put_prompt_rejects_empty(client):
    assert client.put("/api/prompt", json={"text": "  "}).status_code == 400


def test_get_presentation(client):
    r = client.get("/api/presentation")
    assert r.status_code == 200
    assert "sections:" in r.json()["text"]


def test_put_presentation_rejects_bad_yaml(client):
    r = client.put("/api/presentation", json={"text": "sections: [unclosed\n"})
    assert r.status_code == 400


def test_variant_roundtrip(client, tmp_path, monkeypatch):
    from app import promptlab
    monkeypatch.setattr(promptlab, "VARIANTS_DIR", tmp_path)

    assert client.post("/api/variants", json={
        "name": "terse", "system_prompt": "Be very terse.",
    }).status_code == 200

    assert "terse" in client.get("/api/variants").json()["variants"]
    assert client.get("/api/variants/terse").json()["system_prompt"] \
        == "Be very terse."
    assert client.delete("/api/variants/terse").json()["deleted"] is True
    assert client.get("/api/variants").json()["variants"] == []


def test_variant_name_is_validated(client, tmp_path, monkeypatch):
    """Variant names become filenames — path traversal must be rejected."""
    from app import promptlab
    monkeypatch.setattr(promptlab, "VARIANTS_DIR", tmp_path)
    r = client.post("/api/variants", json={
        "name": "../../etc/passwd", "system_prompt": "x",
    })
    assert r.status_code == 400
