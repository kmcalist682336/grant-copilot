"""Tests for deferred setup.

The app must come up unconfigured and be fixable from the browser —
`docker compose up` has no terminal to prompt from, so a container that
dies on a missing env var is a dead end.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app import setup_state


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point .env writes at a temp file so tests never touch the repo's."""
    p = tmp_path / ".env"
    monkeypatch.setattr(setup_state, "ENV_PATH", p)
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    return p


@pytest.fixture
def client(monkeypatch):
    """Unconfigured app: no context, no lifespan."""
    monkeypatch.setattr(app_main, "CTX", None)
    monkeypatch.setattr(app_main, "STARTUP_ERROR", None)
    monkeypatch.setattr(app_main, "LOADING", False)
    return TestClient(app_main.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------

def test_write_env_creates_and_sets_environ(env_file, monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    setup_state.write_env({"GCP_PROJECT_ID": "my-project"})
    assert "GCP_PROJECT_ID=my-project" in env_file.read_text()
    # Must apply to the running process — the point is avoiding a restart.
    import os
    assert os.environ["GCP_PROJECT_ID"] == "my-project"


def test_write_env_updates_in_place_preserving_comments(env_file):
    env_file.write_text(
        "# a comment\nGCP_PROJECT_ID=old\n\n# another\nCENSUS_API_KEY=keep\n")
    setup_state.write_env({"GCP_PROJECT_ID": "new"})
    text = env_file.read_text()
    assert "GCP_PROJECT_ID=new" in text
    assert "GCP_PROJECT_ID=old" not in text
    assert "# a comment" in text
    assert "CENSUS_API_KEY=keep" in text


def test_write_env_refuses_unexpected_keys(env_file):
    """A stray POST must not be able to set arbitrary env vars."""
    with pytest.raises(ValueError):
        setup_state.write_env({"PATH": "/evil"})
    with pytest.raises(ValueError):
        setup_state.write_env({"AWS_SECRET_ACCESS_KEY": "x"})


def test_load_dotenv_does_not_clobber_real_env(env_file, monkeypatch):
    env_file.write_text("GCP_PROJECT_ID=from-file\n")
    monkeypatch.setenv("GCP_PROJECT_ID", "from-shell")
    setup_state.load_dotenv(env_file)
    import os
    assert os.environ["GCP_PROJECT_ID"] == "from-shell"


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def test_report_never_raises_when_nothing_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(setup_state, "ENV_PATH", tmp_path / ".env")
    for var in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "CENSUS_API_KEY",
                "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    rep = setup_state.report()
    assert rep.ready is False
    assert {c.key for c in rep.checks} == {
        "adc", "gcp_project", "data_layer", "census_key", "hf_token"}


def test_optional_checks_do_not_block(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rep = setup_state.report()
    blocking = {c.key for c in rep.blocking}
    assert "census_key" not in blocking
    assert "hf_token" not in blocking


def test_secrets_are_never_echoed(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("CENSUS_API_KEY", "abcdef0123456789")
    rep = setup_state.report()
    census = next(c for c in rep.checks if c.key == "census_key")
    assert "abcdef0123456789" not in census.detail
    assert "abcdef" not in census.detail       # no prefix either
    assert "6789" in census.detail             # last 4 only, to confirm which
    assert census.value_set is True


def test_adc_is_host_fixable_not_input():
    """ADC is a host-side gcloud login. Offering an input for it would
    be a lie — the browser can't collect a credential file."""
    rep = setup_state.report()
    adc = next(c for c in rep.checks if c.key == "adc")
    assert adc.fix == "host"


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

def test_app_serves_while_unconfigured(client):
    assert client.get("/healthz").json()["status"] == "alive"
    assert client.get("/").status_code == 200
    assert client.get("/api/setup").status_code == 200


def test_status_reports_needs_setup(client, monkeypatch, tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    assert client.get("/api/setup").json()["status"] == "needs_setup"


def test_chat_gives_actionable_error_not_a_traceback(client, monkeypatch,
                                                     tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    r = client.post("/chat", json={"query": "median income in DeKalb"})
    assert r.status_code == 503
    assert "missing" in r.json()["detail"].lower()


def test_readyz_reports_status_when_not_ready(client):
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["detail"]["status"] in {"needs_setup", "loading", "failed"}


def test_post_setup_writes_and_rechecks(client, env_file, monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    r = client.post("/api/setup", json={"gcp_project": "proj-from-browser"})
    assert r.status_code == 200
    assert "GCP_PROJECT_ID" in r.json()["saved"]
    assert "GCP_PROJECT_ID=proj-from-browser" in env_file.read_text()

    project = next(c for c in r.json()["checks"] if c["key"] == "gcp_project")
    assert project["ok"] is True


def test_post_setup_ignores_blank_fields(client, env_file):
    r = client.post("/api/setup", json={"gcp_project": "   ",
                                        "census_key": ""})
    assert r.status_code == 200
    assert r.json()["saved"] == []


def test_initialize_refuses_while_still_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(setup_state, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    r = client.post("/api/setup/initialize", json={})
    assert r.status_code == 400
    assert "missing" in r.json()["detail"].lower()


def test_hydrate_requires_a_token(client, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    r = client.post("/api/setup/hydrate")
    assert r.status_code == 400
    assert "HF_TOKEN" in r.json()["detail"]


def test_hydrate_status_is_readable_before_any_run(client):
    s = client.get("/api/setup/hydrate").json()
    assert s["running"] is False
    assert s["lines"] == []
