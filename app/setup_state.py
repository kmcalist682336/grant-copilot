"""Readiness checks and in-browser setup.

The app must come up even when nothing is configured. `docker compose up`
is non-interactive, so a container that dies on a missing env var leaves
the user with a traceback and no way to act on it. Instead the server
starts, reports exactly what's missing, and collects the fixable parts
through the UI.

This is not a return of the CLI setup wizard removed in 0b41455. That
wizard sat in front of a terminal the user was already at; this exists
because in the container there is no terminal to prompt from.

Three kinds of check:

  fixable here   a value we can write to .env (project id, API keys)
  host-side      something only the user can do outside the container
                 (`gcloud auth application-default login`) — detect and
                 instruct, never pretend to collect
  downloadable   the data layer, which we can fetch on request
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

# Keys the browser may write. Anything not on this list is refused, so a
# stray POST can't set arbitrary environment variables.
WRITABLE_ENV_KEYS = {"GCP_PROJECT_ID", "GCP_LOCATION", "CENSUS_API_KEY",
                     "HF_TOKEN"}

_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class Check:
    key: str
    label: str
    ok: bool
    required: bool
    detail: str = ""
    # "input" -> collectable in the browser
    # "host"  -> user must act outside the container
    # "download" -> we can fetch it
    # "none"  -> informational
    fix: str = "none"
    value_set: bool = False       # for secrets: set or not, never the value


@dataclass
class SetupReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    @property
    def ready(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [
                {
                    "key": c.key, "label": c.label, "ok": c.ok,
                    "required": c.required, "detail": c.detail,
                    "fix": c.fix, "value_set": c.value_set,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------

def load_dotenv(path: Optional[Path] = None) -> None:
    """Read .env into os.environ without clobbering real env vars."""
    p = path or ENV_PATH
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def write_env(updates: dict[str, str], path: Optional[Path] = None) -> None:
    """Upsert keys into .env, preserving comments and unrelated lines.

    Also updates os.environ so the running process sees the change
    without a restart — the whole point is to avoid one.
    """
    bad = set(updates) - WRITABLE_ENV_KEYS
    if bad:
        raise ValueError(f"Refusing to write unexpected keys: {sorted(bad)}")
    for key in updates:
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"Invalid env key: {key!r}")

    p = path or ENV_PATH
    if not p.exists() and (REPO_ROOT / ".env.example").exists():
        # Start from the template so the file keeps its explanatory
        # comments rather than becoming three bare lines.
        p.write_text((REPO_ROOT / ".env.example").read_text())

    lines = p.read_text().splitlines() if p.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        for key, val in remaining.items():
            out.append(f"{key}={val}")

    p.write_text("\n".join(out).rstrip() + "\n")
    for key, val in updates.items():
        os.environ[key] = val          # visible immediately, no restart


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

def _check_adc() -> Check:
    """Application Default Credentials.

    Cannot be fixed from the browser: ADC is a host-side gcloud login,
    and inside the container it arrives through the mounted
    ~/.config/gcloud. Detect and instruct.
    """
    try:
        import google.auth
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return Check(
            "adc", "Google Cloud credentials", True, True,
            "Application Default Credentials found.", fix="host",
        )
    except Exception as e:
        in_docker = Path("/.dockerenv").exists()
        hint = (
            "Run on your HOST machine (not in the container):\n"
            "    gcloud auth application-default login\n"
            "then restart: docker compose restart"
            if in_docker else
            "Run:  gcloud auth application-default login"
        )
        return Check(
            "adc", "Google Cloud credentials", False, True,
            f"{hint}\n\n({e})", fix="host",
        )


def _check_project() -> Check:
    """GCP project id — the thing that most often blocks a first run,
    and the easiest to collect."""
    for var in ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
        val = os.environ.get(var, "").strip()
        if val:
            return Check("gcp_project", "GCP project", True, True,
                         f"{val} (from {var})", fix="input", value_set=True)

    # ADC may carry a default project; offer it rather than making the
    # user look it up.
    suggested = ""
    try:
        import google.auth
        _, adc_project = google.auth.default()
        suggested = adc_project or ""
    except Exception:
        pass

    detail = ("Billed for Vertex AI calls (~$0.001/query). "
              "Needs the Vertex AI API enabled.")
    if suggested:
        detail += f"\nYour credentials suggest: {suggested}"
    return Check("gcp_project", "GCP project", False, True, detail,
                 fix="input")


def _check_census_key() -> Check:
    val = os.environ.get("CENSUS_API_KEY", "").strip()
    if val:
        return Check("census_key", "Census API key", True, False,
                     f"set (…{val[-4:]})", fix="input", value_set=True)
    return Check(
        "census_key", "Census API key", False, False,
        "Optional. Without one you're rate-limited to 500 requests/day. "
        "Free: https://api.census.gov/data/key_signup.html",
        fix="input",
    )


def _check_hf_token() -> Check:
    val = os.environ.get("HF_TOKEN", "").strip()
    if val:
        return Check("hf_token", "HuggingFace token", True, False,
                     f"set (…{val[-4:]})", fix="input", value_set=True)
    return Check(
        "hf_token", "HuggingFace token", False, False,
        "Only needed to download the data layer. Free, read scope: "
        "https://huggingface.co/settings/tokens",
        fix="input",
    )


def _manifest_artifacts() -> list[dict]:
    path = REPO_ROOT / "config" / "data_manifest.yaml"
    try:
        return (yaml.safe_load(path.read_text()) or {}).get("artifacts", [])
    except Exception:
        return []


def _check_data_layer() -> Check:
    """Which artifacts are on disk. Only the gazetteer and metadata index
    are truly required; the FAISS index and peer DB degrade to reduced
    functionality, which the app already handles."""
    artifacts = _manifest_artifacts()
    if not artifacts:
        return Check("data_layer", "Data layer", False, True,
                     "config/data_manifest.yaml is missing or unreadable.",
                     fix="none")

    required_dests = {"data/geo/gazetteer.db", "data/metadata/search_index.db"}
    missing, present = [], []
    missing_required = False
    for art in artifacts:
        dest = art.get("dest", "")
        name = art.get("name", dest)
        if (REPO_ROOT / dest).exists():
            present.append(name)
        else:
            missing.append(name)
            if dest in required_dests:
                missing_required = True

    if not missing:
        return Check("data_layer", "Data layer", True, True,
                     f"all {len(present)} artifacts present", fix="download")

    detail = f"Missing: {', '.join(missing)}"
    if present:
        detail += f"\nPresent: {', '.join(present)}"
    detail += "\n~8 GB total, downloaded once."
    if not missing_required:
        detail += ("\nThe required artifacts are present — the app can run "
                   "with reduced functionality.")
    return Check("data_layer", "Data layer", not missing_required,
                 True, detail, fix="download")


def report() -> SetupReport:
    """Run every check. Never raises."""
    load_dotenv()
    return SetupReport(checks=[
        _check_adc(),
        _check_project(),
        _check_data_layer(),
        _check_census_key(),
        _check_hf_token(),
    ])


# ---------------------------------------------------------------------
# Data-layer download
# ---------------------------------------------------------------------

class HydrateJob:
    """Runs hydrate_data_artifacts as a subprocess, streaming its output.

    A subprocess rather than an in-process call so a multi-GB download
    can't wedge the event loop, and so killing it is straightforward.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._proc: Optional[subprocess.Popen] = None
        self._running = False
        self._returncode: Optional[int] = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> bool:
        """Begin the download. Returns False if one is already running."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._lines = []
            self._returncode = None

        def run() -> None:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m",
                     "scripts.setup.hydrate_data_artifacts"],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ},
                )
                with self._lock:
                    self._proc = proc
                for line in proc.stdout or []:
                    line = line.rstrip("\n")
                    with self._lock:
                        self._lines.append(line)
                        # Bounded: an 8 GB download with a progress bar
                        # emits a lot of lines.
                        if len(self._lines) > 500:
                            del self._lines[:250]
                rc = proc.wait()
                with self._lock:
                    self._returncode = rc
            except Exception as e:
                with self._lock:
                    self._lines.append(f"hydrate failed to start: {e}")
                    self._returncode = 1
            finally:
                with self._lock:
                    self._running = False
                    self._proc = None

        threading.Thread(target=run, daemon=True).start()
        return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "returncode": self._returncode,
                "lines": list(self._lines[-60:]),
            }

    def cancel(self) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        proc.terminate()
        return True


HYDRATE = HydrateJob()
