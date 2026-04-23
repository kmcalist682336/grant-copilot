"""Interactive first-run setup wizard.

Walks the user through entering the three credentials the runtime
needs (CENSUS_API_KEY, GCP_PROJECT_ID, HF_TOKEN), writes them to
``.env`` in the repo root (preserving any other keys already
there), optionally runs ``gcloud auth application-default login``
for them, optionally hydrates the data layer, and optionally runs
the preflight.

Usage:
    python -m scripts.setup.wizard
    python -m scripts.setup.wizard --non-interactive  # dry run

Design goals:
  - Zero surprises.  Shows every file it's going to touch
    before touching it.
  - Idempotent.  Safe to re-run; skips steps already complete.
  - Keeps existing .env keys; only overwrites the ones the user
    explicitly enters this session.
  - Never displays a credential after the user types it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_IS_TTY = sys.stdout.isatty()
_G = "\033[32m" if _IS_TTY else ""
_Y = "\033[33m" if _IS_TTY else ""
_R = "\033[31m" if _IS_TTY else ""
_B = "\033[1m" if _IS_TTY else ""
_D = "\033[2m" if _IS_TTY else ""
_X = "\033[0m" if _IS_TTY else ""


def _banner() -> None:
    print(f"""
{_B}grant-copilot — interactive setup wizard{_X}

This will walk you through entering three credentials the runtime
needs, write them to {_D}.env{_X} in the repo root, and optionally run
the Google Cloud auth step + data-layer download for you.

Everything is reversible.  Re-run this wizard any time to update.

Press Enter at any prompt to skip that step.
""")


def _ask(prompt: str, default: Optional[str] = None,
        secret: bool = False, validator=None) -> Optional[str]:
    suffix = f" {_D}[{default}]{_X}" if default else ""
    while True:
        if secret:
            import getpass
            val = getpass.getpass(f"{prompt}{suffix}: ").strip()
        else:
            val = input(f"{prompt}{suffix}: ").strip()
        if not val and default is not None:
            return default
        if not val:
            return None
        if validator:
            err = validator(val)
            if err:
                print(f"  {_R}{err}{_X}")
                continue
        return val


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        v = input(f"{prompt}{suffix}: ").strip().lower()
        if not v:
            return default
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False
        print(f"  {_R}Please answer y or n.{_X}")


# --- .env I/O ---------------------------------------------------------

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            # Strip optional surrounding quotes.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            out[key] = val
    return out


def _write_env(path: Path, keys: dict[str, str],
               existing: dict[str, str]) -> list[str]:
    """Merge ``keys`` into ``path``, preserving ordering for
    existing keys and appending new ones.  Returns a list of
    (key, action) tuples for the caller to report."""
    actions: list[str] = []
    # Start with .env.example as a template if .env doesn't exist.
    if not path.exists() and ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, path)

    lines = path.read_text().splitlines() if path.exists() else []
    seen_keys: set[str] = set()
    new_lines: list[str] = []
    for raw in lines:
        m = _ENV_LINE_RE.match(raw.rstrip())
        if m:
            k = m.group(1)
            if k in keys and keys[k] is not None:
                new_lines.append(f"{k}={keys[k]}")
                seen_keys.add(k)
                if existing.get(k) != keys[k]:
                    actions.append(f"{k}: {'updated' if k in existing else 'set'}")
                continue
        new_lines.append(raw.rstrip())
    # Append any keys not already in the file.
    for k, v in keys.items():
        if v is None or k in seen_keys:
            continue
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"{k}={v}")
        actions.append(f"{k}: set")
    # Write back.
    path.write_text("\n".join(new_lines) + "\n")
    return actions


# --- Validators ------------------------------------------------------

def _val_census(v: str) -> Optional[str]:
    if len(v) < 30 or not re.fullmatch(r"[A-Za-z0-9]+", v):
        return "That doesn't look like a Census key (expect 40 hex chars)."
    return None


def _val_hf_token(v: str) -> Optional[str]:
    if not v.startswith("hf_"):
        return "HuggingFace tokens start with 'hf_'."
    if len(v) < 20:
        return "That seems too short for an HF token."
    return None


def _val_gcp_project(v: str) -> Optional[str]:
    # GCP project IDs: 6-30 chars, lowercase letters, digits, hyphens,
    # must start with a letter, can't end with a hyphen.
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", v):
        return ("GCP project IDs are 6–30 lowercase alphanumeric "
                "chars + hyphens, starting with a letter.")
    return None


# --- Steps -----------------------------------------------------------

def _census_step(existing: dict[str, str]) -> Optional[str]:
    print(f"\n{_B}Step 1/3: Census API key{_X}")
    print("Sign up at https://api.census.gov/data/key_signup.html "
          "(free, ~2 minutes).")
    cur = existing.get("CENSUS_API_KEY", "").strip()
    if cur:
        print(f"  A CENSUS_API_KEY is already set (starts with "
              f"'{cur[:4]}…').")
        if not _confirm("Replace it?", default=False):
            return None
    return _ask("Paste your Census API key (leave blank to skip)",
                secret=True, validator=_val_census)


def _gcp_step(existing: dict[str, str]) -> Optional[str]:
    print(f"\n{_B}Step 2/3: Google Cloud project ID{_X}")
    print("Create or pick a project at https://console.cloud.google.com")
    print("and enable the Vertex AI API (see docs/STARTUP_GUIDE.md §5).")
    cur = existing.get("GCP_PROJECT_ID", "").strip()
    if cur:
        print(f"  GCP_PROJECT_ID is already set to {cur!r}.")
        if not _confirm("Replace it?", default=False):
            return None
    return _ask("GCP project ID (e.g. grant-copilot-demo-458203)",
                validator=_val_gcp_project)


def _hf_step(existing: dict[str, str]) -> Optional[str]:
    print(f"\n{_B}Step 3/3: HuggingFace token{_X}")
    print("Create a read-only token at "
          "https://huggingface.co/settings/tokens (free).")
    print("Needed only for the one-time data-layer download.")
    cur = existing.get("HF_TOKEN", "").strip()
    if cur:
        print(f"  An HF_TOKEN is already set (starts with "
              f"'{cur[:6]}…').")
        if not _confirm("Replace it?", default=False):
            return None
    return _ask("Paste your HuggingFace token (hf_…)",
                secret=True, validator=_val_hf_token)


def _maybe_gcloud_auth() -> None:
    print(f"\n{_B}Optional: run `gcloud auth application-default login`{_X}")
    if shutil.which("gcloud") is None:
        print(f"  {_Y}gcloud CLI not on PATH.  Install it first "
              f"(see docs/STARTUP_GUIDE.md §5.6), then re-run "
              f"this wizard.{_X}")
        return
    if not _confirm(
        "Run gcloud auth application-default login now?  "
        "This opens a browser."
    ):
        return
    try:
        subprocess.run(["gcloud", "auth", "application-default", "login"],
                       check=False)
    except KeyboardInterrupt:
        print(f"  {_Y}aborted.{_X}")


def _maybe_hydrate() -> None:
    print(f"\n{_B}Optional: hydrate the data layer now{_X}")
    print("Downloads ~4 GB from HuggingFace (3–8 minutes on broadband).")
    if not _confirm("Run `python -m scripts.setup.hydrate_data_artifacts` now?"):
        return
    # The current env dict we've been building isn't in os.environ yet
    # until the user sources .env — but the wizard has the value in
    # memory.  Easiest path: re-read .env and set HF_TOKEN in this
    # process's env, then invoke the hydrate module via subprocess so
    # it sees it.
    env = os.environ.copy()
    env_vars = _read_env(ENV_PATH)
    if "HF_TOKEN" in env_vars:
        env["HF_TOKEN"] = env_vars["HF_TOKEN"]
    if "CENSUS_API_KEY" in env_vars:
        env["CENSUS_API_KEY"] = env_vars["CENSUS_API_KEY"]
    if "GCP_PROJECT_ID" in env_vars:
        env["GCP_PROJECT_ID"] = env_vars["GCP_PROJECT_ID"]
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.setup.hydrate_data_artifacts"],
            env=env, check=False, cwd=str(REPO_ROOT),
        )
    except KeyboardInterrupt:
        print(f"  {_Y}aborted.{_X}")


def _maybe_preflight() -> None:
    print(f"\n{_B}Optional: run preflight{_X}")
    if not _confirm("Run `python -m scripts.setup.preflight` now?"):
        return
    env = os.environ.copy()
    env_vars = _read_env(ENV_PATH)
    env.update(env_vars)
    subprocess.run(
        [sys.executable, "-m", "scripts.setup.preflight"],
        env=env, check=False, cwd=str(REPO_ROOT),
    )


def run() -> int:
    _banner()
    existing = _read_env(ENV_PATH)
    if existing:
        print(f"{_D}Found existing .env with "
              f"{len(existing)} key(s).  The wizard will offer to "
              f"update them individually.{_X}")
    else:
        print(f"{_D}No .env found yet; will create one.{_X}")

    collected: dict[str, str] = {}
    v = _census_step(existing)
    if v:
        collected["CENSUS_API_KEY"] = v
    v = _gcp_step(existing)
    if v:
        collected["GCP_PROJECT_ID"] = v
    v = _hf_step(existing)
    if v:
        collected["HF_TOKEN"] = v

    if collected:
        print(f"\n{_B}Writing to {ENV_PATH}…{_X}")
        actions = _write_env(ENV_PATH, collected, existing)
        for a in actions:
            print(f"  ✓ {a}")
    else:
        print(f"\n{_D}No changes to .env.{_X}")

    _maybe_gcloud_auth()
    _maybe_hydrate()
    _maybe_preflight()

    print(f"\n{_B}{_G}Setup wizard complete.{_X}")
    print("Start the REPL with:")
    print("    set -a; source .env; set +a")
    print("    grant-copilot --execute")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--non-interactive", action="store_true",
                    help="Print what the wizard would do and exit.")
    args = ap.parse_args()
    if args.non_interactive:
        print("Non-interactive mode: would prompt for CENSUS_API_KEY, "
              "GCP_PROJECT_ID, HF_TOKEN; offer gcloud auth; offer "
              "hydrate; offer preflight.")
        return 0
    try:
        return run()
    except KeyboardInterrupt:
        print(f"\n{_Y}Wizard cancelled by user.{_X}")
        return 130


if __name__ == "__main__":
    sys.exit(main())
