"""Pre-flight check: verify that the three required credentials
and the five hydrated data-layer artifacts are in place before
you try to run the chatbot.

Usage:
    python -m scripts.setup.preflight
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_IS_TTY = sys.stdout.isatty()
_G = "\033[32m" if _IS_TTY else ""
_Y = "\033[33m" if _IS_TTY else ""
_R = "\033[31m" if _IS_TTY else ""
_X = "\033[0m" if _IS_TTY else ""


def main() -> int:
    ok = True

    def check(label, predicate, ok_detail="", fail_detail=""):
        nonlocal ok
        if predicate:
            print(f"  {_G}✓{_X} {label}" + (f"  {ok_detail}" if ok_detail else ""))
        else:
            print(f"  {_R}✗{_X} {label}" + (f"  {fail_detail}" if fail_detail else ""))
            ok = False

    print(f"{_G}grant-copilot preflight{_X}\n")

    # --- Credentials ---
    print("Credentials:")
    check("CENSUS_API_KEY",
          bool(os.environ.get("CENSUS_API_KEY")),
          fail_detail="Sign up at https://api.census.gov/data/key_signup.html "
                      "and export CENSUS_API_KEY")
    check("GCP_PROJECT_ID",
          bool(os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")),
          fail_detail="Set GCP_PROJECT_ID to a project with the Vertex AI API enabled")
    # Application-Default Credentials
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    check("gcloud Application-Default Credentials",
          adc_path.exists(),
          fail_detail="Run `gcloud auth application-default login`")

    # HF token only required if we're about to hydrate — but still
    # useful to verify it's set so the user doesn't get confused.
    check("HF_TOKEN",
          bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
          fail_detail="Needed for hydrating artifacts.  Create a read token "
                      "at https://huggingface.co/settings/tokens")

    # --- Data-layer artifacts ---
    print("\nData-layer artifacts:")
    manifest_path = REPO_ROOT / "config" / "data_manifest.yaml"
    if not manifest_path.exists():
        print(f"  {_R}✗{_X} manifest missing: {manifest_path}")
        return 2
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    for art in manifest.get("artifacts", []):
        dest = REPO_ROOT / art["dest"]
        size = dest.stat().st_size if dest.exists() else 0
        size_str = f"{size/1e6:.1f} MB" if size else "missing"
        check(f"{art['name']:22s}",
              dest.exists() and size > 0,
              ok_detail=f"({size_str}, {dest.relative_to(REPO_ROOT)})",
              fail_detail=f"hydrate with `python -m scripts.setup.hydrate_data_artifacts`")

    print()
    if ok:
        print(f"{_G}All checks passed.{_X}  Try: "
              f"`python -m scripts.chatbot.pipeline_repl --execute`")
        return 0
    print(f"{_Y}Some checks failed.{_X}  Fix the items above, then "
          f"re-run this script.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
