"""Hydrate the pre-built data-layer artifacts from HuggingFace.

The runtime pipeline needs five files under ``data/``:

    data/geo/gazetteer.db
    data/metadata/embeddings.faiss
    data/metadata/search_index.db
    data/metadata/peer_features.sqlite
    data/metadata/universe_embeddings.npz

Rather than rebuilding them from scratch (which takes roughly a
day of Census scraping + LLM inference + embedding work; see
``docs/REPLICATION.md``), you can download the pre-built set from
a HuggingFace dataset repo in about five minutes.

Prerequisites:
  1. Free HuggingFace account: https://huggingface.co
  2. Read-only access token:
         https://huggingface.co/settings/tokens
     Export it in your shell:
         export HF_TOKEN=hf_xxxxx
  3. The dataset repo in ``config/data_manifest.yaml`` must be
     readable with your token.  If the publisher made the repo
     private, ask them to add you as a member or make it public.

Usage:
    python -m scripts.setup.hydrate_data_artifacts
    python -m scripts.setup.hydrate_data_artifacts --force   # re-download
    python -m scripts.setup.hydrate_data_artifacts --verify-only
    python -m scripts.setup.hydrate_data_artifacts --only gazetteer

Design:
  - We read the manifest (config/data_manifest.yaml) to learn the
    HF repo, the per-file SHA-256, and the local destination path.
  - We use ``huggingface_hub.hf_hub_download`` with resume support
    so an interrupted large download resumes where it left off.
  - Every file's SHA-256 is verified against the manifest.  A
    mismatch aborts rather than silently using corrupt data.
  - Idempotent: running twice with everything already hydrated is
    a no-op.  ``--force`` re-downloads regardless.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "config" / "data_manifest.yaml"

# Plain ANSI colors — no external dep.  Disabled if stdout isn't a TTY.
_IS_TTY = sys.stdout.isatty()
_GREEN = "\033[32m" if _IS_TTY else ""
_YELLOW = "\033[33m" if _IS_TTY else ""
_RED = "\033[31m" if _IS_TTY else ""
_DIM = "\033[2m" if _IS_TTY else ""
_RESET = "\033[0m" if _IS_TTY else ""


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        _die(f"Manifest missing: {MANIFEST_PATH}")
    with MANIFEST_PATH.open() as f:
        return yaml.safe_load(f)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def _die(msg: str) -> None:
    print(f"{_RED}error:{_RESET} {msg}", file=sys.stderr)
    sys.exit(2)


def _warn(msg: str) -> None:
    print(f"{_YELLOW}warning:{_RESET} {msg}", file=sys.stderr)


def _require_token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        _die(
            "HF_TOKEN not set.  Create a free HuggingFace account + "
            "read token at https://huggingface.co/settings/tokens, "
            "then export HF_TOKEN=hf_..."
        )
    return tok


def _download_one(
    hf_api,
    repo: str, revision: str, repo_type: str,
    art: dict[str, Any], force: bool,
) -> tuple[Path, bool]:
    """Download one artifact into its final resting place.

    Returns (final_path, downloaded_flag).  When the file already
    exists and its SHA matches the manifest, we skip the network
    call entirely and return downloaded_flag=False.
    """
    name = art["name"]
    dest = REPO_ROOT / art["dest"]
    expected_sha = art.get("sha256")

    # Fast-path: if the destination already exists AND its SHA
    # matches the manifest, there's nothing to do.
    if dest.exists() and not force:
        if expected_sha:
            actual = _sha256(dest)
            if actual == expected_sha:
                print(f"  {_GREEN}✓{_RESET} {name} already hydrated "
                      f"({_human_bytes(dest.stat().st_size)})")
                return dest, False
            _warn(
                f"{name} exists at {dest} but SHA doesn't match "
                f"manifest — re-downloading"
            )
        else:
            # No SHA to verify against — trust it.
            print(f"  {_GREEN}✓{_RESET} {name} already present ({_human_bytes(dest.stat().st_size)}) "
                  f"{_DIM}(manifest has no SHA yet, skipping verify){_RESET}")
            return dest, False

    print(f"  ↓ downloading {name} ({art.get('hf_path')})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached_path = hf_api.hf_hub_download(
        repo_id=repo,
        filename=art["hf_path"],
        revision=revision,
        repo_type=repo_type,
        local_dir=str(dest.parent),
        resume_download=True,
    )
    cached = Path(cached_path)
    # huggingface_hub 0.24+ honors local_dir and places the file
    # at local_dir/<hf_path>.  For single-file artifacts this is
    # usually what we want, but paper over edge cases by moving
    # the result into place if needed.
    if cached.resolve() != dest.resolve():
        shutil.move(str(cached), str(dest))

    # Verify SHA if manifest supplies one.
    if expected_sha:
        actual = _sha256(dest)
        if actual != expected_sha:
            _die(
                f"{name}: SHA-256 mismatch after download\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual}\n"
                f"  file:     {dest}"
            )
        print(f"    {_GREEN}✓{_RESET} SHA-256 verified")

    return dest, True


def hydrate(only: Optional[list[str]] = None, force: bool = False,
            verify_only: bool = False) -> int:
    manifest = _load_manifest()
    repo = manifest["hf_repo"]
    revision = manifest.get("hf_revision", "main")
    repo_type = manifest.get("hf_repo_type", "dataset")
    artifacts = manifest.get("artifacts", [])

    if repo.startswith("REPLACE_ME"):
        _die(
            f"hf_repo in {MANIFEST_PATH} is still the placeholder "
            "'REPLACE_ME/...'.  Edit config/data_manifest.yaml to "
            "point at the actual HuggingFace dataset repo that "
            "hosts the pre-built artifacts."
        )

    # Optionally filter to just named artifacts.
    if only:
        artifacts = [a for a in artifacts if a["name"] in only]
        unknown = set(only) - {a["name"] for a in manifest["artifacts"]}
        if unknown:
            _die(f"Unknown artifact names: {sorted(unknown)}")

    if verify_only:
        print(f"Verifying {len(artifacts)} artifact(s) (no network)…")
        ok = True
        for art in artifacts:
            dest = REPO_ROOT / art["dest"]
            if not dest.exists():
                print(f"  {_RED}✗{_RESET} {art['name']} missing at {dest}")
                ok = False
                continue
            expected = art.get("sha256")
            if expected:
                actual = _sha256(dest)
                if actual == expected:
                    print(f"  {_GREEN}✓{_RESET} {art['name']}")
                else:
                    print(f"  {_RED}✗{_RESET} {art['name']} SHA mismatch")
                    ok = False
            else:
                print(f"  {_DIM}?{_RESET} {art['name']} present "
                      f"({_human_bytes(dest.stat().st_size)}) — no SHA in manifest")
        return 0 if ok else 1

    _require_token()

    # Lazy-import so the unit tests / help text don't require
    # huggingface_hub.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _die(
            "huggingface_hub not installed.  Run "
            "`pip install -r requirements.txt`."
        )

    class _Adapter:
        """Tiny wrapper so the download function can be mocked in tests."""
        def hf_hub_download(self, **kw):
            return hf_hub_download(**kw)

    hf_api = _Adapter()

    print(f"Hydrating {len(artifacts)} artifact(s) from "
          f"{_GREEN}{repo}{_RESET} @ {revision}\n")
    total_downloaded = 0
    for art in artifacts:
        _, downloaded = _download_one(
            hf_api, repo, revision, repo_type, art, force=force,
        )
        if downloaded:
            total_downloaded += 1

    print(f"\n{_GREEN}Done.{_RESET} "
          f"{total_downloaded} file(s) downloaded, "
          f"{len(artifacts) - total_downloaded} already present.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="Re-download every artifact even if SHA matches.")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip downloads; only check local SHA-256s.")
    ap.add_argument("--only", nargs="+",
                    help="Only hydrate the named artifact(s) "
                         "(e.g. --only gazetteer peer_features).")
    args = ap.parse_args()
    return hydrate(
        only=args.only, force=args.force, verify_only=args.verify_only,
    )


if __name__ == "__main__":
    sys.exit(main())
