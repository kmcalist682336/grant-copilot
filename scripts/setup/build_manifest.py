"""Recompute SHA-256 + sizes for every artifact in
``config/data_manifest.yaml`` and print the updated YAML to stdout.

Run this after (re-)building the data layer locally and before
pushing the artifacts to HuggingFace.  Paste the output back into
``config/data_manifest.yaml`` so the next ``hydrate_data_artifacts``
verifies against the fresh hashes.

Usage:
    python -m scripts.setup.build_manifest > new_manifest.yaml
    # then diff/replace config/data_manifest.yaml
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "config" / "data_manifest.yaml"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    with MANIFEST_PATH.open() as f:
        manifest = yaml.safe_load(f)
    any_missing = False
    for art in manifest.get("artifacts", []):
        dest = REPO_ROOT / art["dest"]
        if not dest.exists():
            print(f"# MISSING: {art['name']} at {dest}", file=sys.stderr)
            any_missing = True
            continue
        art["sha256"] = _sha256(dest)
        art["size_bytes_approx"] = dest.stat().st_size
        print(f"# {art['name']:24s} {art['sha256']}  "
              f"{art['size_bytes_approx']:>12,} B", file=sys.stderr)
    yaml.dump(manifest, sys.stdout, sort_keys=False, default_flow_style=False)
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
