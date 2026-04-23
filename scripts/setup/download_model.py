"""Download and pin Gemma model weights from HuggingFace.

This script:
  1. Authenticates with HuggingFace using HF_TOKEN env var
  2. Resolves the exact commit hash (or pins to a specific one you supply)
  3. Downloads all model files to models/gemma_weights/<model_name>/
  4. Computes sha256 hashes for every downloaded file
  5. Writes a MANIFEST.json capturing everything needed for exact reproduction
  6. Updates config/model_registry.yaml and config/inference.yaml

Prerequisites:
  pip install huggingface_hub pyyaml

Usage:
  # Download Gemma 4 E4B (default — fits in 32GB VRAM)
  HF_TOKEN=hf_xxx python -m scripts.setup.download_model

  # Download a different model
  HF_TOKEN=hf_xxx python -m scripts.setup.download_model \\
      --model google/gemma-4-26B-A4B-IT

  # Pin to a specific commit hash (recommended for strict reproducibility)
  HF_TOKEN=hf_xxx python -m scripts.setup.download_model \\
      --revision abc123def456...

  # Download to a custom directory
  HF_TOKEN=hf_xxx python -m scripts.setup.download_model \\
      --output-dir /mnt/fast_nvme/models

  # Verify an existing download without re-downloading
  python -m scripts.setup.download_model --verify-only

Notes:
  - You must accept the Gemma 4 license on HuggingFace before downloading:
    https://huggingface.co/google/gemma-4-E2B-IT
    https://huggingface.co/google/gemma-4-E4B-IT
    https://huggingface.co/google/gemma-4-26B-A4B-IT
  - Set HF_TOKEN as an environment variable, never hardcode it.
  - Once downloaded and verified, commit MANIFEST.json to version control
    (not the weights themselves).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default model — Gemma 4 E4B fits comfortably in 32GB VRAM (RTX 5090, etc.)
# Override with --model flag to download a different variant.
DEFAULT_MODEL_REPO_ID = "google/gemma-4-E4B-IT"

# Known Gemma 4 variants for reference:
#   google/gemma-4-E2B-IT   ~3GB  — edge/mobile devices
#   google/gemma-4-E4B-IT   ~8GB  — consumer GPU (default)
#   google/gemma-4-26B-A4B-IT ~52GB — requires >32GB VRAM (does NOT fit in 5090)
#   google/gemma-4-31B-IT   ~62GB — requires >64GB VRAM

# Files to skip during download
SKIP_PATTERNS = {
    ".gitattributes",
}


def _model_local_name(repo_id: str) -> str:
    """Derive a normalized filesystem-safe name from a HuggingFace repo ID.

    e.g. 'google/gemma-4-E4B-IT' → 'gemma-4-e4b-it'
    """
    return repo_id.split("/")[-1].lower()


def _model_registry_key(repo_id: str) -> str:
    """Derive a model registry key from a repo ID."""
    return f"{_model_local_name(repo_id)}-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# sha256 computation
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 of a file in chunks (memory-efficient for large files)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_file_hashes(model_dir: Path) -> dict[str, str]:
    """Compute sha256 for every file in the model directory."""
    hashes = {}
    files = sorted(model_dir.rglob("*"))
    total = sum(1 for f in files if f.is_file())
    logger.info("Computing sha256 for %d files...", total)

    for i, path in enumerate(files, 1):
        if not path.is_file():
            continue
        rel = str(path.relative_to(model_dir))
        logger.info("  [%d/%d] Hashing %s", i, total, rel)
        hashes[rel] = sha256_file(path)

    return hashes


# ---------------------------------------------------------------------------
# MANIFEST builder
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    model_repo_id: str,
    model_local_name: str,
    revision_hash: str,
    download_date: str,
    file_hashes: dict[str, str],
    model_dir: Path,
    notes: str = "",
) -> dict:
    """Build the MANIFEST.json content."""
    # Compute total size
    total_bytes = sum(
        (model_dir / rel).stat().st_size
        for rel in file_hashes
        if (model_dir / rel).exists()
    )

    return {
        "schema_version": "1.0.0",
        "model_repo_id": model_repo_id,
        "model_local_name": model_local_name,
        "model_registry_key": _model_registry_key(model_repo_id),
        "source": f"https://huggingface.co/{model_repo_id}",
        "revision_hash": revision_hash,
        "download_date": download_date,
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / 1024**3, 2),
        "file_count": len(file_hashes),
        "file_hashes": file_hashes,
        "inference": {
            "recommended_engine": "vllm",
            "recommended_engine_version": ">=0.7.0",
            "guided_decoding": "outlines (default in vLLM 0.7+, no flag needed)",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 32768,
            "tensor_parallel_size": 1,
        },
        "license": "Apache 2.0 (Gemma Terms of Use)",
        "license_url": f"https://huggingface.co/{model_repo_id}/blob/main/LICENSE",
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_manifest(model_dir: Path, manifest: dict) -> tuple[bool, list[str]]:
    """Verify that all files match their MANIFEST hashes."""
    issues = []
    file_hashes = manifest.get("file_hashes", {})

    for rel_path, expected_hash in file_hashes.items():
        abs_path = model_dir / rel_path
        if not abs_path.exists():
            issues.append(f"Missing: {rel_path}")
            continue
        actual_hash = sha256_file(abs_path)
        if actual_hash != expected_hash:
            issues.append(
                f"Hash mismatch: {rel_path}\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# model_registry.yaml update
# ---------------------------------------------------------------------------

def update_model_registry(repo_root: Path, model_dir: Path, revision_hash: str) -> None:
    """Update config/model_registry.yaml with the pinned model entry."""
    registry_path = repo_root / "config" / "model_registry.yaml"
    if not registry_path.exists():
        logger.warning("model_registry.yaml not found at %s — skipping update", registry_path)
        return

    with registry_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    # Read the MANIFEST to get the correct model ID values
    manifest_path = model_dir / "MANIFEST.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        new_active = manifest["model_local_name"]   # e.g. gemma-4-e4b-it
        new_registry_key = manifest["model_registry_key"]  # e.g. gemma-4-e4b-it-v1
    else:
        # Derive from directory name as fallback
        new_active = model_dir.name
        new_registry_key = f"{model_dir.name}-v1"

    manifest_path_str = str(manifest_path)

    # Update every role that uses a local LLM (i.e. not hosted)
    roles = registry.get("roles", {})
    updated = []
    for role_name, role_cfg in roles.items():
        if role_cfg.get("hosted"):
            continue  # skip hosted models like Vertex/Gemini
        role_cfg["active"] = new_active
        role_cfg["model_registry_key"] = new_registry_key
        role_cfg["manifest"] = manifest_path_str
        role_cfg["revision_hash"] = revision_hash
        updated.append(role_name)

    if updated:
        with registry_path.open("w", encoding="utf-8") as f:
            yaml.dump(registry, f, default_flow_style=False, allow_unicode=True)
        logger.info(
            "Updated model_registry.yaml: active='%s' for roles: %s",
            new_active, updated,
        )
    else:
        logger.info("No local-LLM roles found in model_registry.yaml — no update made")


# ---------------------------------------------------------------------------
# inference.yaml update
# ---------------------------------------------------------------------------

def update_inference_config(repo_root: Path, model_dir: Path) -> None:
    """Update config/inference.yaml to point at the downloaded model path."""
    inference_path = repo_root / "config" / "inference.yaml"
    if not inference_path.exists():
        logger.warning("inference.yaml not found — skipping update")
        return

    with inference_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Use relative path from repo root for portability
    try:
        rel_path = str(model_dir.relative_to(repo_root))
    except ValueError:
        rel_path = str(model_dir)  # absolute if outside repo

    cfg.setdefault("vllm", {})["model_path"] = rel_path

    with inference_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    logger.info("Updated inference.yaml: vllm.model_path = %s", rel_path)


# ---------------------------------------------------------------------------
# Main download flow
# ---------------------------------------------------------------------------

def download_model(
    *,
    model_repo_id: str,
    output_dir: Path,
    revision: Optional[str] = None,
    token: str,
    verify_only: bool = False,
) -> int:
    """Download the model and write the MANIFEST. Returns exit code."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Run:\n"
            "  pip install huggingface_hub --break-system-packages"
        )
        return 1

    model_local_name = _model_local_name(model_repo_id)
    model_dir = output_dir / model_local_name
    manifest_path = model_dir / "MANIFEST.json"

    # --- Verify-only mode ---
    if verify_only:
        if not manifest_path.exists():
            logger.error("No MANIFEST.json found at %s", manifest_path)
            logger.error("Run without --verify-only to download the model first.")
            return 1
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        logger.info("Verifying %d files against MANIFEST...", len(manifest["file_hashes"]))
        ok, issues = verify_manifest(model_dir, manifest)
        if ok:
            logger.info("✓ All files verified successfully")
            return 0
        else:
            logger.error("✗ Verification FAILED — %d issue(s):", len(issues))
            for issue in issues:
                logger.error("  %s", issue)
            return 1

    # --- Resolve revision hash ---
    logger.info("Connecting to HuggingFace...")
    api = HfApi(token=token)

    try:
        model_info = api.model_info(model_repo_id, revision=revision or "main")
    except Exception as e:
        logger.error("Failed to fetch model info for %s: %s", model_repo_id, e)
        logger.error(
            "Common causes:\n"
            "  1. Invalid or missing HF_TOKEN\n"
            "  2. You haven't accepted the model license at:\n"
            "     https://huggingface.co/%s\n"
            "  3. Network connectivity issue",
            model_repo_id,
        )
        return 1

    revision_hash = model_info.sha
    download_date = datetime.now(timezone.utc).isoformat()

    logger.info("Model:            %s", model_repo_id)
    logger.info("Local name:       %s", model_local_name)
    logger.info("Revision hash:    %s", revision_hash)
    if revision:
        logger.info("Pinned to requested revision: %s", revision)
    else:
        logger.info("Pinning to latest commit at time of download")
    logger.info("Download destination: %s", model_dir)

    # Warn if this is a re-download
    if model_dir.exists() and any(model_dir.iterdir()):
        if manifest_path.exists():
            with manifest_path.open() as f:
                existing = json.load(f)
            if existing.get("revision_hash") == revision_hash:
                logger.info(
                    "Model already downloaded at revision %s. "
                    "Run with --verify-only to check integrity.",
                    revision_hash[:16],
                )
                return 0
            else:
                logger.warning(
                    "Existing download at different revision (%s). "
                    "Proceeding with new download to same directory.",
                    existing.get("revision_hash", "unknown")[:16],
                )

    # --- Download ---
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting download...")
    t0 = time.monotonic()

    try:
        snapshot_download(
            repo_id=model_repo_id,
            revision=revision_hash,  # Always pin to exact hash
            local_dir=str(model_dir),
            token=token,
            ignore_patterns=list(SKIP_PATTERNS),
        )
    except Exception as e:
        logger.error("Download failed: %s", e)
        return 1

    elapsed = time.monotonic() - t0
    logger.info("Download complete in %.1f minutes", elapsed / 60)

    # --- Compute file hashes ---
    file_hashes = compute_file_hashes(model_dir)

    # --- Write MANIFEST ---
    manifest = build_manifest(
        model_repo_id=model_repo_id,
        model_local_name=model_local_name,
        revision_hash=revision_hash,
        download_date=download_date,
        file_hashes=file_hashes,
        model_dir=model_dir,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("MANIFEST.json written to %s", manifest_path)
    logger.info(
        "Model size: %.1f GB across %d files",
        manifest["total_size_gb"],
        manifest["file_count"],
    )

    # --- Verify the download ---
    logger.info("Verifying download integrity...")
    ok, issues = verify_manifest(model_dir, manifest)
    if not ok:
        logger.error("✗ Verification failed after download:")
        for issue in issues:
            logger.error("  %s", issue)
        return 1
    logger.info("✓ All files verified")

    return 0




# ---------------------------------------------------------------------------
# Config sync — update inference.yaml and model_registry.yaml from MANIFEST
# ---------------------------------------------------------------------------

def sync_configs_from_manifest(repo_root: Path, output_dir: Path) -> int:
    """Update inference.yaml and model_registry.yaml from an existing MANIFEST.

    Use this to fix config files without re-downloading the model.
    Searches output_dir for any MANIFEST.json and applies the first one found.
    """
    manifests = list(output_dir.rglob("MANIFEST.json"))
    if not manifests:
        logger.error(
            "No MANIFEST.json found under %s\n"
            "Run the download first: python -m scripts.setup.download_model",
            output_dir,
        )
        return 1

    if len(manifests) > 1:
        logger.info("Found %d MANIFEST files — processing all:", len(manifests))
        for m in manifests:
            logger.info("  %s", m)

    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        model_dir = manifest_path.parent
        revision_hash = manifest.get("revision_hash", "")
        model_local_name = manifest.get("model_local_name", model_dir.name)

        logger.info(
            "Syncing configs for model: %s (revision: %s)",
            model_local_name, revision_hash[:16] if revision_hash else "unknown",
        )
        update_model_registry(repo_root, model_dir, revision_hash)
        update_inference_config(repo_root, model_dir)

    logger.info("✓ Config sync complete")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and pin Gemma model weights from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Known Gemma 4 variants:
  google/gemma-4-E2B-IT      ~3 GB   edge/mobile
  google/gemma-4-E4B-IT      ~8 GB   consumer GPU (default, fits in 32GB VRAM)
  google/gemma-4-26B-A4B-IT  ~52 GB  requires >32GB VRAM — does NOT fit in RTX 5090
  google/gemma-4-31B-IT      ~62 GB  requires >64GB VRAM
        """,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_REPO_ID,
        help=(
            f"HuggingFace repo ID to download (default: {DEFAULT_MODEL_REPO_ID}). "
            "You must accept the model's license on HuggingFace before downloading."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Specific commit hash to download (default: latest). "
            "Specifying a hash ensures exact reproducibility. "
            "Find hashes on the model's HuggingFace 'Files and versions' page."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to store model weights "
            "(default: models/gemma_weights/ inside repo root)"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing download against MANIFEST without re-downloading",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help=(
            "Update inference.yaml and model_registry.yaml from an existing MANIFEST "
            "without downloading anything. Use this to fix config files after switching models."
        ),
    )
    parser.add_argument(
        "--no-update-configs",
        action="store_true",
        help="Skip updating inference.yaml and model_registry.yaml after download",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    model_repo_id = args.model
    model_local_name = _model_local_name(model_repo_id)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "models" / "gemma_weights"

    # --sync-only: update configs from existing MANIFEST, no download
    if args.sync_only:
        return sync_configs_from_manifest(repo_root, output_dir)

    # Get HF token — never accept it as a CLI argument
    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.verify_only:
        logger.error(
            "HF_TOKEN environment variable is not set.\n\n"
            "To set it:\n"
            "  export HF_TOKEN=hf_your_token_here\n\n"
            "Get a token at: https://huggingface.co/settings/tokens\n"
            "Then accept the model license at:\n"
            "  https://huggingface.co/%s",
            model_repo_id,
        )
        return 1

    rc = download_model(
        model_repo_id=model_repo_id,
        output_dir=output_dir,
        revision=args.revision,
        token=token,
        verify_only=args.verify_only,
    )

    # Update configs after a successful download
    model_dir = output_dir / model_local_name
    if rc == 0 and not args.verify_only and not args.no_update_configs:
        manifest_path = model_dir / "MANIFEST.json"
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
            revision_hash = manifest.get("revision_hash", "")
            update_model_registry(repo_root, model_dir, revision_hash)
            update_inference_config(repo_root, model_dir)

    if rc == 0:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Model ready at: %s", model_dir)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review MANIFEST.json and commit it to version control")
        logger.info("     (not the model weights — add models/ to .gitignore)")
        logger.info("  2. Start vLLM:")
        logger.info("     python -m vllm.entrypoints.openai.api_server \\")
        logger.info("       --model %s \\", model_dir)
        logger.info("       --port 8000 \\")
        logger.info("       --max-model-len 32768 \\")
        logger.info("       --gpu-memory-utilization 0.90")
        logger.info("  3. Run Stage 4 preflight:")
        logger.info("     python -m scripts.enrichment.run_stage4 --preflight")
        logger.info("=" * 60)

    return rc


if __name__ == "__main__":
    sys.exit(main())
