#!/usr/bin/env python3
"""Launch the grant-copilot web app.

    python run_app.py                 # normal
    python run_app.py --no-router     # skip the 4.4 GB FAISS load, fast boot
    python run_app.py --port 8080

The server starts whether or not it's configured. Anything missing —
GCP project, API keys, the data layer — is reported in the browser and
collected there, because ``docker compose up`` has no terminal to prompt
from. ``python -m scripts.setup.preflight`` does the same checks from a
shell if you prefer.

prompts/v1/synthesizer.yaml and config/presentation.yaml are hot-reloaded
on every request — editing those never needs a restart.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv() -> None:
    """Minimal .env reader. Existing environment variables win, so
    `CENSUS_API_KEY=x python run_app.py` overrides the file."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-router", action="store_true",
                   help="Skip the FAISS index for a fast boot "
                        "(legacy tier cascade, worse concept routing).")
    p.add_argument("--no-record", action="store_true",
                   help="Don't capture LLM calls for the trace pane.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _load_dotenv()

    from app import main as app_main
    app_main.BOOT_OPTIONS["no_router"] = args.no_router
    app_main.BOOT_OPTIONS["record_llm"] = not args.no_record

    print(f"  UI:  http://{args.host}:{args.port}\n")

    import uvicorn
    uvicorn.run(app_main.app, host=args.host, port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
