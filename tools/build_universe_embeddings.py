"""Pre-compute universe-text embeddings for every (table, dataset) pair.

The universe-aware variable picker (see
``scripts/chatbot/nodes/universe_picker.py``) scores candidate tables
by cosine similarity between the user's query and each table's
``universe_plain`` text from stage 4 enrichment. Embedding this text
at query time would cost one Vertex call per candidate, so we
pre-compute once and load from disk.

Output: ``data/metadata/universe_embeddings.npz`` containing:
    keys:    array[str] of "table_id|dataset" keys
    vectors: (N, 768) float32, L2-normalized

Runtime at ~6,800 embeddings: 2-3 min with region rotation, ~$0.05.

Usage:
    python -m tools.build_universe_embeddings               # full rebuild
    python -m tools.build_universe_embeddings --dry-run     # 25 sample
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "metadata" / "universe_embeddings.npz"

EMBED_DIM = 768


def _load_universes(
    db: sqlite3.Connection,
) -> list[tuple[str, str, str]]:
    """Return one (table_id, dataset, universe_text) per distinct pair.

    When the same (table_id, dataset) appears across multiple years we
    collapse to the most recent year's text (universes rarely change
    year over year, and newer stage 4 runs produce cleaner text).

    If ``universe_plain`` is missing or very short we fall back to the
    table's ``description``. Tables with neither are skipped — the
    picker will treat them as zero universe match and fall back to the
    legacy picker's router prior.
    """
    rows = db.execute("""
        SELECT table_id, dataset, year, universe_plain, description
          FROM tables
         WHERE (universe_plain IS NOT NULL AND length(universe_plain) > 10)
            OR (description   IS NOT NULL AND length(description)    > 10)
         ORDER BY table_id, dataset, year DESC
    """).fetchall()

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for r in rows:
        key = (r["table_id"], r["dataset"])
        if key in seen:
            continue
        seen.add(key)
        text = (r["universe_plain"] or r["description"] or "").strip()
        if len(text) < 10:
            continue
        out.append((r["table_id"], r["dataset"], text))
    return out


def _embed_all(
    texts: list[str], *, project: str, locations: list[str],
    batch_size: int = 250,
) -> np.ndarray:
    """Embed every text and return an (N, EMBED_DIM) L2-normalized
    float32 matrix.

    Reuses the VertexEmbedder from tools/build_embedding_index which
    already implements region rotation on 429. One Vertex client, many
    small batches, no query-time cost.
    """
    from tools.build_embedding_index import VertexEmbedder

    embedder = VertexEmbedder(project=project, locations=locations)
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        vecs = embedder.embed_batch(chunk)
        vectors.append(vecs)
        if (i // batch_size) % 10 == 0:
            logger.info(
                "embedded %d/%d (%.1f%%)", i + len(chunk), len(texts),
                100 * (i + len(chunk)) / len(texts),
            )
    arr = np.vstack(vectors).astype(np.float32)
    # L2-normalize so a plain dot product at query time is cosine.
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype(np.float32)


def run(
    *, out_path: Path, dry_run: bool = False, limit: Optional[int] = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from scripts.chatbot.auth_check import check_auth
    with open(REPO_ROOT / "config" / "chatbot.yaml") as f:
        config = yaml.safe_load(f)
    auth = check_auth(verbose=False)
    project = auth["project_id"]
    primary = auth.get("location", "us-central1")
    fallbacks = list(
        (config.get("vertex_ai") or {}).get("fallback_locations") or []
    )
    locations: list[str] = [primary]
    for loc in fallbacks:
        if loc not in locations:
            locations.append(loc)

    db_path = REPO_ROOT / config["paths"]["metadata_db"]
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    universes = _load_universes(db)
    logger.info("universes: %d distinct (table, dataset) pairs",
                len(universes))
    if dry_run:
        universes = universes[:25]
        logger.info("dry-run: capping at %d", len(universes))
    elif limit is not None:
        universes = universes[:limit]

    keys = [f"{t}|{d}" for t, d, _ in universes]
    texts = [u for _, _, u in universes]

    t0 = time.time()
    vecs = _embed_all(texts, project=project, locations=locations)
    logger.info("embedded in %.1fs", time.time() - t0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        keys=np.array(keys, dtype=object),
        vectors=vecs.astype(np.float32),
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(
        "wrote %s  (%d keys, %.1f MB)",
        out_path, len(keys), size_mb,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Embed only the first 25 entries.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(out_path=args.out, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
