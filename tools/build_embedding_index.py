"""K.3 — embed every card in the metadata DB and build a FAISS HNSW index.

Reads unembedded rows from the ``cards`` table, sends them to Vertex
``text-embedding-005`` in batches, persists the resulting 768-dim
vectors into a FAISS HNSW index using fp16 scalar quantization (cuts
RAM ~2× vs fp32 with ~1% recall hit), and stamps ``cards.embedded_at``
so re-runs skip what's done.

Storage layout (after a successful run):
    data/metadata/embeddings.faiss   — the FAISS HNSW SQ-fp16 index
                                       keyed on ``cards.rowid``.
    cards.embedded_at                 — per-row timestamp (NULL = pending).
    cards.embedding_version           — stamped with model id.

The index is HNSW (M=32, efConstruction=200) wrapped in IndexIDMap so
the FAISS rowid matches ``cards.rowid`` directly.

Resumability: crash mid-run = lose at most one batch of cards (those
embedded but not yet committed). Just re-run; the WHERE embedded_at IS
NULL clause picks up where it left off.

Cost: ~55M tokens total → ~$1.50 at $0.025/1M.
Wall-clock: ~30-60 min depending on Vertex throughput.

Usage:
    python -m tools.build_embedding_index --dry-run    # 1k probe
    python -m tools.build_embedding_index               # full run
    python -m tools.build_embedding_index --limit 5000  # small run
    python -m tools.build_embedding_index --rebuild     # ignore prior progress
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


class AllRegionsExhaustedError(RuntimeError):
    """Raised when every configured Vertex region has returned 429 and
    there's nowhere left to rotate to."""

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DB = REPO_ROOT / "data" / "metadata" / "search_index.db"
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "metadata" / "embeddings.faiss"

# Vertex text-embedding-005 limits:
#   - max 250 inputs per request
#   - max 20,000 *total* input tokens per request (across all inputs)
#   - max 2048 tokens per single input
# Our card sizes vary wildly (var_question ~75 chars, table_description
# ~899 chars), so we can't safely batch by count alone — a 250-batch of
# table_description cards blows past the 20k-token request cap. We pack
# by estimated token count instead, capped at both knobs.
DEFAULT_BATCH_SIZE = 250
MAX_TOKENS_PER_BATCH = 18_000        # leave 2k headroom under Vertex's 20k cap
TOKENS_PER_CHAR_ESTIMATE = 0.25      # conservative ~4 chars/token

EMBED_MODEL_ID = "text-embedding-005"
EMBED_DIM = 768
EMBEDDING_VERSION = f"vertex/{EMBED_MODEL_ID}"

# HNSW + SQ parameters
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# Vertex backoff
MAX_RETRIES = 10
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 300.0

# Concurrency — Vertex per-project per-region embedding QPS is generous
# (1500/min for text-embedding-005). 8-16 concurrent workers stay well
# under that and saturate IO.
DEFAULT_WORKERS = 12


# ---------------------------------------------------------------------------
# Vertex embedding client wrapper
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    """True iff the exception is a Vertex 429 / RESOURCE_EXHAUSTED.
    Other 4xx errors are permanent and must NOT trigger rotation
    (e.g. a 400 on a too-long input won't get better in another region)."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    # Some SDK versions expose the code via .status_code or stringified msg.
    if str(code) == "429":
        return True
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return True
    return False


class VertexEmbedder:
    """Embed texts via Vertex text-embedding-005, rotating across
    regions when a region's rate-limit quota is exhausted.

    Thread-safe: all worker threads share one instance. When one thread
    hits a 429, it atomically rotates the underlying client to the next
    region and retries the batch. Other threads that are simultaneously
    inside a call will either succeed (if their call landed before the
    region-level quota actually clipped them) or also see a 429 and
    retry — at which point they'll pick up whichever region is current
    at that moment, rather than forcing a second rotation.

    When every configured region has returned 429 at least once, the
    embedder raises ``AllRegionsExhaustedError`` so the caller can
    surface a clear "wait for quota reset" message rather than keep
    cycling forever.
    """

    def __init__(
        self, project: str,
        locations: list[str] | str,
        model: str = EMBED_MODEL_ID,
    ):
        from google import genai
        from google.genai import types  # noqa
        if isinstance(locations, str):
            locations = [locations]
        if not locations:
            raise ValueError("VertexEmbedder needs at least one location")
        self._genai = genai
        self._types = types
        self._project = project
        self._locations = list(locations)
        self._model = model
        self._lock = threading.Lock()
        self._idx = 0
        self._exhausted: set[int] = set()
        self._client = self._build_client()
        self.calls_made = 0
        self.tokens_in = 0

    def _build_client(self):
        loc = self._locations[self._idx]
        logger.info(
            "Vertex embedder initialised in region=%s (primary of %d)",
            loc, len(self._locations),
        )
        return self._genai.Client(
            vertexai=True, project=self._project, location=loc,
        )

    @property
    def current_location(self) -> str:
        return self._locations[self._idx]

    def _try_rotate(self, from_idx: int) -> bool:
        """Rotate to the next non-exhausted region, but only if no
        other thread has already rotated past ``from_idx`` since the
        caller last looked. Returns True if the caller should retry
        (either because we rotated or because someone else already
        rotated). Returns False when every region is exhausted."""
        with self._lock:
            # Case 1: another thread already advanced past from_idx.
            # Our client reference is now stale — retry will pick up
            # the new client.
            if self._idx != from_idx:
                return True
            # Case 2: we own the rotation. Mark current region spent
            # and find the next clean one.
            self._exhausted.add(self._idx)
            for cand in range(len(self._locations)):
                if cand not in self._exhausted:
                    old_loc = self._locations[self._idx]
                    self._idx = cand
                    new_loc = self._locations[cand]
                    logger.warning(
                        "Vertex 429 in region=%s — rotating to region=%s "
                        "(%d/%d regions exhausted so far)",
                        old_loc, new_loc,
                        len(self._exhausted), len(self._locations),
                    )
                    self._client = self._build_client()
                    return True
            # Case 3: no clean region left.
            return False

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, EMBED_DIM) float32 array.

        Retries with exponential backoff on transient errors (5xx,
        network). On 429 (quota), rotates to the next configured
        region and retries there. On permanent 4xx other than 429,
        raises immediately so the caller can fix the input.
        """
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        from google.genai import errors as genai_errors

        # Outer loop: rotate regions when the current one is exhausted.
        # `safety_rotations` caps total rotations at roughly one full
        # cycle + 2 just in case; if exceeded we raise so the caller
        # doesn't spin forever on pathological quota state.
        safety_rotations = len(self._locations) + 2
        rotations = 0

        while True:
            current_idx = self._idx
            client = self._client     # snapshot — may change under us
            backoff = INITIAL_BACKOFF_S
            last_err: Optional[Exception] = None

            for attempt in range(MAX_RETRIES):
                try:
                    resp = client.models.embed_content(
                        model=self._model,
                        contents=texts,
                        config=self._types.EmbedContentConfig(
                            output_dimensionality=EMBED_DIM,
                        ),
                    )
                    self.calls_made += 1
                    self.tokens_in += sum(len(t) for t in texts) // 4
                    vecs = np.array(
                        [e.values for e in resp.embeddings],
                        dtype=np.float32,
                    )
                    if vecs.shape != (len(texts), EMBED_DIM):
                        raise RuntimeError(
                            f"Unexpected embed shape: got {vecs.shape}, "
                            f"expected ({len(texts)}, {EMBED_DIM})"
                        )
                    return vecs
                except genai_errors.ClientError as e:
                    if _is_rate_limit_error(e):
                        # Break out of the inner-retry loop so the
                        # outer loop can rotate regions.
                        last_err = e
                        break
                    # Permanent 4xx — input issue, auth, etc. No retry.
                    raise
                except Exception as e:
                    # Transient server / network error. Retry in same
                    # region with exponential backoff.
                    last_err = e
                    if attempt + 1 == MAX_RETRIES:
                        break
                    logger.warning(
                        "embed batch failed (attempt %d, region=%s): "
                        "%s — sleeping %.1fs",
                        attempt + 1,
                        self._locations[current_idx],
                        type(e).__name__, backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)

            # Fell out of the inner loop. Either the last error was a
            # 429 (rotate) or we blew MAX_RETRIES on transient errors
            # (give up with a clear message).
            if last_err is not None and _is_rate_limit_error(last_err):
                if not self._try_rotate(from_idx=current_idx):
                    raise AllRegionsExhaustedError(
                        f"All {len(self._locations)} Vertex regions "
                        f"returned 429 — quota exhausted everywhere. "
                        f"Wait ~24h for rolling window reset or request "
                        f"a quota increase."
                    ) from last_err
                rotations += 1
                if rotations > safety_rotations:
                    raise AllRegionsExhaustedError(
                        f"Exceeded safety cap of {safety_rotations} "
                        f"rotations — aborting."
                    ) from last_err
                continue
            # Non-429 retry exhaustion.
            raise RuntimeError(
                f"embed batch failed after {MAX_RETRIES} retries"
            ) from last_err


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

def _new_index():
    """Create a fresh IndexIDMap(IndexHNSWSQ(768, M=32, fp16))."""
    import faiss
    base = faiss.IndexHNSWSQ(
        EMBED_DIM, faiss.ScalarQuantizer.QT_fp16, HNSW_M,
    )
    base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    # IndexIDMap2 supports add_with_ids and is reversible (id ↔ rowid).
    return faiss.IndexIDMap2(base)


def _is_trained(idx) -> bool:
    return idx.index.is_trained


def load_or_create_index(path: Path):
    """Load existing index from disk, or return a new one."""
    import faiss
    if path.exists():
        logger.info("Loading existing FAISS index from %s", path)
        return faiss.read_index(str(path))
    logger.info("Creating new FAISS HNSW SQ-fp16 index (M=%d, dim=%d)",
                HNSW_M, EMBED_DIM)
    return _new_index()


def save_index(idx, path: Path) -> None:
    import faiss
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(idx, str(tmp))
    tmp.replace(path)
    size_gb = path.stat().st_size / 1024**3
    logger.info("Saved FAISS index to %s (%.2f GB)", path, size_gb)


# ---------------------------------------------------------------------------
# Card streaming
# ---------------------------------------------------------------------------

def fetch_pending_cards(
    db: sqlite3.Connection, *, limit: Optional[int] = None,
) -> list[tuple[int, str]]:
    """Return [(rowid, text), ...] for cards not yet embedded.

    Order by rowid for deterministic resume — even if SQLite returns
    rows in different insertion order across runs.
    """
    sql = (
        "SELECT rowid, text FROM cards "
        "WHERE embedded_at IS NULL "
        "ORDER BY rowid"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    return [(int(r[0]), r[1]) for r in db.execute(sql, params)]


def mark_embedded(
    db: sqlite3.Connection, rowids: list[int], version: str,
) -> None:
    now = time.time()
    db.executemany(
        "UPDATE cards SET embedded_at=?, embedding_version=? WHERE rowid=?",
        [(now, version, r) for r in rowids],
    )
    db.commit()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def chunked(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def chunked_by_tokens(
    cards: list[tuple[int, str]],
    *,
    max_count: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = MAX_TOKENS_PER_BATCH,
) -> Iterator[list[tuple[int, str]]]:
    """Pack (rowid, text) tuples into batches that respect both Vertex's
    per-request input cap and total-token cap. Token cost is estimated
    from char length (Vertex doesn't expose a tokenizer for this model)."""
    batch: list[tuple[int, str]] = []
    batch_tokens = 0
    for rowid, text in cards:
        est_tok = max(1, int(len(text) * TOKENS_PER_CHAR_ESTIMATE))
        if batch and (
            len(batch) >= max_count or batch_tokens + est_tok > max_tokens
        ):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append((rowid, text))
        batch_tokens += est_tok
    if batch:
        yield batch


def run_build(
    metadata_db_path: Path,
    index_path: Path,
    *,
    project: str,
    location: str | list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    rebuild: bool = False,
    workers: int = DEFAULT_WORKERS,
    save_every_batches: int = 50,
) -> dict[str, float | int]:
    import faiss
    db = sqlite3.connect(metadata_db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")

    if rebuild:
        logger.info("--rebuild: clearing embedded_at on every card")
        db.execute("UPDATE cards SET embedded_at=NULL, embedding_version=NULL")
        db.commit()
        if index_path.exists():
            index_path.unlink()
            logger.info("Removed existing %s", index_path)

    # Accept either a single region (str) or a prioritized list.
    embedder = VertexEmbedder(project=project, locations=location)
    index = load_or_create_index(index_path)

    # Materialize pending cards. For ~2.5M rows this is ~150 MB in RAM
    # which is fine; alternative would be cursor-streaming with chunked
    # progress checkpoints.
    pending = fetch_pending_cards(db, limit=limit)
    n_pending = len(pending)
    logger.info("Pending cards: %d (limit=%s)", n_pending, limit)
    if n_pending == 0:
        logger.info("Nothing to embed.")
        save_index(index, index_path)
        return {"embedded": 0, "calls": 0, "elapsed_s": 0}

    # Train SQ on the first chunk if needed.
    if not _is_trained(index):
        train_size = min(50_000, n_pending)
        train_cards = pending[:train_size]
        logger.info("Training SQ on first %d cards…", train_size)
        train_vecs = []
        train_rowids = []
        for batch in chunked_by_tokens(
            train_cards, max_count=batch_size,
        ):
            train_vecs.append(embedder.embed_batch([t for _, t in batch]))
            train_rowids.extend(r for r, _ in batch)
        train_arr = np.vstack(train_vecs).astype(np.float32)
        index.train(train_arr)
        # Add the training vectors to the index now (so we don't waste them)
        index.add_with_ids(
            train_arr, np.array(train_rowids, dtype=np.int64),
        )
        mark_embedded(db, train_rowids, EMBEDDING_VERSION)
        logger.info("SQ trained + initial %d cards added", len(train_rowids))
        pending = pending[train_size:]
        save_index(index, index_path)

    # Stream-embed remaining cards CONCURRENTLY.
    # Worker threads each call Vertex; the main thread serializes
    # FAISS writes + DB updates (FAISS add_with_ids isn't thread-safe).
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(pending), desc="embed", unit="card",
                   mininterval=2.0)
    except ImportError:
        bar = None

    t_start = time.time()
    embedded_count = 0
    cost_per_token = 0.025 / 1_000_000
    n_workers = workers

    def _embed_batch_worker(
        batch: list[tuple[int, str]],
    ) -> tuple[list[int], np.ndarray]:
        """Thread-worker: takes a batch, returns (rowids, vecs)."""
        rowids = [r for r, _ in batch]
        texts = [t for _, t in batch]
        vecs = embedder.embed_batch(texts)
        return rowids, vecs

    batches = list(chunked_by_tokens(pending, max_count=batch_size))
    save_every_n_completed = max(50, n_workers * 4)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches up front; as_completed orders by finish time.
        futures: list[Future] = [
            executor.submit(_embed_batch_worker, b) for b in batches
        ]
        completed_since_save = 0
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            try:
                rowids, vecs = fut.result()
            except Exception as e:
                logger.error("Embed batch failed: %s", e)
                # Cancel remaining work and bail out so we can resume later
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            ids_arr = np.array(rowids, dtype=np.int64)
            index.add_with_ids(vecs.astype(np.float32), ids_arr)
            mark_embedded(db, rowids, EMBEDDING_VERSION)
            embedded_count += len(rowids)
            completed_since_save += 1
            if bar:
                est_cost = embedder.tokens_in * cost_per_token
                bar.set_postfix({
                    "calls": embedder.calls_made,
                    "tok": embedder.tokens_in,
                    "cost$": f"{est_cost:.3f}",
                    "wrk": n_workers,
                })
                bar.update(len(rowids))
            if completed_since_save >= save_every_n_completed:
                save_index(index, index_path)
                completed_since_save = 0

    if bar:
        bar.close()
    save_index(index, index_path)
    db.close()

    elapsed = time.time() - t_start
    return {
        "embedded": embedded_count,
        "calls": embedder.calls_made,
        "tokens_estimate": embedder.tokens_in,
        "cost_usd_estimate": round(
            embedder.tokens_in * cost_per_token, 4,
        ),
        "elapsed_s": round(elapsed, 1),
        "index_path": str(index_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--metadata-db", default=str(DEFAULT_METADATA_DB),
        help=f"(default: {DEFAULT_METADATA_DB})",
    )
    ap.add_argument(
        "--index-path", default=str(DEFAULT_INDEX_PATH),
        help=f"(default: {DEFAULT_INDEX_PATH})",
    )
    ap.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Texts per Vertex API call (default {DEFAULT_BATCH_SIZE}).",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Embed at most N cards (smoke test).",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="Clear all embedded_at flags + delete existing index, "
             "then build from scratch.",
    )
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent Vertex API workers (default {DEFAULT_WORKERS}).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Equivalent to --limit 1000 (verifies pipeline end-to-end "
             "with minimal cost).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Auth check up front so we fail fast.
    from scripts.chatbot.auth_check import AuthCheckError, check_auth
    try:
        auth = check_auth(verbose=False)
    except AuthCheckError as e:
        logger.error("Auth failed: %s", e)
        return 1
    project = auth["project_id"]
    primary_location = auth.get("location", "us-central1")

    # Build the region rotation list: primary first, then any configured
    # fallbacks (deduped, primary guaranteed not to appear twice). The
    # fallback list lets one embed run continue across multiple regions
    # when any one region's rate-limit quota is exhausted.
    import yaml as _yaml
    _cfg_path = (
        Path(__file__).resolve().parents[1] / "config" / "chatbot.yaml"
    )
    fallback_locations: list[str] = []
    try:
        with _cfg_path.open() as _f:
            _cfg = _yaml.safe_load(_f) or {}
        fallback_locations = list(
            (_cfg.get("vertex_ai") or {}).get("fallback_locations") or []
        )
    except OSError:
        logger.debug("no chatbot.yaml found for fallback_locations")
    locations: list[str] = [primary_location]
    for loc in fallback_locations:
        if loc not in locations:
            locations.append(loc)
    logger.info(
        "Vertex project=%s locations=%s (primary=%s, %d fallbacks)",
        project, locations, primary_location, len(locations) - 1,
    )

    if not Path(args.metadata_db).exists():
        logger.error("Metadata DB missing: %s", args.metadata_db)
        return 1

    limit = args.limit
    if args.dry_run and limit is None:
        limit = 1000
        logger.info("--dry-run: capping at 1000 cards")

    stats = run_build(
        Path(args.metadata_db),
        Path(args.index_path),
        project=project, location=locations,
        batch_size=args.batch_size,
        limit=limit, rebuild=args.rebuild,
        workers=args.workers,
    )

    print("\nEmbed run summary:")
    for k, v in stats.items():
        print(f"  {k:25s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
