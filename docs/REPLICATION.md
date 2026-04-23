# Rebuilding the Data Layer from Scratch

This guide is for maintainers or contributors who want to
regenerate the ~4 GB pre-built data layer that
`hydrate_data_artifacts.py` normally downloads from HuggingFace.
Typical reasons:

- A new ACS vintage dropped and you want to refresh the
  embedding corpus and peer feature catalog.
- You expanded the gazetteer (e.g. added neighborhoods for
  another metro).
- You're debugging a regression in one of the offline stages.
- You want a fully reproducible build from source.

**Normal users should not do this.**  Run
`python -m scripts.setup.hydrate_data_artifacts` instead.

**Time budget**: ~20–24 hours of wall clock time on a well-
resourced workstation.  Most of it is unattended compute.

---

## Hardware prerequisites

- CPU with ≥ 8 cores recommended (FAISS build parallelizes).
- Python 3.10+.
- SpatiaLite extension for SQLite (usually installed via
  `apt-get install libsqlite3-mod-spatialite` or
  `brew install libspatialite`).
- **A CUDA GPU with ≥ 24 GB VRAM** for Stage 4 (local Gemma 4
  26B A4B IT inference).  RTX 5090 (32 GB) or A100 40 GB
  recommended.  Without a GPU, Stage 4 is effectively not
  runnable.
- ~100 GB free disk (~50 GB model weights, ~30 GB Census
  metadata, ~20 GB intermediate outputs).
- **Census API key** (~500-rps tier recommended — ask the
  Census Bureau for a higher rate limit if yours is slow).
- **Google Cloud project** with the Vertex AI Embedding API
  enabled (used in Stage 6 for ~2.4 M card embeds).

---

## Build order

The data layer has two independent tracks that can run
concurrently on separate machines:

### Track A — Census metadata + cards + values (Stages 1–8)

```
scripts/ingestion/run_stage1.py        Stage 1: walk Census API,
                                         fan out per-table metadata
                                         (~60 K tables)
scripts/ingestion/run_stage23.py       Stages 2–3: structural analysis,
                                         family / canonical / profile
                                         classification
scripts/enrichment/run_stage4.py       Stage 4: Gemma 4 26B LLM rewrite
                                         of ~1 M variable labels
                                         (~20 hours GPU)
scripts/enrichment/run_stage56.py      Stages 5–6: tagging +
                                         question generation
scripts/enrichment/run_stage7.py       Stage 7: variable finalization +
                                         coverage join
scripts/ingestion/run_stage8.py        Stage 8: peer feature value
                                         ingestion (~13 K places)
tools/build_embedding_index.py         Embed the ~2.4 M cards into
                                         FAISS HNSW32 + SQ8
                                         (~6-8 hours Vertex)
tools/build_universe_embeddings.py     Pre-compute variable-label
                                         vectors (~5 min)
tools/fetch_peer_features.py           Fetch peer feature raw values
                                         (~15 min Census API)
tools/fetch_neighborhood_features.py   Aggregate features for 248
                                         ATL neighborhoods (~5 min)
```

### Track B — Gazetteer (TIGER + OSM + ATL + …)

```
scripts/geography/build_gazetteer.py   Full build: downloads TIGER/Line
                                         shapefiles, loads OSM roads +
                                         landmarks, populates alias
                                         tables, pre-computes spatial
                                         joins.  ~1–2 hours.
```

---

## Track A walkthrough

### Stage 1: Metadata collection

```bash
# Single dataset first to verify your Census API key works
python -m scripts.ingestion.run_stage1 --dataset acs/acs5/cprofile

# Full run (all years, all allowlisted datasets)
python -m scripts.ingestion.run_stage1
```

Outputs live at `data/processed/census/<year>/<dataset>/` — one
`dataset_document.json` per dataset + one `stage1_metadata.json`
per table.  See [`docs/system_design.qmd`](system_design.qmd)
§4.2 for the fan-out logic.

### Stages 2–3

```bash
python -m scripts.ingestion.run_stage23
```

Parses variable trees, classifies tables into family +
canonical + profile roles.  ~15 minutes.

### Stage 4: Local Gemma 4 26B inference

First download the model:

```bash
export HF_TOKEN=hf_...       # with Gemma 4 license accepted
python -m scripts.setup.download_model
```

This takes ~20 minutes depending on connection.  The download
script pins a specific commit hash and writes a
`MANIFEST.json` with SHA-256s for every weight file.

Then run Stage 4:

```bash
python -m scripts.enrichment.run_stage4
```

This is the longest single step — **~20 hours on an RTX 5090**
for the full variable set.  vLLM serves the model with batching.
Checkpoint files let the process resume on interruption.

### Stages 5–7

```bash
python -m scripts.enrichment.run_stage56    # ~2-3 hours
python -m scripts.enrichment.run_stage7     # ~30 minutes
```

### Stage 8: Peer features + value ingestion

```bash
python -m scripts.ingestion.run_stage8      # ~30 min
python -m tools.fetch_peer_features         # ~15 min
python -m tools.fetch_neighborhood_features # ~5 min (ATL only)
```

### Build the FAISS index

```bash
python -m tools.build_embedding_index
```

Embeds all ~2.4 M cards via Vertex AI `text-embedding-005`.
Cost: ~$50–$100 depending on card count.  Uses region rotation
across `us-east1`, `us-west1`, `us-central1`, `us-east4`,
`us-west4` to tolerate single-region quota hits.

Expected runtime: **6–8 hours**.  The script can resume from
checkpoints if interrupted.

### Build the universe embeddings

```bash
python -m tools.build_universe_embeddings
```

~5 minutes.  Produces `data/metadata/universe_embeddings.npz`.

---

## Track B walkthrough

```bash
# Full build (downloads TIGER, OSM, ARC, ATL opendata).
python -m scripts.geography.build_gazetteer

# Skip downloads and use already-cached shapefiles.
python -m scripts.geography.build_gazetteer --no-download

# Skip expensive spatial pre-computation for quick iteration.
python -m scripts.geography.build_gazetteer --no-precompute
```

Outputs `data/geo/gazetteer.db` (~1.2 GB SpatiaLite file).

To add a new metro's curated neighborhoods, edit
`scripts/geography/load_all_sources.py` to include a new source
(see how `atl_opendata` is wired) and re-run with `--overwrite`.

---

## Publishing to HuggingFace

Once both tracks complete and all five artifacts are in place
(`data/geo/gazetteer.db`, `data/metadata/embeddings.faiss`, etc.):

```bash
# Recompute SHA-256s + sizes and print the updated manifest
python -m scripts.setup.build_manifest > config/data_manifest.yaml.new
# diff, review, and replace:
diff config/data_manifest.yaml config/data_manifest.yaml.new
mv config/data_manifest.yaml.new config/data_manifest.yaml
```

Then upload the five files (plus your `config/data_manifest.yaml`
as a versioned reference) to your HuggingFace dataset repo:

```bash
huggingface-cli upload \
    kmcalist/grant-copilot-data-layer \
    data/geo/gazetteer.db \
    gazetteer.db --repo-type dataset

huggingface-cli upload \
    kmcalist/grant-copilot-data-layer \
    data/metadata/embeddings.faiss \
    embeddings.faiss --repo-type dataset

# ... same for the other three artifacts
```

Finally, commit the updated `config/data_manifest.yaml` to
the grant-copilot Git repo so downstream users get the fresh
SHA-256s on their next `hydrate_data_artifacts` run.

---

## What each artifact looks like on disk

After a successful full rebuild:

| Artifact                          | Size     | SHA-256 (example) |
|-----------------------------------|----------|-------------------|
| `data/geo/gazetteer.db`           | ~1.2 GB  | `sha256:…`        |
| `data/metadata/embeddings.faiss`  | ~2.4 GB  | `sha256:…`        |
| `data/metadata/search_index.db`   | ~400 MB  | `sha256:…`        |
| `data/metadata/peer_features.sqlite` | ~35 MB   | `sha256:…`        |
| `data/metadata/universe_embeddings.npz` | ~20 MB | `sha256:…`        |

Verify locally with:

```bash
python -m scripts.setup.hydrate_data_artifacts --verify-only
```

---

## Common rebuild problems

### Stage 4 OOMs on a 24 GB GPU

The default Gemma 4 26B batch size assumes 32 GB VRAM.  Lower
it in `scripts/enrichment/run_stage4.py`:

```python
MAX_BATCH_SIZE = 16    # was 32
```

### Vertex embedding quota hits in build_embedding_index

The script rotates across five regions automatically.  If all
five hit daily quota, rerun tomorrow — checkpoints resume from
where you left off.

### SpatiaLite extension not found

```bash
# Debian / Ubuntu
sudo apt-get install -y libsqlite3-mod-spatialite

# macOS
brew install libspatialite
```

### Gemma 4 license not accepted

Gemma 4 is a gated model.  Go to
<https://huggingface.co/google/gemma-4-26B-A4B-IT> and click
"Agree and access repository" before running
`download_model`.
