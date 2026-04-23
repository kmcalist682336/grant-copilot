# grant-copilot

**Data-grounded grant-writing assistant over U.S. Census data.**

> 🎓 **First time here?**  See
> [`docs/STARTUP_GUIDE.md`](docs/STARTUP_GUIDE.md) for a ~45-minute
> beginner-friendly walkthrough (Google Cloud project creation,
> API enablement, Census + HuggingFace signup, installation,
> data-layer download, and a guided REPL tour with example
> queries).  The sections below are the quick-reference for
> someone who's already done that once.

Takes a natural-language question — written the way a nonprofit
grant writer would phrase it ("I'm writing a food pantry grant
for Southwest Atlanta") — and produces a paragraph of
quantitative prose with comparator numbers, trend deltas, peer
references, and full variable-level source citations.  Every
number in the output is traceable to a specific Census variable,
year, and geography.

Architecture, data-layer build, and the 50-query gold-standard
smoke-test report live in
[`docs/system_design.qmd`](docs/system_design.qmd) (render to PDF
with `quarto render docs/system_design.qmd --to pdf`).

---

## Quick demo

Want to try it before reading?

```bash
# one-time
git clone https://github.com/kmcalist682336/grant-copilot.git && cd grant-copilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# guided setup — walks you through credentials, gcloud auth,
# data hydration, and preflight in one interactive session
grant-copilot-setup

# launch the REPL
set -a; source .env; set +a
grant-copilot --execute
```

Type a question at the `>` prompt and get a grant-ready
paragraph back with full citations.  Each answer also writes a
full audit-trail folder under `derived_data/data_<timestamp>/`
with per-geography CSVs and a `variables.txt` decoding every
Census variable to a plain-English description.

---

## Prerequisites

Three things every user needs.  grant-copilot is a
bring-your-own-credentials tool: **your** GCP project pays for
**your** Vertex AI calls, **your** Census API key pulls **your**
quota, **your** HuggingFace token downloads the prebuilt data
artifacts.

1. **Python 3.10+** (3.11 or 3.12 recommended).
2. **gcloud CLI** installed and authenticated.  Install from
   <https://cloud.google.com/sdk/docs/install>.  You need a GCP
   project with the **Vertex AI API enabled** and billing on —
   all Gemini 2.5 Flash calls bill to this project.
3. **A Census API key** — free, takes 2 minutes, increases your
   rate limit from 500/day to essentially unlimited.  Sign up at
   <https://api.census.gov/data/key_signup.html>.
4. **A HuggingFace account + read token** — free, needed only for
   the one-time download of the pre-built data artifacts.  Token
   setup at <https://huggingface.co/settings/tokens>.
5. **~8 GB free disk space** for the data artifacts + caches.

---

## Install

```bash
git clone https://github.com/kmcalist682336/grant-copilot.git
cd grant-copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` registers the `grant-copilot` console
script so you can launch the REPL with a single command.

---

## Credentials setup

Three credentials, three environment variables.  Put them all in
`.env` (copied from `.env.example`).  The code reads them at
startup.

### 1. Census API key

```bash
# one-time signup
open https://api.census.gov/data/key_signup.html
# you'll get a key by email; put it in .env
echo "CENSUS_API_KEY=your_key_here" >> .env
```

### 2. Google Cloud (Vertex AI)

grant-copilot uses Gemini 2.5 Flash on Vertex AI for the four
LLM-driven stages of the pipeline (extraction, plan review,
followup suggestion, synthesis).  You need your own GCP project
with the Vertex AI API enabled.

```bash
# install the gcloud CLI, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
gcloud auth application-default login
```

The last line creates Application-Default Credentials at
`~/.config/gcloud/application_default_credentials.json`, which is
what the runtime uses.  Record your project in `.env`:

```bash
echo "GCP_PROJECT_ID=your_project_id" >> .env
```

**Cost expectation**: Gemini 2.5 Flash at our cached-prompt usage
rates costs roughly **$0.002 – $0.01 per query**.  A 50-query
smoke run costs under $0.50.  We do not use Vertex context
caching at runtime (see `config/chatbot.yaml`) because it adds
per-query storage fees for demo users; the implicit prefix
cache from Google is free and still applies.

### 3. HuggingFace token

Only needed for the one-time data-layer hydration.

```bash
# Free signup
open https://huggingface.co/settings/tokens
# Create a token with "Read" scope.
echo "HF_TOKEN=hf_your_token_here" >> .env
```

---

## Hydrate the data layer

grant-copilot runs against five pre-built artifacts that would
take roughly a day to build from scratch (a 2.4 M-card embedding
job alone takes six to eight hours).  Instead, you download them
once from HuggingFace:

```bash
source .env      # load HF_TOKEN into the shell
python -m scripts.setup.hydrate_data_artifacts
```

Expected output:

```
Hydrating 5 artifact(s) from kmcalist682336/grant-copilot-data-layer @ main

  ↓ downloading gazetteer (gazetteer.db)
    ✓ SHA-256 verified
  ↓ downloading embeddings (embeddings.faiss)
    ✓ SHA-256 verified
  ↓ downloading search_index (search_index.db)
    ✓ SHA-256 verified
  ↓ downloading peer_features (peer_features.sqlite)
    ✓ SHA-256 verified
  ↓ downloading universe_embeddings (universe_embeddings.npz)
    ✓ SHA-256 verified

Done.  5 file(s) downloaded, 0 already present.
```

Total download is ~4 GB.  On a decent home connection this
takes 3–8 minutes.  The script is idempotent: re-running
verifies local SHA-256s against the manifest and skips any file
that's already in place.  Pass `--force` to re-download.

### If you'd rather rebuild from scratch

See [`docs/REPLICATION.md`](docs/REPLICATION.md).  Expect ~1 day
of wall-clock time (Census API scraping + local Gemma 4 26B
inference on a 24 GB+ GPU + Vertex AI embedding of 2.4 M cards).
The rebuild is fully scripted in `scripts/ingestion/`,
`scripts/enrichment/`, `scripts/geography/`.

---

## Verify setup

Before running queries, preflight everything:

```bash
python -m scripts.setup.preflight
```

Expected:

```
Credentials:
  ✓ CENSUS_API_KEY
  ✓ GCP_PROJECT_ID
  ✓ gcloud Application-Default Credentials
  ✓ HF_TOKEN

Data-layer artifacts:
  ✓ gazetteer              (1200.0 MB, data/geo/gazetteer.db)
  ✓ embeddings             (2400.0 MB, data/metadata/embeddings.faiss)
  ✓ search_index           (400.0 MB, data/metadata/search_index.db)
  ✓ peer_features          (35.0 MB, data/metadata/peer_features.sqlite)
  ✓ universe_embeddings    (20.0 MB, data/metadata/universe_embeddings.npz)

All checks passed.  Try: `python -m scripts.chatbot.pipeline_repl --execute`
```

---

## Run it

### Interactive REPL

```bash
grant-copilot --execute
```

The `--execute` flag runs the full pipeline on every query
(extract → resolve → plan → fetch → aggregate → synthesize →
dump).  Without it the REPL only previews extraction and doesn't
hit Census.

Example session:

```
pipeline REPL  session=…  (/quit, /help)

> What's the median household income in DeKalb County?
  [14:32:07] › Extracting intent from query
  [14:32:08] › Resolving geographies (+1.4s)
  [14:32:08] › Planning Census API calls (+0.0s)
  …

  ──── ANSWER ────

In 2024, the median household income in DeKalb County was $80,644.
This figure is roughly in line with the metro level ($92,344) and
…

  Sources:
    [acs/acs5/2024/B19013]  - median household income
        · B19013_001E  Median household income in the past 12 months
  …
```

### One-shot mode

```bash
python -m scripts.chatbot.pipeline_repl --execute \
    --query "What's the poverty rate in Fulton County?"
```

### Run the 50-query gold-standard smoke test

```bash
python -m tools.run_smoke_suite
```

Reruns the full suite documented in
[`reports/smoke_suite_report.md`](reports/smoke_suite_report.md).
Expect ~30 minutes with the built-in 10 s inter-query pause.

---

## Every answer is auditable

Each query writes a folder like
`derived_data/data_<YYYYMMDD_HHMMSS>_<microseconds>/` containing:

| File                                    | Content                                                       |
|-----------------------------------------|---------------------------------------------------------------|
| `query.txt`                             | The verbatim user question.                                  |
| `response.txt`                          | Prose + key findings + caveats + sources + peer comparisons. |
| `logs.txt`                              | Every stage event logged during this query.                  |
| `variables.txt`                         | `variable_id\tdataset\tplain-English description`.           |
| `relevant_variables_tract.csv`          | One row per (tract × variable × role) cell.                  |
| `relevant_variables_county.csv`         | County-level cells.                                          |
| `relevant_variables_msa.csv`            | MSA-level cells.                                             |
| `relevant_variables_state.csv`          | State-level cells.                                           |
| `relevant_variables_national.csv`       | US-level cells.                                              |

Tract rows include a `parent_display_name` column linking each
underlying tract back to the composite geography (neighborhood,
corridor, etc.) that triggered its fetch — so a reader can
confirm that "Midtown Atlanta" expanded to exactly the right
tract set.

---

## Tests

```bash
pytest -q
```

Expected: **429 tests pass, ~5 s**.  The test suite runs against
mocked LLM responses + synthetic fixtures — no network or data-
layer artifacts required.

---

## What's in the box

```
grant-copilot/
├── README.md                        ← you are here
├── LICENSE                          ← Apache 2.0
├── pyproject.toml                   ← project metadata + CLI entry
├── requirements.txt                 ← deps
├── .env.example                     ← credential template
│
├── config/
│   ├── chatbot.yaml                 ← runtime config
│   ├── concept_map.yaml             ← 177 curated Tier 1 concepts
│   ├── frames.yaml                  ← 55 grant-narrative frames, 419 triggers
│   ├── pipeline.yaml                ← offline-build config
│   ├── datasets_allowlist.yaml      ← Census dataset allowlist for offline builds
│   └── data_manifest.yaml           ← HuggingFace hydration manifest
│
├── prompts/v1/                      ← every LLM prompt (runtime + offline)
│
├── scripts/
│   ├── chatbot/                     ← runtime pipeline (43 modules)
│   │   ├── orchestrator.py          ← the top-level answer_query()
│   │   ├── pipeline_repl.py         ← interactive REPL
│   │   ├── nodes/                   ← per-stage LLM + deterministic nodes
│   │   └── ...
│   ├── ingestion/                   ← offline stages 1–3, 8 (metadata + values)
│   ├── enrichment/                  ← offline stages 4–7 (LLM enrichment, tagging,
│   │                                  question generation, variable finalization)
│   ├── geography/                   ← gazetteer build (TIGER + OSM + ATL + …)
│   └── setup/                       ← hydrate, build_manifest, preflight, model download
│
├── tools/                           ← one-off utilities
│   ├── trace_orchestrator.py        ← single-query non-interactive trace
│   ├── run_smoke_suite.py           ← 50-query gold-standard runner
│   ├── build_embedding_index.py     ← offline FAISS build (stages 5–6 output)
│   ├── build_universe_embeddings.py
│   ├── fetch_peer_features.py       ← Census fetcher for peer_features.sqlite
│   └── fetch_neighborhood_features.py
│
├── tests/                           ← 429 unit tests (pytest)
├── docs/
│   ├── system_design.qmd            ← full architecture doc (render to PDF via quarto)
│   └── REPLICATION.md               ← rebuild the data layer from scratch
└── reports/
    └── smoke_suite_report.md        ← per-query results from the gold-standard suite
```

---

## Troubleshooting

### "Vertex call timed out"

Gemini 2.5 Flash is usually fast (1–5 s) but occasionally cold-
starts at 15–30 s.  The client has a 60 s hard timeout.  If you
see a timeout, re-run the query; if it persists, check Vertex
quota on your GCP project.

### "HF_TOKEN not set"

The hydration script needs a HuggingFace token.  Source your
`.env`, then rerun:

```bash
source .env
python -m scripts.setup.hydrate_data_artifacts
```

### "No geographies resolved"

Some small GA places aren't in the gazetteer's TIGER snapshot.
See the "Small-place gazetteer gaps" section of §9 in
`docs/system_design.qmd` for the fix (relax the TIGER loader
threshold + rebuild the gazetteer).

### Preflight fails on "gcloud Application-Default Credentials"

```bash
gcloud auth application-default login
```

---

## License

Apache License 2.0.  See `LICENSE`.

Census data is public-domain.  The pre-built data artifacts
distributed via HuggingFace are derived from Census Bureau data
(ACS 5-year, ACS 1-year, TIGER/Line) and may be redistributed
freely.

---

## Citation

If you use grant-copilot in grant writing, research, or
demonstrations, a citation is appreciated but not required.

```
grant-copilot: a data-grounded grant-writing assistant over U.S.
  Census data.  2026.  https://github.com/kmcalist682336/grant-copilot
```
