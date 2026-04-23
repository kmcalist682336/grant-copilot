# grant-copilot: Startup Guide

This is the beginner-friendly walkthrough.  If you've never
touched Google Cloud, never created a Census API key, and never
worked with HuggingFace, this guide will still get you from zero
to a running chatbot in about an hour of attended time (most of
which is filling in web forms and waiting for a download).

If you already know all of that, just read the README's quick
demo and skip this document.

**Total time budget**
- Active work: ~20 minutes (mostly clicking through signup forms).
- Waiting: ~10–15 minutes for the data-layer download (~8 GB).
- **Total wall clock**: ~45 minutes.

---

## Table of contents

1. [What you're installing](#1-what-youre-installing)
2. [System prerequisites](#2-system-prerequisites)
3. [Clone and install](#3-clone-and-install)
4. [Sign up for a Census API key](#4-sign-up-for-a-census-api-key)
5. [Set up Google Cloud (the one that takes the longest)](#5-set-up-google-cloud)
6. [Sign up for HuggingFace and get a token](#6-sign-up-for-huggingface-and-get-a-token)
7. [Store your credentials in `.env`](#7-store-your-credentials-in-env)
8. [Download the pre-built data layer](#8-download-the-pre-built-data-layer)
9. [Verify everything with preflight](#9-verify-everything-with-preflight)
10. [Launch the REPL and try example queries](#10-launch-the-repl-and-try-example-queries)
11. [How the REPL works (commands, flags, tips)](#11-how-the-repl-works)
12. [What to do when something fails](#12-troubleshooting)

---

## 1. What you're installing

grant-copilot is a Python tool that answers natural-language
questions about U.S. demographic / economic / housing data using
the U.S. Census Bureau's public API.  It runs on your laptop but
makes outbound calls to two cloud services:

- **Google Cloud Vertex AI** — for the Gemini 2.5 Flash language
  model that does the narrative writing and a few intermediate
  reasoning steps.  You'll bring your own Google Cloud project;
  all charges bill to you.  Expect **$0.01 or less per query**
  under normal use.

- **Census Bureau API** — for the actual data.  Free, but rate-
  limited to 500 calls/day without a key.  With a key (also
  free, takes 2 minutes to get), effectively unlimited.

One more thing you'll need is a **HuggingFace** account — only
for the one-time download of the pre-built data artifacts.
Nothing after that requires HuggingFace.

**Why three separate accounts?**  Each service is owned by a
different org (Google, the U.S. Census Bureau, HuggingFace), so
each has its own API key.  The alternative would be for the
project maintainer to proxy everything, which means the
maintainer pays for your usage and knows every question you ask.
This way, grant-copilot is fully self-contained on your
machine — nobody sees your queries but you.

---

## 2. System prerequisites

You need:

- **A computer** running Linux, macOS, or Windows with WSL2.
- **Python 3.10 or newer** — 3.11 or 3.12 recommended.
  - Check with `python3 --version`.
  - If missing, install from <https://www.python.org/downloads/>
    or use your OS package manager.
- **Git**, for cloning the repo.  `git --version` to check.
- **~12 GB free disk space** for the data artifacts + caches (the artifacts themselves are ~8 GB).
- **A stable broadband connection** for the one-time data
  download (~8 GB, usually 5–15 minutes).

That's it.  No GPUs, no local models, no Docker.

---

## 3. Clone and install

Install system-level dependencies first (SpatiaLite for the
gazetteer, SWIG to let `faiss-cpu` build cleanly on macOS).
Skipping this step is the most common fresh-install failure.

### 3.1. System packages

**macOS (Homebrew)**

```bash
brew install libspatialite swig
```

**Linux (Debian / Ubuntu)**

```bash
sudo apt-get update
sudo apt-get install -y libsqlite3-mod-spatialite swig build-essential
```

**Linux (Fedora / RHEL)**

```bash
sudo dnf install -y libspatialite-devel swig
```

Why each one:

- `libspatialite` / `libsqlite3-mod-spatialite` is a SQLite
  extension our gazetteer loader calls via
  `conn.load_extension('mod_spatialite')`.  Without it, every
  geography lookup fails at startup with
  `OperationalError: unable to load extension`.
- `swig` is a build-time dependency for `faiss-cpu` on macOS
  when pip falls through to a source build.  On most Linux
  systems a wheel is available and `swig` isn't strictly needed,
  but installing it up-front avoids the surprise.

### 3.2. Clone + Python environment

```bash
# pick any folder you like; we'll use ~/code here
cd ~/code
git clone https://github.com/kmcalist682336/grant-copilot.git
cd grant-copilot

# create an isolated Python environment for this project
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` registers a command called `grant-copilot` on
your `PATH` that launches the REPL directly.  You'll use it in
step 10.

### 3.3. If `faiss-cpu` refuses to install (macOS)

If `pip install -r requirements.txt` errors out on `faiss-cpu`
(typical on a fresh macOS install where pip wants to build from
source and can't), force the pre-built wheel:

```bash
pip install --only-binary=:all: faiss-cpu
pip install -r requirements.txt
```

Then rerun `pip install -e .`.

### Sanity check

Run the unit tests — no network or data artifacts needed:

```bash
pytest -q
```

Expected: **429 tests pass in ~5 seconds**.  If you see failures
here, something is wrong with your Python environment; stop and
fix it before proceeding.

---

## 4. Sign up for a Census API key

The fastest of the three signups.

1. Go to <https://api.census.gov/data/key_signup.html>.
2. Fill in your name, organization (can be personal), and email
   address.
3. Click "Request Key".
4. Within a couple minutes you'll get an email containing the
   key — a 40-character hexadecimal string like
   `abc123def456abc123def456abc123def456abcd`.
5. Save the key somewhere safe.  You'll paste it into `.env` in
   step 7.

**Do not** share your Census API key publicly.  It isn't
catastrophic if leaked — it just means someone else could use up
your daily quota — but treat it like any other credential.

---

## 5. Set up Google Cloud

This is the step that takes the longest, and it's the one new
users most often get stuck on.  Follow along.

grant-copilot uses **Vertex AI**, Google's managed ML platform,
to call the Gemini 2.5 Flash model.  You'll create a Google
Cloud project, enable the API, set up billing (free tier is
generous but Vertex itself isn't free), and authenticate your
laptop.

### 5.1. Create (or pick) a Google account

If you already have a Google account (Gmail, etc.), you can use
that one.  Otherwise, create one at <https://accounts.google.com>.

### 5.2. Open the Google Cloud Console

1. Go to <https://console.cloud.google.com/>.
2. Accept the terms of service if prompted.  First-time visitors
   get a $300 free-trial credit — accept it; Vertex calls for
   grant-copilot use a tiny fraction of that.

### 5.3. Create a new project

1. Click the project dropdown at the top of the page (next to
   "Google Cloud").
2. Click "**NEW PROJECT**".
3. Give it a name like `grant-copilot-demo`.  The project ID is
   auto-generated (e.g. `grant-copilot-demo-458203`).  **Write
   down the project ID** — you'll paste it into `.env` shortly.
4. Click "**CREATE**" and wait ~30 seconds for the project to
   provision.
5. After creation, make sure that project is selected in the
   dropdown at the top.

### 5.4. Enable billing

Vertex AI requires a billing account, even on the free-trial
credits.

1. In the left-hand menu, click **Billing** (you may need to
   search "billing" in the top search bar).
2. If you don't have a billing account, click "**MANAGE BILLING
   ACCOUNTS**" → "**CREATE ACCOUNT**" and follow the steps.  You
   need a credit card, but the free-trial credit covers many
   thousands of grant-copilot queries before anything charges
   your card.
3. Link the billing account to your project.

**Expected cost for demo use**: under $1 for a 50-query smoke
test.  Typical per-query cost is **$0.002 – $0.01**.

### 5.5. Enable the Vertex AI API

1. In the search bar at the top, type "Vertex AI API" and click
   the top result.
2. Click "**ENABLE**".  It takes ~30 seconds.
3. (Optional but nice to have) in the same console, enable the
   **Cloud Resource Manager API** — some authentication flows
   use it.

You can do both with the CLI too (after installing gcloud in
the next step):

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

### 5.6. Install the gcloud CLI

gcloud is Google's command-line tool.  grant-copilot doesn't
call it at runtime, but you need it **once** to authenticate
your laptop.

**macOS (Homebrew)**

```bash
brew install --cask google-cloud-sdk
```

Alternative: <https://cloud.google.com/sdk/docs/install-sdk#mac>

**Linux (Debian / Ubuntu)** — official Google apt repository:

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
  https://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update
sudo apt-get install -y google-cloud-cli
```

**Linux (Fedora / RHEL)**

```bash
sudo tee /etc/yum.repos.d/google-cloud-sdk.repo <<EOF
[google-cloud-cli]
name=Google Cloud CLI
baseurl=https://packages.cloud.google.com/yum/repos/cloud-sdk-el9-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=0
gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF
sudo dnf install -y google-cloud-cli
```

**Windows**: download the installer from
<https://cloud.google.com/sdk/docs/install-sdk#windows> and
open PowerShell to use it after install.

Verify the install:

```bash
gcloud --version
```

You should see several lines listing the SDK version and
components.

### 5.7. Authenticate

Two commands.  The first one logs your gcloud CLI in; the
second creates credentials grant-copilot can read.

```bash
# 1. One-time browser login.  Opens a browser, sign in with the
#    Google account you used in 5.1.
gcloud auth login

# 2. Set your default project so commands don't need a --project flag.
gcloud config set project YOUR_PROJECT_ID

# 3. Create application-default credentials.  This is what the
#    Python SDK reads at runtime.  Opens a browser again.
gcloud auth application-default login
```

The third command writes a credentials file at
`~/.config/gcloud/application_default_credentials.json` (on
Linux/macOS) or `%APPDATA%\gcloud\application_default_credentials.json`
(on Windows).  grant-copilot reads that file automatically.

### 5.8. Sanity-check

```bash
gcloud ai endpoints list --region=us-east1
```

You should see an empty list or existing endpoints — either is
fine.  If you see an error about the API not being enabled, go
back to step 5.5.

---

## 6. Sign up for HuggingFace and get a token

HuggingFace is a free hosting service for AI datasets and
models.  We use it to distribute the pre-built data artifacts
(~8 GB) that grant-copilot needs at runtime.

1. Go to <https://huggingface.co/join> and create an account
   (free; email + password).
2. Verify your email.
3. Go to <https://huggingface.co/settings/tokens>.
4. Click "**New token**".
5. Name it something like `grant-copilot-download`.
6. Set the role to **Read** (write access is not needed).
7. Click "**Generate a token**".
8. Copy the token immediately — it looks like
   `hf_abcdef1234567890…`.  You won't be able to see it again
   once you leave the page.
9. Save it somewhere safe.

---

## 7. Store your credentials in `.env`

You now have three secrets:

- A Census API key.
- A Google Cloud project ID.
- A HuggingFace token.

Store them in `.env` in the project root.  This file is
**gitignored** — it will never be committed.

```bash
cp .env.example .env

# open .env in your editor and fill in:
#   CENSUS_API_KEY=abc123...
#   GCP_PROJECT_ID=grant-copilot-demo-458203
#   HF_TOKEN=hf_abcdef...
```

To load the variables into your current shell so commands that
need them can see them:

```bash
set -a; source .env; set +a
```

You'll want to run that at the start of every new terminal
session, or add it to your shell's startup (`~/.bashrc`,
`~/.zshrc`).  Alternatively, many IDEs and shells (e.g.,
direnv) load `.env` automatically.

---

## 8. Download the pre-built data layer

With `HF_TOKEN` in your shell, one command does it:

```bash
python -m scripts.setup.hydrate_data_artifacts
```

Expect:

```
Hydrating 5 artifact(s) from kmcalist682336/grant-copilot-data-layer @ main

  ↓ downloading gazetteer (gazetteer.db)
    ✓ SHA-256 verified
  ↓ downloading embeddings (embeddings.faiss)
    ✓ SHA-256 verified
  …

Done.  5 file(s) downloaded, 0 already present.
```

This downloads five files totaling ~8 GB:

| File                       | Size      | What it is                         |
|----------------------------|-----------|------------------------------------|
| `gazetteer.db`             | ~286 MB   | SpatiaLite geographies             |
| `embeddings.faiss`         | ~4.4 GB   | 2.4 M-card semantic search index   |
| `search_index.db`          | ~3.4 GB   | Census metadata + variable labels  |
| `peer_features.sqlite`     | ~162 MB   | Peer-city feature catalog          |
| `universe_embeddings.npz`  | ~21 MB    | Pre-computed variable-label vecs   |

The script is resumable — if your download is interrupted, just
rerun and it picks up.  It's also idempotent: re-running with
everything already downloaded is a no-op.

---

## 9. Verify everything with preflight

```bash
python -m scripts.setup.preflight
```

Expected:

```
grant-copilot preflight

Credentials:
  ✓ CENSUS_API_KEY
  ✓ GCP_PROJECT_ID
  ✓ gcloud Application-Default Credentials
  ✓ HF_TOKEN

Data-layer artifacts:
  ✓ gazetteer              (286.1 MB, data/geo/gazetteer.db)
  ✓ embeddings             (4377.6 MB, data/metadata/embeddings.faiss)
  ✓ search_index           (3380.2 MB, data/metadata/search_index.db)
  ✓ peer_features          (162.4 MB, data/metadata/peer_features.sqlite)
  ✓ universe_embeddings    (20.9 MB, data/metadata/universe_embeddings.npz)

All checks passed.
```

If anything fails, the message tells you which step to revisit.

---

## 10. Launch the REPL and try example queries

You're ready.  Launch the interactive REPL:

```bash
grant-copilot --execute
```

You should see:

```
pipeline REPL  session=abc12345  (/quit, /help)

>
```

(The `--execute` flag tells the REPL to run the full pipeline
on every query.  Without it, the REPL only previews intent
extraction and doesn't hit Census.)

### Example queries to try

Paste these one at a time at the `>` prompt.  Each takes 10–30
seconds and returns a data-grounded paragraph with full
citations.

#### A. Simple county lookup

```
What's the median household income in DeKalb County, Georgia?
```

Expected: one paragraph stating the 2024 figure with comparator
framings against the Atlanta metro area, Georgia state median,
and U.S. median, plus a trend delta showing how the value
changed since 2021.  Sources cite `B19013`.

#### B. Neighborhood-level lookup

```
What's the poverty rate in Midtown Atlanta?
```

Expected: one paragraph with the neighborhood-level rate,
multiplier framing against the metro ("2.7× the metro average"),
and an aggregation caveat.  Sources cite `B17001` and list the
specific tracts aggregated.

#### C. Grant-narrative (the hero demo)

```
I'm writing a food pantry grant for Southwest Atlanta.
```

Expected: a grant-ready paragraph expanding the query into five
concepts (SNAP rate, poverty rate, vehicle access, child
poverty, median income) and building the case across all five,
plus a "Peer comparisons" section showing size-matched peer
cities.  Sources cite five distinct tables.

#### D. Trend

```
How has SNAP participation changed in Clayton County over the last 5 years?
```

Expected: a paragraph walking through each year 2020–2024 with
comparator framings, plus related concepts like poverty and
vehicle access.

#### E. Scope-gate refusal (intentional)

```
How many Waffle Houses are in Fulton County?
```

Expected: a polite refusal — the Census doesn't track
individual restaurants, and the scope gate catches that before
any LLM call.

### What every answer comes with

Every query writes an **audit folder** under
`derived_data/data_<timestamp>/` containing:

```
query.txt       the verbatim question you asked
response.txt    prose + key findings + caveats + sources + peer comparisons
logs.txt        every pipeline log event (useful for debugging)
variables.txt   every Census variable used, with plain-English labels
relevant_variables_tract.csv
relevant_variables_county.csv
relevant_variables_msa.csv
relevant_variables_state.csv
relevant_variables_national.csv
```

You can open `response.txt` in any text editor to see the full
formatted answer.  The CSVs give you the raw values that fed the
answer — so every claim in the prose is defensible to a grant
reviewer.

---

## 11. How the REPL works

### Commands

At the `>` prompt, you can either type a question or a command.

| Command    | Effect                                              |
|------------|-----------------------------------------------------|
| `/help`    | Show all commands.                                  |
| `/quit`    | Exit.                                               |
| `/raw <q>` | Run query `q` and print the raw extraction JSON.    |
| `/plan <q>`| Run query `q` through extraction + planning only.   |

Arrow keys work (line editing, previous-command history).
Backspace correctly handles wrapped long lines.  Type `Ctrl+D`
or `/quit` to exit.

### Flags

```bash
grant-copilot --help
```

The two flags that matter:

- `--execute` — **always use this** for real queries.  Without
  it, the REPL only shows intent extraction + geography
  resolution and doesn't fetch Census data.
- `--mock-llm` — offline mode for testing the plumbing without
  burning Vertex credits.  Answers are canned fixtures.

### One-shot mode

If you just want to ask a single question without entering the
REPL:

```bash
python -m scripts.chatbot.pipeline_repl --execute \
    --query "What's the poverty rate in Fulton County?"
```

### Batch mode

If you want to run many queries with a 10-second pause between
each (to avoid Census rate-limit pressure), edit the
`QUERIES` list in `tools/run_smoke_suite.py` and run:

```bash
python -m tools.run_smoke_suite
```

Results get written as JSONL to
`reports/smoke_suite_progress.jsonl`, and per-query dumps land
in `derived_data/` as usual.

### A note about clarifications

The pipeline includes a **clarifier** stage that, in
interactive UIs, can ask the user follow-up questions when the
plan's confidence is low (e.g., "Did you mean North Buckhead or
the whole Buckhead area?").  For this release, **the clarifier
is off by default** — it occasionally proposes patched
geography names that don't resolve, producing empty answers.
The plan reviewer still runs and logs confidence; the pipeline
simply proceeds with its best plan.

You don't need to do anything to keep the clarifier off: it's
disabled in `config/chatbot.yaml` via
`clarification.enabled: false`.  If you ever want to try it
(at your own risk), flip that flag to `true` and re-run.

---

## 12. Troubleshooting

### "Vertex call timed out after 120s"

Gemini occasionally cold-starts at 30–60s (occasionally longer on a busy region) on first call.  Retry
the query.  If it happens repeatedly, check Vertex AI quota for
your project:

```bash
gcloud alpha quotas list --service=aiplatform.googleapis.com
```

### "DefaultCredentialsError: Your default credentials were not found"

Your application-default credentials aren't set.  Rerun:

```bash
gcloud auth application-default login
```

### "HF_TOKEN not set" during hydration

Source your `.env` before running hydrate:

```bash
set -a; source .env; set +a
python -m scripts.setup.hydrate_data_artifacts
```

### "CENSUS_API_KEY not set — proceeding unkeyed"

Also the `.env` wasn't sourced.  You can still run queries but
you'll hit the 500-requests-per-day Census rate limit fast.

### "No geographies resolved" / 0 plan calls

The query names a place that isn't in the bundled gazetteer
(small GA cities, some out-of-state places).  See
`docs/REPLICATION.md` → "Small-place gazetteer gaps" for how to
expand it.

### A query returns "data not available" when it shouldn't

Open the query's dump folder (`derived_data/data_<timestamp>/`)
and inspect `logs.txt`.  The per-stage log lines show exactly
where the plan produced no calls or the data came back empty.

### The chatbot starts but the REPL hangs on the first query

Usually the first Vertex call cold-starts slowly (10–30s).  If
it takes longer than a minute, check the Vertex quota and your
gcloud auth.  `Ctrl+C` will interrupt and return you to the
prompt.

---

## What's next

Once the chatbot is working:

- Skim [`docs/system_design.qmd`](system_design.qmd) to
  understand what's happening at each pipeline stage.
- The 50-query reference run lives in
  [`reports/smoke_suite_report.md`](../reports/smoke_suite_report.md).
  Spot-check a few queries to see the full answer format.
- If you want to rebuild the data layer from scratch (e.g., to
  bump to a new ACS vintage), see
  [`docs/REPLICATION.md`](REPLICATION.md).  Expect ~1 day of
  wall-clock time.
