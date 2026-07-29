# grant-copilot

**Data-grounded grant-writing assistant over U.S. Census data.**

Ask a question the way a grant writer would phrase it — *"I'm writing a
food pantry grant for Southwest Atlanta"* — and get back a paragraph of
quantitative prose with comparator numbers, trend deltas, peer
references, and variable-level citations. Every number in the output
traces to a specific Census variable, year, and geography.

---

## Quick start

```bash
git clone https://github.com/kmcalist682336/grant-copilot.git
cd grant-copilot

gcloud auth application-default login    # once, on your machine
docker compose up --build
```

Open <http://localhost:8000>. The app starts unconfigured and walks you
through the rest in the browser.

That's the whole install. The container carries SpatiaLite, SWIG, and
FAISS — the three dependencies that account for nearly every failed
setup — so you don't install them yourself.

### What you need first

| | Why | Cost |
|---|---|---|
| **Docker Desktop** (or Docker Engine + Compose) | runs everything | free |
| **gcloud CLI**, logged in | Vertex AI credentials | free |
| **A GCP project** with the Vertex AI API enabled and billing on | pays for Gemini calls | ~$0.001/query |
| **~12 GB free disk** | the 8 GB data layer plus caches | — |
| *(optional)* **Census API key** | lifts the 500 requests/day cap | free |
| *(optional)* **HuggingFace token** | downloads the data layer | free |

grant-copilot is bring-your-own-credentials: **your** GCP project pays
for **your** Vertex calls, **your** Census key pulls **your** quota.

Only the `gcloud` login has to happen outside the app — Application
Default Credentials are a file on your machine that gets mounted into
the container, so the browser can detect it but can't create it.
Everything else is collected in the setup screen.

---

## Setup, in the browser

On first run the app shows a **Finish setup** screen listing what's
missing. Each item is one of three kinds:

**Things you type.** GCP project, Census API key, HuggingFace token.
Fill them in, hit *Save & continue*. They're written to `.env` in the
repo and applied immediately — no restart. The project field usually
pre-suggests the right value, read out of your gcloud credentials.

**Things only you can do.** If `gcloud auth application-default login`
hasn't been run, the screen says so and gives you the command. Run it on
your machine, then `docker compose restart`.

**Things it downloads.** The data layer is five prebuilt artifacts
(~8 GB) that would take about a day to rebuild from scratch — a 2.4 M-card
embedding job alone runs six to eight hours. Click **Download data
layer** and watch the log; it takes 5–15 minutes on a decent connection.
It verifies SHA-256 against `config/data_manifest.yaml` and skips files
already in place, so it's safe to re-run.

Once everything's green the app loads the artifacts — about a minute,
almost all of it the 4.4 GB vector index — and switches to the main
screen on its own.

> Prefer the command line? Everything above has a manual equivalent:
> see [Manual setup](#manual-setup) below.

---

## Using the app

The screen is split. **Left is the conversation.** **Right is the
workshop** — five tabs for seeing and changing how answers come out.

### Asking a question

Type into the box and hit **Ask** (or ⌘/Ctrl+Enter). Progress streams
as the pipeline runs — extracting intent, resolving geographies,
planning calls, fetching, synthesizing — because a full query takes
15–25 seconds and silence looks like a hang.

The answer comes back in sections:

| Section | What it is |
|---|---|
| **Output checks** | Automated warnings about the answer. See below. |
| **Answer** | The prose. |
| **Key findings** | Bullets for the highlights. |
| **Caveats** | Data limitations the model was told to surface. |
| **Sources** | Dataset, year, and table for every figure. Derived from the query plan, never written by the model, so they can't be fabricated. |
| **Retrieved values** | The actual numbers, as a table. |
| **Peer comparisons** | Similar geographies, with why each one matched. |
| **Try next** | Suggested follow-up questions. |
| **Run** | Timing, LLM calls, cost, Census calls and cache hits. |

### The three buttons

They do very different amounts of work, and knowing which to use is most
of the skill:

| Button | Time | Cost | Re-runs |
|---|---|---|---|
| **Ask** | 15–25 s | full | Everything |
| **Re-synthesize** | ~3 s | ~$0.0005 | Only the write-up, against **the same numbers** |
| **Re-render** | instant | $0 | Only formatting and which sections show |

**Re-synthesize is the one that matters.** It reuses the data already
fetched, so you can change the prompt and see a new answer about
identical numbers in about three seconds. Changing a prompt and hitting
*Ask* instead means comparing two answers that may not even be about the
same data — you learn nothing.

### Output checks

Warnings that appear above the answer. They never modify anything.

The important one is **unsourced numbers**. It pulls every figure out of
the prose and checks it against what the pipeline actually retrieved,
allowing for rounding. Because citations come from the plan rather than
the model, a number that traces to nothing was invented. That makes
hallucination mechanically detectable here in a way it usually isn't.

It also flags numbers the model *derived* — a difference or percentage
it worked out itself. Those aren't wrong, but they're figures the
pipeline can't cite, which is worth knowing before text goes in a grant
application.

Others check that a year is stated (ACS vintages differ, so an undated
figure is ambiguous), that peer names haven't leaked into prose that
should focus on the anchor geography, and that length is in range.

### The right-hand tabs

**Upstream** — every artifact the pipeline produced for this query, and
whether it reached the writer. This answers the question you can't
otherwise answer: *what could I be using?* Each entry shows what it is,
whether it's being sent, and a preview.

Some are marked *available* but not sent — computed, shown to you, but
deliberately withheld from the model. `peer_contexts` ships that way:
the UI renders peers in their own section, so sending them to the writer
too invites duplicate or contradictory peer talk in the prose.

**Prompt** — the instructions the writer follows. Edit and *Save*, or
*Try without saving* to test a draft. Either way the change applies to
the next Re-synthesize with no restart.

**Presentation** — one config file controlling what's sent to the model,
how output is cleaned up, and which sections render. Save it, hit
Re-render, see the result. Free.

**Trace** — exactly what the model was sent and what it returned. Not a
summary — the real strings, captured as the call was made, for every
stage of the pipeline. When an answer is wrong, read this before
rewriting the prompt: most "the model hallucinated" turns out to be "the
model was handed something confusing."

**Variants** — save named copies of the prompt before big edits. The
version that worked is otherwise one *Save* away from gone.

### Which knob to turn

Four things decide what a user sees, and reaching for the wrong one is
the most common mistake:

| Want to change | Use |
|---|---|
| What the model knows about | **Presentation** → `bundle.send` |
| How it writes | **Prompt** |
| Formatting, length, cleanup | **Presentation** → `formatting` |
| Which sections appear | **Presentation** → `sections` |

If an answer keeps mentioning something you don't want, the instinct is
to add *"DO NOT mention X"* to the prompt. That's usually the wrong
layer. Turn the artifact off in `bundle.send` instead — **the model
can't mention what it was never shown**, and no amount of instruction is
as reliable as not sending it.

---

## Every answer is auditable

Each query writes `derived_data/data_<timestamp>/`:

| File | Content |
|---|---|
| `query.txt` | The verbatim question |
| `response.txt` | Prose, findings, caveats, sources, peers |
| `logs.txt` | Every stage event for this query |
| `variables.txt` | `variable_id`, dataset, plain-English description |
| `relevant_variables_<level>.csv` | One row per (geography × variable × role), per geo level |

Tract rows carry a `parent_display_name` column linking each tract back
to the composite geography that pulled it in — so a reader can confirm
that "Midtown Atlanta" expanded to exactly the right tracts.

---

## Other ways to run it

### Without Docker

Works if your system packages are in order. `libspatialite` and `swig`
must be installed **before** `pip install`, or `faiss-cpu` and
`mod_spatialite` will fail to import — the most common fresh-install
papercut, and the reason the container exists.

<details>
<summary>System packages, per OS</summary>

**macOS (Homebrew)**
```bash
brew install libspatialite swig
```

**Debian / Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y libsqlite3-mod-spatialite swig build-essential
```

**Fedora / RHEL**
```bash
sudo dnf install -y libspatialite-devel swig
```
</details>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_app.py                 # http://127.0.0.1:8000
python run_app.py --no-router     # skip the 4.4 GB index; boots in seconds
```

`--no-router` is the fast loop when you're working on presentation. Don't
judge *answer quality* with it — concept routing falls back to a weaker
path.

### Terminal REPL

```bash
grant-copilot --execute
```

`--execute` runs the full pipeline. Without it the REPL only previews
extraction and never calls Census.

### One-shot

```bash
python -m scripts.chatbot.pipeline_repl --execute \
    --query "What's the poverty rate in Fulton County?"
```

### The 50-query smoke suite

```bash
python -m tools.run_smoke_suite
```

Reruns the suite in
[`reports/smoke_suite_report.md`](reports/smoke_suite_report.md). Expect
~30 minutes with the built-in inter-query pause.

---

## Manual setup

Skip this unless you want to configure by hand instead of in the browser.

```bash
cp .env.example .env      # then fill in the three values
```

| Variable | Get one at |
|---|---|
| `CENSUS_API_KEY` | <https://api.census.gov/data/key_signup.html> |
| `GCP_PROJECT_ID` | your GCP console; needs Vertex AI enabled + billing |
| `HF_TOKEN` | <https://huggingface.co/settings/tokens> (read scope) |

```bash
gcloud auth application-default login
set -a; source .env; set +a
python -m scripts.setup.hydrate_data_artifacts     # ~8 GB
python -m scripts.setup.preflight                  # verify
```

Rebuilding the data layer from scratch instead of downloading it is
documented in [`docs/REPLICATION.md`](docs/REPLICATION.md) — expect about
a day of wall-clock time.

---

## Tests

```bash
pytest -q                                   # native
docker compose run --rm app python -m pytest -q     # in the container
```

Expect **548 passing** once the data layer is hydrated. LLM responses
are mocked — no Vertex calls, no network — but 83 of those tests read the
hydrated SQLite and FAISS artifacts.

Run them *before* hydrating and you'll see **465 passing and 83 errors**
reading `unable to open database file`. That's the missing data, not a
failure.

---

## Documentation

| Doc | For |
|---|---|
| [`docs/APP.md`](docs/APP.md) | The app and prompt lab in depth |
| [`docs/UI_DEVELOPMENT.md`](docs/UI_DEVELOPMENT.md) | Building on the UI — start here if you're improving the interface |
| [`docs/STARTUP_GUIDE.md`](docs/STARTUP_GUIDE.md) | ~45-minute beginner walkthrough |
| [`docs/system_design.qmd`](docs/system_design.qmd) | Architecture and the smoke-test report |
| [`docs/REPLICATION.md`](docs/REPLICATION.md) | Rebuilding the data layer |

---

## Troubleshooting

**The setup screen says Google Cloud credentials are missing.** Run
`gcloud auth application-default login` on your machine — not inside the
container — then `docker compose restart`. The container reads a mounted
copy of your credentials and can't create them.

**Setup can't save to `.env`.** The container writes through the repo
mount and needs permission. If your user id isn't 1000:

```bash
APP_UID=$(id -u) APP_GID=$(id -g) docker compose up
```

(Not `UID=` — bash marks that variable readonly.)

**"Vertex call timed out."** Gemini 2.5 Flash is usually 1–5 s but
occasionally cold-starts at 15–30 s. The client allows 120 s. Re-run;
if it persists, check your Vertex quota.

**Every answer has no data and the checks complain.** Confirm the Census
API is reachable from your network:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://api.census.gov/data/2023/acs/acs5?get=B19013_001E&for=county:121&in=state:13"
```

Anything other than `200` — a `302` to an HTML page especially — means
something between you and the API is intercepting requests. That's a
network problem, not a bug.

**"No geographies resolved."** Some small Georgia places aren't in the
gazetteer's TIGER snapshot. See "Small-place gazetteer gaps" in §9 of
`docs/system_design.qmd`.

**Startup is slow.** Expected — it's the 4.4 GB vector index. Use
`--no-router` when you don't need routing quality.

---

## License

Apache License 2.0. See `LICENSE`.

Census data is public domain. The prebuilt artifacts distributed via
HuggingFace derive from Census Bureau data (ACS 5-year, ACS 1-year,
TIGER/Line) and may be redistributed freely.

---

## Citation

```
grant-copilot: a data-grounded grant-writing assistant over U.S.
  Census data.  2026.  https://github.com/kmcalist682336/grant-copilot
```
