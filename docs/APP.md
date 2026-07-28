# The web app — running it, and shaping what users see

A browser UI over the existing pipeline, plus the tools for iterating on
how answers read. It loads the artifacts once and answers queries; the
right-hand panel is where you change the output.

---

## Running it

**Docker (recommended).** The gazetteer is a SpatiaLite database, and
`libspatialite` is the single most common fresh-install failure on both
macOS and Linux. The image has it.

```bash
# once, on the host — the data layer is mounted, not baked in
python -m scripts.setup.hydrate_data_artifacts     # needs HF_TOKEN
gcloud auth application-default login

docker compose up --build
# http://localhost:8000
```

The repo is bind-mounted, so editing prompts, `presentation.yaml`, or
Python takes effect without a rebuild. That's not a convenience — the
prompt lab *writes* to `prompts/v1/synthesizer.yaml` and
`config/presentation.yaml`, so a baked-in copy would lose every edit on
restart.

**Native.** Works if your system packages are in order — see the
README's Install section.

```bash
python run_app.py                # http://127.0.0.1:8000
python run_app.py --no-router    # skip the 4.4 GB FAISS load, fast boot
```

Startup takes a minute, almost all of it the FAISS index. `/readyz`
reports per-artifact status; the UI header shows it too.

---

## The four dials

What a user sees is decided in four places, in this order. Picking the
right one is most of the skill:

| Layer | Where | Cost | Controls |
|---|---|---|---|
| 1 | `presentation.yaml` → `bundle.send` | free | **What the model is shown** |
| 2 | `prompts/v1/synthesizer.yaml` | ~$0.0005/try | **How it's told to write** |
| 3 | `presentation.yaml` → `formatting`, `lints`<br>`app/postprocess.py` | free | Deterministic cleanup after |
| 4 | `presentation.yaml` → `sections` | free | Which sections render |

All four hot-reload. Save, then use the appropriate button — no restart.

### The most common mistake

When an answer keeps mentioning something you don't want, the instinct
is to add `DO NOT mention X` to the system prompt. That's layer 2
fighting a layer 1 problem, and the model wins often enough to matter.

This codebase already lost that fight once: there are ~15 lines in the
synthesizer prompt asking the model not to name peer geographies.
Setting `bundle.send.peer_contexts: false` makes all of them
unnecessary, because **the model cannot mention what it was never
shown**.

Check the Upstream panel first. Always.

---

## The Upstream panel

Every artifact the pipeline produced for the current query, whether it
reached the synthesizer, how big it is, and a preview.

This is the synthesis *bundle* — one dict assembled by
`build_synthesis_bundle()` and handed to `synthesize()`. The panel is
generated from its keys, so:

- An artifact a new dataset adds shows up here **with no code change**.
- Unregistered keys are listed but **not sent** until you opt in, so new
  pipeline work never silently changes answers or per-query cost.
- Keys are omitted when empty, so `"peer_contexts" in bundle` is a real
  test of "did this query actually produce peers".

That last property is what makes conditional cleanup possible:

```python
# app/postprocess.py
def custom_clean(answer, bundle, formatting):
    if "peer_contexts" in bundle:
        ...                       # only when peers were actually computed
    return answer
```

Available at `GET /api/upstream/{run_id}` as JSON if you'd rather script
against it.

---

## The three buttons

| Button | Time | Cost | What re-runs |
|---|---|---|---|
| **Re-render** | instant | $0 | Layers 3 + 4 on the existing answer |
| **Re-synthesize** | ~3 s | ~$0.0005 | The synthesizer LLM, **same numbers** |
| **Ask** | 15–25 s | full | The entire pipeline |

Re-synthesize is the one that matters. It reuses the cached bundle — no
extraction, no routing, no data fetches — so you can rewrite the prompt
and see a new writeup against identical data in about three seconds.
Editing the prompt and hitting Ask instead means comparing two answers
that may not even be about the same numbers.

"Try without saving" re-synthesizes with the draft in the editor without
writing it to disk. The last 25 runs stay cached.

---

## Reading the trace

The Trace tab shows the exact strings sent and returned — captured at
the LLM client boundary, so it covers **every** node, not just the
synthesizer: scope gate, clarifier, frame matcher, router critic, plan
reviewer, followup finder.

When an answer is wrong, read the payload before rewriting the prompt.
Most "the model hallucinated" turns out to be "the model was sent
something confusing", and you can only tell them apart by looking.

---

## Output checks

Lints run after every synthesis and appear above the answer. They never
modify anything.

**`forbid_unsourced_numbers`** is the important one. It extracts every
number from the prose and checks it against the values actually
retrieved, allowing for the rounding the prompt requests. Since
citations are derived from the plan rather than the model, a number that
traces to nothing was invented. It also fires on numbers the model
*derived* — a difference or percentage it computed itself — which is
still worth knowing, because the prose is stating a figure the pipeline
can't cite.

Others: `require_year_mentioned`, `forbid_peer_names` (catches layer-1
leakage), `prose_word_range` (advisory).

---

## Prompt variants

Save a named copy before a big edit:

```
prompts/variants/<name>.yaml
```

Use the Variants tab to save, load, and delete. Load pulls a variant
into the editor without activating it. Do this *before* experimenting —
the version that worked is otherwise one Save away from gone.

---

## API

The frontend binds to `ChatResponse` in `app/schemas.py`, **not** to the
pipeline's `QueryResponse`. `to_public()` maps between them and is
deliberately the only place that knows both shapes — when the backend
changes, that function absorbs it and the contract holds.

| Endpoint | Purpose |
|---|---|
| `GET /` | The UI |
| `GET /healthz` | Process alive |
| `GET /readyz` | Artifacts loaded, per-artifact status |
| `POST /chat` | Full pipeline |
| `GET /chat/stream?q=` | Full pipeline, SSE stage events |
| `POST /resynthesize` | Synthesizer only, cached bundle |
| `POST /rerender` | Layers 3+4 only, no LLM |
| `GET /api/upstream/{run_id}` | The bundle inventory |
| `GET/PUT /api/prompt` | The synthesizer prompt |
| `GET/PUT /api/presentation` | `presentation.yaml` |
| `GET/POST/DELETE /api/variants` | Named prompt variants |

Interactive schema docs at `/docs`.

---

## How this stays working as the backend changes

`app/pipeline_adapter.py` is the **only** module in `app/` that imports
from `scripts.chatbot`. Everything else imports from it. When the
pipeline changes, that one file absorbs it.

`tests/test_app_contract.py` asserts the pipeline still exposes what the
app needs — the `answer_query` kwargs, the `QueryResponse` fields, the
`synthesize` signature. A breaking change fails a named test rather than
surfacing in front of a class.

If a contract test fails, the fix is usually a few lines in the adapter
or in `to_public()`, not a change to the pipeline.

---

## Things to know

**Queries serialize.** All pipeline work runs on one dedicated thread,
because the gazetteer and metadata index are SQLite connections and
SQLite refuses cross-thread use. Fine for a prompt lab; making it
concurrent means per-thread connections and a FAISS access strategy, not
a bigger thread pool.

**The clarifier is off over HTTP.** It needs a blocking round-trip
mid-pipeline. The orchestrator degrades gracefully (proceeds with the
original plan). `status` reserves `"needs_clarification"` so adding the
multi-turn flow later isn't breaking.

**A broken config won't take the app down.** Invalid
`presentation.yaml` falls back to defaults with a banner; invalid
`synthesizer.yaml` falls back to the built-in prompt; a crashing
`custom_clean` falls back to the uncleaned answer. All three say so.
