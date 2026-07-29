# Building on the UI

For students improving the interface. It covers how the current UI is
put together, what you can safely change, and how to work productively
with an LLM coding assistant like Claude Code on this particular
codebase.

The existing UI is a **working reference implementation, not a finished
product**. It was built to prove the pipeline out and to give the prompt
work somewhere to live. Making it genuinely good for a grant writer is
the open problem, and it's yours.

---

## Part 1 — How it's built

### The whole thing is one file

[`app/static/index.html`](../app/static/index.html) — about 760 lines:
CSS, markup, and JavaScript in one file, no framework, no build step,
no `node_modules`.

That's a deliberate choice, and worth understanding before you decide to
change it:

- **No toolchain to install.** You edit a file and reload the page.
  Nobody debugs a broken webpack config instead of doing the actual work.
- **The repo is bind-mounted into the container**, so a save is live
  immediately. A build step would break that loop.
- **An LLM assistant can hold the whole file in context.** This matters
  more than it sounds — it means you can ask for a change and get one
  that's consistent with everything else, rather than one that fights a
  half-seen codebase.

The cost is real too: no components, no scoped styles, no type checking.
Past ~1500 lines that trade stops paying. See
[Should I switch to a framework?](#should-i-switch-to-a-framework) below.

### Client and server split

The server does everything that involves data. The browser does nothing
but ask and render:

```
browser ──► GET /chat/stream?q=…  ──► pipeline (15-25s, streamed)
        ──► POST /resynthesize    ──► only the write-up (~3s)
        ──► POST /rerender        ──► formatting only (instant, free)
        ──► GET/PUT /api/prompt, /api/presentation, /api/variants
        ──► GET  /api/upstream/{run_id}
        ──► GET  /readyz, /api/setup
```

There is **no application state in the browser** beyond `currentRunId`
and a couple of UI flags. Reload the page and nothing is lost except the
current view. Keep it that way — it's why the UI is easy to change
without breaking anything.

### The contract you render against

Every response is a `ChatResponse`, defined in
[`app/schemas.py`](../app/schemas.py). That file is the single source of
truth for what you can display. Read it before you start.

```python
class ChatResponse:
    status: str                    # "ok" | "no_data" | "error"
    query: str
    prose: str
    key_findings: list[str]
    caveats: list[str]
    citations: list[Citation]      # table_id, dataset, year, title, variables, url
    geos: list[GeoOut]             # name, geo_level, geoid, confidence, tract_count
    values: list[ValueOut]         # geo, concept, year, value, ratio, units, caveat
    peer_groups: list[PeerGroupOut]
    anomalies: list[AnomalyOut]
    followups: list[FollowupOut]
    lints: list[Lint]              # rule, severity, message, detail
    upstream: list[UpstreamItem]
    trace: TraceOut | None
    metrics: MetricsOut | None
    error: str | None
    config_warning: str | None     # e.g. presentation.yaml failed to parse
    run_id: str | None
```

**This is deliberately not the pipeline's internal shape.** The mapping
lives in one function, `to_public()` in `app/schemas.py`, precisely so
that backend work — a new dataset, a reshaped model — lands there
instead of in your UI. Bind to `ChatResponse` and backend churn won't
break you.

Live example to look at while you work:

```bash
curl -s "http://localhost:8000/chat/stream?q=median%20income%20in%20DeKalb%20County" \
  | tail -1 | sed 's/^data: //' | python -m json.tool
```

### Map of `index.html`

| Lines | What |
|---|---|
| 7–106 | All CSS. Custom properties at the top, then layout, then components. |
| 108–233 | Markup: header, setup gate, left pane, right tabs. |
| 239–265 | Helpers — `esc()`, `fmtNum()`, `banner()`, `setBusy()`. |
| 267–425 | Rendering — `render()`, `renderUpstream()`, `renderTrace()`. |
| 427–482 | Actions — `ask()` (SSE), `resynth()`, `rerender()`. |
| 484–536 | Config panes — prompt, presentation, variants. |
| 537–567 | Event wiring and tab switching. |
| 568–732 | Setup screen. |
| 733–764 | Boot: poll status, reveal the app when ready. |

If you're improving the *answer* presentation — which is where the value
is — you'll spend nearly all your time in `render()`, lines 267–361.

### Things in there for a reason

**`esc()` on every interpolated value.** All rendering is template
strings assigned to `innerHTML`. Census place names and LLM prose are
untrusted input. If you add a field and skip `esc()`, you've added an
XSS hole. Every existing interpolation uses it — match that.

**The re-render guard in `renderSetup()`.** The status poll runs every
2 seconds. An earlier version reassigned `innerHTML` on every poll,
which destroyed the `<input>` elements — users could click into the
token box and watch their paste vanish. The fix only rebuilds when state
actually changed, and preserves values, focus, and cursor position when
it does. **If you add inputs that live inside a polled region, you have
to handle this**, or you'll reintroduce the same bug.

**Theme via `prefers-color-scheme`.** Colors are CSS custom properties
declared twice, once per scheme. Add colors by adding variables, not
hex codes in rules, or dark mode breaks in ways you won't notice on your
own machine.

---

## Part 2 — Working with an LLM assistant

You'll move much faster with Claude Code than without it. You'll also
generate subtly broken code faster. This section is about the difference.

### Give it the right context

Bad: *"make the UI look better."* You'll get a redesign that ignores the
data model and invents fields that don't exist.

Good — name the file, the contract, and the constraint:

> Read `app/static/index.html` and `app/schemas.py`. I want to redesign
> the "Retrieved values" table in `render()` so a grant writer can scan
> it quickly. Keep using only fields that exist on `ValueOut`. No
> external libraries — the page must work offline. Keep the existing CSS
> custom properties so dark mode still works.

The three constraints in that prompt — real fields, no CDN, keep the
variables — are exactly the three things an assistant gets wrong here.

### Verify these every time

An assistant will produce plausible code that fails in ways that don't
show up until a specific query. Check:

**Invented fields.** `r.answer.confidence`, `v.margin_of_error`,
`r.summary` — none of these exist. The tell is a field that would be
useful but isn't in `schemas.py`. Grep before you trust it:

```bash
grep -n "class ChatResponse" -A 30 app/schemas.py
```

**External dependencies.** A suggestion to `<script src="https://cdn...">`
for a chart library or icon set breaks offline use and adds a supply-chain
dependency to a class project. If you want charts, ask for inline SVG.

**Missing `esc()`.** Especially in newly added interpolations.

**Empty and error states.** Assistants write the happy path. Every
section needs to handle: field absent, empty list, `status: "error"`,
and `status: "no_data"`. Ask explicitly — *"handle the case where
`values` is empty and where `status` is `error`"* — and then actually
test it by asking something nonsensical.

**Long content.** Some Census place names are 60+ characters; a trend
query can return 40 rows; prose can run 300 words. Ask for a query with
a lot of geographies and watch the layout.

### A loop that works

1. **One component at a time.** "Redesign the citations block" gets a
   good result. "Redesign the UI" gets a plausible-looking rewrite you
   can't review.
2. **Ask for the reasoning first** on anything structural. *"Before
   writing code, tell me how you'd restructure the answer pane and
   why."* Cheap to redirect at that stage.
3. **Read the diff.** Every line. This file has no type checker and no
   tests — you are the only review.
4. **Test against a real query**, not a fixture you made up.
5. **Commit working states often.** `git add -p` and a message per
   change. Being able to get back to the last thing that worked is worth
   more than a tidy history.

### Give it the failure, not a summary

When something breaks, paste the actual browser console error, the
actual JSON, the actual screenshot. *"The table looks wrong"* produces
guesses. *"`v.units` is undefined for these three rows, here's the
JSON"* produces a fix.

### Ask it to explain rather than patch

If you don't understand why something works — the SSE loop, the re-render
guard, `setBusy()` — ask for an explanation before changing it. The
2-second poll interacting with focus is exactly the kind of thing that
looks like dead code and isn't.

---

## Part 3 — What to actually build

Ordered roughly by value to a real grant writer. You are not expected to
do all of these.

### Worth doing first

**Make the answer scannable.** Right now prose, findings, and caveats
are three similar stacked blocks. A grant writer wants to find the
number, confirm the source, and paste something into a document. What
should be biggest? What can collapse?

**Make copy-out work.** There is no way to get an answer into a grant
application except selecting text. A "copy as formatted text" or "copy
with citations" button is small work and immediately useful. Consider
what a pasted citation should look like in Word.

**Make the numbers legible.** `72483.0` should be `$72,483`. `0.135`
should be `13.5%`. `ValueOut` carries `units` — use it. This is the most
visible quality gap in the current UI.

**Show the geography.** Answers cover neighborhoods, corridors, counties,
metros. `geos` carries `geo_level`, `tract_count`, `confidence`. A reader
who doesn't know what "34 tracts in Fulton County" means needs help; low
confidence should be visible.

**Make output checks feel useful, not noisy.** An unsourced-number
warning is important; a "prose is 55 words" note is not. Same visual
weight today. Severity is on every lint — use it.

### Worth doing next

**Query history.** Nothing persists. Even in-memory history with
click-to-reload would help, and the backend already caches 25 runs.

**Loading that reflects reality.** Stage names stream from the server.
A 20-second wait with real progress feels very different from a spinner.

**Comparison view.** Re-synthesize returns a new `run_id` and the old run
stays cached, so two answers about identical numbers can sit side by
side. The backend supports it; nothing in the UI uses it.

**Charts.** A trend query returns a series. Inline SVG, no library.
Small and correct beats large and impressive.

**Mobile.** There's one breakpoint at 950px that stacks the panes. It's
not really usable on a phone.

### Ideas worth considering

- Two modes — a clean "writer" view and the current "workshop" view,
  since students tuning prompts and grant writers using answers want
  different things
- Export to Word/PDF with citations intact
- Inline source hover: point at a number, see which variable produced it
- Saved/annotated answers for a grant in progress

---

## Part 4 — Ground rules

### Don't change these

**`app/pipeline_adapter.py`.** The only module importing from
`scripts.chatbot`. It exists so backend changes land in one file.

**`to_public()` in `app/schemas.py`** — unless you're deliberately adding
a field, in which case change `ChatResponse` *and* the mapping *and*
`tests/test_app_contract.py` together.

**Anything under `scripts/chatbot/`.** That's the pipeline. If the UI
needs data it can't get, the fix is usually a new field in
`ChatResponse`, not a pipeline change.

### Safe to change freely

Everything in `app/static/index.html`. All of `config/presentation.yaml`.
`app/postprocess.py`, which is explicitly yours.

### Before you commit

```bash
pytest -q      # 548 passing with the data layer hydrated;
               # 465 passing + 83 errors without it
```

The Python tests won't catch UI regressions — there's no JS test setup —
so check by hand:

- [ ] A normal query renders every section
- [ ] A nonsense query ("jet ski ownership in Buckhead") shows a clean refusal
- [ ] Re-render and Re-synthesize both still work
- [ ] Dark mode and light mode both look right
- [ ] Narrow window (~600px) doesn't overflow horizontally
- [ ] Browser console has no errors
- [ ] The setup screen still accepts a pasted token — click in, paste,
      wait five seconds, confirm it's still there

That last one has a real bug behind it. Don't skip it.

### If you want to test JavaScript properly

There's no JS test harness. You can get one cheaply by extracting a
function and running it in headless Chromium:

```bash
# see docs/APP.md; the pattern is:
#   1. pull the function text out of index.html
#   2. wrap it in a page with a stub DOM and assertions
#   3. chromium --headless=new --dump-dom http://localhost:PORT/harness.html
```

That's how the paste bug above was verified — the same harness fails
against the buggy version and passes against the fix, which is the
property a regression test needs.

---

## Should I switch to a framework?

Probably not, and here's the honest trade.

**Reasons to stay:** no build step, instant reload through the bind
mount, whole file fits in an assistant's context, nothing to learn before
being useful, no dependency surface.

**Reasons to move:** past ~1500 lines the single file gets genuinely hard
to navigate; real component reuse; scoped styles; a type checker across
the `ChatResponse` boundary.

If you do move, the sane version is a small React or Svelte app that
consumes the same endpoints, lives in its own directory, and is served
by the same FastAPI process. **Do not change the API to suit the
frontend** — the contract is what lets the backend keep moving.

And be clear-eyed about the cost: you'll spend the first chunk of your
time on tooling rather than on the interface. If the goal is a better
experience for a grant writer, that time is usually better spent on the
"worth doing first" list above.

---

## Where to start

1. Get it running — [README](../README.md).
2. Ask three real questions. Watch what's awkward. Write it down.
3. Read `app/schemas.py`, then `render()` in `index.html`.
4. Pick one item from "worth doing first."
5. Make it work. Commit. Pick the next one.

The best thing you can bring is having actually used it. Most of what's
wrong with the current UI is obvious within ten minutes of real use and
invisible from reading the code.
