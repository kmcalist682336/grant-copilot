# Demo script: five queries to show off the system

Once you've finished [`STARTUP_GUIDE.md`](STARTUP_GUIDE.md) and
have the REPL running, paste these queries one at a time to show
off the full range of the pipeline.

Launch with:

```bash
set -a; source .env; set +a
grant-copilot --execute
```

---

## 1. The "easy win" — Tier 1 single-concept lookup

```
What's the median household income in DeKalb County, Georgia?
```

**What to notice:**
- Completes in 10–15 seconds.
- Prose gives the 2024 figure AND a 2021→2024 trend delta.
- Comparator chain cites metro, state, and national medians.
- Sources section cites `B19013` with the plain-English
  variable label.

**Under the hood:** `food_access` didn't fire; `specific_lookup`
frame matched deterministically via the "what's" trigger; Tier 1
short-circuit found `median household income → B19013` in the
concept map, so the semantic router / critic / rewriter chain
was entirely skipped.  One LLM call total for the synthesis.

---

## 2. Atlanta neighborhood (composite geography)

```
What's the poverty rate in Midtown Atlanta?
```

**What to notice:**
- Response says "**29.26%**" with a "2.68× the metro average"
  framing.
- A caveat surfaces: *"value is the unweighted mean across 12
  tracts."*
- Open the derived-data dump folder — the tract-level CSV has
  12 rows, each tagged `parent_display_name=Midtown`.

**Under the hood:** The gazetteer's `place_tract_map` resolved
"Midtown Atlanta" to ≈ 12 census tracts.  The planner issued a
tract-level `B17001` fetch restricted to those tracts.  The
aggregator summed numerators + denominators across the filtered
set and reported a single neighborhood-level ratio.

---

## 3. The hero demo — grant-narrative frame

```
I'm writing a food pantry grant for Southwest Atlanta.
```

**What to notice:**
- A full paragraph weaves five different indicators (SNAP rate,
  poverty rate, vehicle access, child poverty, median income)
  into one grant-ready narrative.
- "Peer comparisons" section below the answer names 3–5 peer
  cities (Columbus OH, Kansas City KS, Chattanooga TN, …) and
  explains *why* each was picked (same state / same region /
  similar population + matched features).
- Takes 30–50 seconds — five concepts × five geographic levels.

**Under the hood:** `food_access` frame matched
deterministically on "food pantry," expanding the intent from
one concept to six.  Five of the six hit Tier 1.  The planner
emitted ~30 Census fetches across primary + comparator chain.
The synthesizer was explicitly told (in its prompt) *not* to
mention peers in the prose — peers live only in the dedicated
comparisons section so the prose stays tight.

---

## 4. Trend (multi-year expansion)

```
How has SNAP participation changed in Clayton County over the last 5 years?
```

**What to notice:**
- Response walks year by year (2020 → 2024).
- Comparator framings ("2.46× the metro level") anchor the
  reader throughout.
- The pipeline fetched and aggregated across 5 years × 5 geo
  levels × 5 concepts = ~125 Census calls, most hitting cache.

---

## 5. Scope-gate refusal (what the system *won't* do)

```
How many Waffle Houses are in Fulton County?
```

**What to notice:**
- Comes back in 3–5 seconds with a polite refusal:
  *"I don't have Census data on that topic.  Contains
  out-of-scope token 'restaurant' — Census does not track
  this subject.  I can help with demographic, economic,
  housing, transportation, health-insurance, and education
  indicators for US geographies."*
- No hallucinated answer pulled from the nearest-adjacent
  Census table.

**Under the hood:** The scope gate ran first, caught the
"restaurant" substring, and skipped extraction + resolution +
routing + synthesis entirely.  This is the most important
safety rail in the system — every query that isn't
Census-answerable gets rejected before the pipeline can
confabulate.

---

## Bonus: exercise the 50-query gold-standard suite

Want to see every query at once?  Run the full smoke-test suite:

```bash
python -m tools.run_smoke_suite
```

This runs all 50 queries from the gold-standard set with a 10-
second inter-query pause (to avoid Census API rate pressure).
Total runtime: ~30 minutes.  Results land in
`reports/smoke_suite_progress.jsonl`; compare to the reference
results in `reports/smoke_suite_report.md`.

On a cold cache the first run will cost roughly **$0.20–$0.50 of
Vertex AI usage** depending on how many queries hit the scope
gate or Tier 1 short-circuit.  Repeat runs are near-free since
Census responses are cached for 30 days.
