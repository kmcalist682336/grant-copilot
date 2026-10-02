# Metadata-driven record datasets

## Adding a dataset

Add a dataset entry with table_id, supported_years, geography_column,
geography_levels, aliases, and variables or categories. See
tests/fixtures/record_datasets.yaml for a complete working example.

- variables can include descriptions, value_type, and values (canonical_value
  plus aliases). categories is a shorthand mapping phrases to stored values.
- metrics contains declared aggregation recipes, including numerator and
  denominator predicates. The existing dataset-scoped record_metric_map.yaml
  remains supported.
- geography_strategy is identity (exact stored area IDs) or tract_prefix
  (shared gazetteer tract expansion); geography_match controls exact/prefix SQL.
- A default geography must be explicitly configured and is disclosed in notes.

The standard configuration loads lazily. load_record_datasets(path) can load a
separate metadata file. No import or subclass is needed for each source.

## Interpretation and execution

After selecting a dataset, call interpret_record_metrics with that explicit ID.
One LLM call receives its available fields, categories, aliases, and relevant
metric recipes. It proposes structured analyses, never SQL. The shared code
validates dataset identity, measures, filter dimensions, and declared categories,
while preserving the original geography and years. Invalid interpretation stops
the website request instead of silently falling back to potentially incomplete
population filters.

plan_record_query uses the same definitions for deterministic planning, and
DuckDBCaller executes supported record layouts. Curated calculations remain
authoritative; the LLM does not define a statistical denominator.

The tests load the synthetic pantry dataset entirely from YAML and run actual
local Parquet through the planner, DuckDB connector, and aggregator. HMDA
calculation regression tests remain in place.

## Scope

Source selection now runs before frame expansion, for extracted concepts and
existing analyses. dataset_selection.py uses a unique declared metric match,
otherwise the global semantic router. Extractor dataset hints do not override
matching evidence. Semantic candidates are grouped by source; ACS/decennial
products count as Census. A winning score below 0.05, or a second source within
90% of the winning score, stops selection with a clarification error. These
are routing heuristics, not calibrated statistical confidence probabilities.
An explicitly user-named dataset ID can resolve a tie, but still needs evidence
that the measure exists. Unsupported winners are never replaced by HMDA.

The selected ID reaches the shared interpreter and planner. Scope checking
also receives the configured field and metric vocabulary. Existing single
record_caller instances retain their legacy HMDA meaning; new sources supply
answer_query(record_caller={"dataset_id": connector}). Missing connectors
return an error without fetching or synthesizing an answer. Currently one
record dataset per query is supported, optionally alongside Census concepts;
multiple record datasets require separate queries.

Registration does not install storage credentials or generate semantic cards.
A real source needs metadata, matching cards for semantic discovery, and its
storage connector. A different storage format may need a reusable connector.

HMDA denominator definitions are unchanged: denial and approval use all matching
records; the completed-application origination recipe restricts its population.
