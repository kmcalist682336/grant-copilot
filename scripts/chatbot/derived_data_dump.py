"""Dump the source data that fed the generated answer.

After every query, if enabled, writes a self-contained folder:

  ``derived_data/data_<YYYYMMDD_HHMMSS>/``   (timestamp = query start)
      ├── query.txt                         verbatim user question
      ├── variables.txt                     variable_id → plain-English
      ├── relevant_variables_tract.csv      one row per (tract × variable)
      ├── relevant_variables_county.csv     ...
      ├── relevant_variables_msa.csv
      ├── relevant_variables_state.csv
      └── relevant_variables_national.csv

Rows in the tract CSV carry a ``parent_display_name`` column that
links each underlying tract back to the composite geography
(neighborhood, corridor, region, directional sub-area, …) whose
request caused it to be fetched — so the reader can tell which
tracts belonged to "Buckhead" vs "North Buckhead" vs a comparator
county's tract set.

The CSVs are populated from the raw Census ``FetchResult`` rows
(filtered by the planner's ``tract_filter`` when applicable), not
from post-aggregation means — so you see the exact numbers that
went into the computation, one row per source record. The composite
aggregate (e.g. Buckhead's single income figure) appears in the
prose answer, not in the CSV, to keep "source data" distinct from
"derived value" in the audit trail.
"""
from __future__ import annotations

import csv
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from scripts.chatbot.aggregator import AggregatedResult
from scripts.chatbot.census_caller import FetchResult
from scripts.chatbot.concept_map import ConceptVariables
from scripts.chatbot.planner import PlanResult, PlannedCall

logger = logging.getLogger(__name__)


# Unified column schema. parent_display_name / parent_geo_level stay
# empty for admin-level rows; they only populate when the row is a
# tract that was pulled because a composite geography asked for it.
_CSV_COLUMNS = [
    "geo_id",
    "display_name",
    "geo_level",
    "parent_display_name",
    "parent_geo_level",
    "role",
    "year",
    "concept",
    "table_id",
    "dataset",
    "variable_id",
    "variable_slot",
    "value",
]

# API geo_level → file-suffix. Only `us` needs remapping; the user's
# natural vocabulary is "national" for that slice. Everything else
# maps through as-is, including non-standard composite levels (which
# fold into the tract CSV via _api_level_for_call).
_LEVEL_FILE_SUFFIX = {
    "us": "national",
}

# Composite geo types whose API calls actually run at tract level —
# so their fetch rows land in the tract CSV with parent_display_name
# pointing back to the composite.
_COMPOSITE_GEO_LEVELS = {
    "neighborhood", "corridor", "region", "landmark_area",
}


def dump_derived_data(
    *,
    query: str,
    plan: PlanResult,
    aggregated: AggregatedResult,
    fetch_results: Optional[Iterable[FetchResult]] = None,
    metadata_db: Optional[sqlite3.Connection],
    folder: Path,
) -> Optional[Path]:
    """Write CSV + sidecar files for one query to ``folder``.

    Returns the folder path on success, or None if there was nothing
    to dump / the folder couldn't be created. The folder path is
    supplied by the caller (the orchestrator chooses it at query
    start so the timestamp reflects when the user asked).
    """
    if not fetch_results:
        # Without raw rows we can't honor the "show me the source
        # data" contract. An empty dump would be misleading.
        return None

    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("derived_data: could not create %s: %s", folder, e)
        return None

    try:
        (folder / "query.txt").write_text(query.rstrip() + "\n")
    except OSError as e:                               # pragma: no cover
        logger.warning("derived_data: failed writing query.txt: %s", e)

    rows_by_suffix: dict[str, list[dict]] = defaultdict(list)
    used_vars: set[tuple[str, str]] = set()

    # Identity index: every FetchResult carries the same APIPlanCall
    # object the PlannedCall wraps, so identity match is exact.
    plan_by_api_id: dict[int, PlannedCall] = {
        id(c.api_call): c for c in plan.calls
    }

    for fr in fetch_results:
        if not fr.succeeded or not fr.rows:
            continue
        call = plan_by_api_id.get(id(fr.plan))
        if call is None:
            continue

        api_level = call.api_call.geo_level
        table_id = call.api_call.table_id
        dataset = call.api_call.dataset
        data_vars = list(call.api_call.variables)
        cv = call.variables

        # The geography the user (or comparator expansion) asked about.
        parent_geo = None
        if 0 <= call.geo_idx < len(plan.resolved_geos):
            parent_geo = plan.resolved_geos[call.geo_idx]
        parent_name = parent_geo.display_name if parent_geo else ""
        parent_level = parent_geo.geo_level if parent_geo else ""

        # Parent linking policy: if the fetch ran at tract level AND
        # the resolved geography is at a coarser level (neighborhood,
        # corridor, region, place — anything but "tract" itself),
        # that coarser geo asked for the tract fetch as a composite
        # and the link is informative. For direct admin queries
        # (parent_level == api_level), the parent columns would just
        # duplicate geo_id — leave them blank.
        if api_level == "tract" and parent_level and parent_level != "tract":
            row_parent_name = parent_name
            row_parent_level = parent_level
        else:
            row_parent_name = ""
            row_parent_level = ""

        # Restrict tract responses to the composite's tract set.
        rows = _filter_rows_for_call(fr.rows, call)

        for row in rows:
            geo_id, display_name = _geo_id_from_row(
                row, api_level, parent_geo,
            )
            for var_id in data_vars:
                raw = row.get(var_id)
                slot = _slot_for(cv, var_id)
                rows_by_suffix[_suffix_for_level(api_level)].append({
                    "geo_id": geo_id,
                    "display_name": display_name,
                    "geo_level": api_level,
                    "parent_display_name": row_parent_name,
                    "parent_geo_level": row_parent_level,
                    "role": call.role,
                    "year": call.year,
                    "concept": _concept_text(plan, call.concept_idx),
                    "table_id": table_id,
                    "dataset": dataset,
                    "variable_id": var_id,
                    "variable_slot": slot,
                    "value": _fmt(raw),
                })
                used_vars.add((var_id, dataset))

    for suffix, rows in rows_by_suffix.items():
        filename = f"relevant_variables_{suffix}.csv"
        try:
            with (folder / filename).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                w.writeheader()
                w.writerows(rows)
        except OSError as e:                           # pragma: no cover
            logger.warning(
                "derived_data: failed writing %s: %s", filename, e,
            )

    # variables.txt sidecar.
    labels = _lookup_labels(used_vars, metadata_db)
    try:
        with (folder / "variables.txt").open("w") as f:
            f.write(
                "# Census variable reference for this query.\n"
                "# Columns: variable_id\\tdataset\\tdescription\n"
                "# Descriptions come from the Census API's 'label' field,\n"
                "# rewritten into plain English during metadata build.\n\n"
            )
            for (vid, dataset), label in sorted(labels.items()):
                f.write(f"{vid}\t{dataset}\t{label}\n")
    except OSError as e:                               # pragma: no cover
        logger.warning("derived_data: failed writing variables.txt: %s", e)

    return folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suffix_for_level(api_level: str) -> str:
    return _LEVEL_FILE_SUFFIX.get(api_level, api_level)


def _concept_text(plan: PlanResult, concept_idx: int) -> str:
    if 0 <= concept_idx < len(plan.intent.concepts):
        return plan.intent.concepts[concept_idx].text
    return ""


def _slot_for(cv: ConceptVariables, var_id: str) -> str:
    """Which ConceptVariables role does ``var_id`` play in this call?"""
    if cv.value == var_id:
        return "value"
    if cv.denominator == var_id:
        return "denominator"
    if cv.numerator:
        nums = (cv.numerator if isinstance(cv.numerator, list)
                else [cv.numerator])
        if var_id in nums:
            return "numerator"
    if cv.components and var_id in cv.components:
        return "component"
    return ""


def _filter_rows_for_call(
    rows: list[dict], call: PlannedCall,
) -> list[dict]:
    """Apply the planner's tract_filter so tract responses are
    restricted to the composite geo's member tracts."""
    if call.api_call.geo_level != "tract" or not call.tract_filter:
        return rows
    wanted = set(call.tract_filter)
    out: list[dict] = []
    for r in rows:
        st = r.get("state") or ""
        co = r.get("county") or ""
        tr = r.get("tract") or ""
        if not (st and co and tr):
            continue
        geoid = f"{st}{co}{tr}"
        if geoid in wanted:
            out.append(r)
    return out


def _geo_id_from_row(
    row: dict, api_level: str, parent_geo,
) -> tuple[str, str]:
    """Reconstruct a geo_id + human display_name from a response row.

    For admin-level rows we can reuse the parent geo's display_name
    when the row's identifiers match — otherwise we leave
    display_name blank and let variable.txt + the variable_id carry
    the meaning.
    """
    def _p_name_if_matches(expected_id: str) -> str:
        if parent_geo and parent_geo.geo_id == expected_id:
            return parent_geo.display_name
        return ""

    if api_level == "tract":
        st = row.get("state") or ""
        co = row.get("county") or ""
        tr = row.get("tract") or ""
        return f"{st}{co}{tr}", ""
    if api_level == "county":
        st = row.get("state") or ""
        co = row.get("county") or ""
        geoid = f"{st}{co}"
        return geoid, _p_name_if_matches(geoid)
    if api_level == "state":
        st = row.get("state") or ""
        return st, _p_name_if_matches(st)
    if api_level == "us":
        # Census uses us=1 as the single-row nation response.
        us_id = row.get("us") or "1"
        return us_id, parent_geo.display_name if parent_geo else "United States"
    if api_level == "msa":
        msa_id = (
            row.get("metropolitan statistical area/"
                    "micropolitan statistical area")
            or row.get("cbsa") or ""
        )
        name = parent_geo.display_name if parent_geo else ""
        return msa_id, name
    if api_level == "place":
        st = row.get("state") or ""
        pl = row.get("place") or ""
        geoid = f"{st}{pl}"
        return geoid, _p_name_if_matches(geoid)
    # Unknown level — fall back to the parent geo's identifiers.
    if parent_geo:
        return parent_geo.geo_id, parent_geo.display_name
    return "", ""


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return f"{x:.6g}"
    return str(x)


def _lookup_labels(
    var_ids: set[tuple[str, str]],
    metadata_db: Optional[sqlite3.Connection],
) -> dict[tuple[str, str], str]:
    """Return {(variable_id, dataset) -> plain_english_label}.

    Unknown entries get empty-string labels so the sidecar still
    lists them rather than silently dropping.
    """
    out: dict[tuple[str, str], str] = {}
    if metadata_db is None or not var_ids:
        return {k: "" for k in var_ids}
    for vid, dataset in var_ids:
        try:
            row = metadata_db.execute(
                "SELECT label_natural FROM variables "
                "WHERE variable_id = ? AND dataset = ? "
                "  AND label_natural IS NOT NULL AND label_natural != '' "
                "ORDER BY year DESC LIMIT 1",
                (vid, dataset),
            ).fetchone()
        except sqlite3.Error as e:
            logger.debug(
                "derived_data: label lookup failed for %s/%s: %s",
                vid, dataset, e,
            )
            row = None
        label = ""
        if row is not None:
            label = row[0] if isinstance(row, tuple) else row["label_natural"]
            label = label or ""
        out[(vid, dataset)] = label
    return out
