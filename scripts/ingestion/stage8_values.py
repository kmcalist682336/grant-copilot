"""Stage 8: Data Ingestion — core fetch and unpivot logic.

For each (table, geo_level) work item:
  1. Construct the correct Census API call from geography_levels metadata
  2. Cache the raw wide-format response to disk
  3. Unpivot wide → long format
  4. Write long-format JSONL ready for BigQuery load

Geography levels fetched (in order of attempt):
  tract            → for=tract:*&in=state:13+county:*
  county           → for=county:*&in=state:13
  state            → for=state:13  (Georgia only)
  state_national   → for=state:*   (all 50 states + DC + territories)
  us               → for=us:1      (single national aggregate row)
  place            → for=place:*&in=state:13
  zcta             → for=zip+code+tabulation+area:* (filtered to GA ZCTAs post-fetch)
  msa              → for=metropolitan+statistical+area/micropolitan+statistical+area:*
                     (kept nationally for cross-metro comparison)
  school_dist      → for=school+district+(unified):*&in=state:13

Long-format schema (one row per variable × geography):
  source          str   e.g. "acs5"
  year            int   e.g. 2022
  dataset_path    str   e.g. "acs/acs5"
  table_id        str   e.g. "B17001"
  variable_id     str   e.g. "B17001_002E"
  geo_level       str   e.g. "tract"
  geo_id          str   full concatenated GEOID e.g. "13121001100"
  state_fips      str   "13" (null for MSA, ZCTA)
  county_fips     str   "121" (null for county and above)
  tract_ce        str   "001100" (null for county and above)
  place_fips      str   "04000" (null for non-place)
  zcta5           str   "30318" (null for non-ZCTA)
  msa_code        str   "12060" (null for non-MSA)
  estimate        float
  moe             float  (null if not available)
  annotation      str    (null if no Census quality flag)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target geography levels
# ---------------------------------------------------------------------------

# Maps our internal geo_level name → Census API geography name(s)
# Multiple Census names can map to the same internal name
# Order matters — determines fetch priority
TARGET_GEO_LEVELS = [
    "tract",
    "county",
    "state",                # Georgia only
    "state_national",       # all 50 states + DC + territories
    "us",                   # national aggregate (single row per variable)
    "place",
    "zcta",
    "msa",
    "school_district_unified",
]

# Census API geography name → internal geo_level name
# Note: state_national reuses the Census name "state" — handled
# specially in resolve_supported_geo_levels
CENSUS_NAME_TO_GEO_LEVEL: dict[str, str] = {
    "tract":                                                   "tract",
    "county":                                                  "county",
    "state":                                                   "state",
    "place":                                                   "place",
    "zip code tabulation area":                                "zcta",
    "metropolitan statistical area/micropolitan statistical area": "msa",
    "school district (unified)":                               "school_district_unified",
    "us":                                                      "us",
}

# Internal geo_level → Census API `for=` clause
GEO_LEVEL_FOR_CLAUSE: dict[str, str] = {
    "tract":                    "tract:*",
    "county":                   "county:*",
    "state":                    "state:13",        # Georgia only
    "state_national":           "state:*",         # all states
    "us":                       "us:1",
    "place":                    "place:*",
    "zcta":                     "zip+code+tabulation+area:*",
    "msa":                      "metropolitan+statistical+area%2Fmicropolitan+statistical+area:*",
    "school_district_unified":  "school+district+%28unified%29:*",
}

# Internal geo_level → Census API `in=` clause (None = no in clause)
GEO_LEVEL_IN_CLAUSE: dict[str, Optional[str]] = {
    "tract":                    "state:13+county:*",
    "county":                   "state:13",
    "state":                    None,
    "state_national":           None,
    "us":                       None,
    "place":                    "state:13",
    "zcta":                     None,   # national fetch, filter post-response
    "msa":                      None,   # keep all MSAs nationally
    "school_district_unified":  "state:13",
}

# geo_levels that return national data and need post-fetch filtering
NATIONAL_GEO_LEVELS = {"zcta", "msa"}

# Georgia FIPS code
GEORGIA_FIPS = "13"

# ---------------------------------------------------------------------------
# Peer metro definitions
# ---------------------------------------------------------------------------

# Default peer metros for cross-city comparison
# Format: {cbsa_code: (metro_name, state_fips_list)}
# state_fips_list covers all states the metro spans (most span 1-2)
DEFAULT_PEER_METROS: dict[str, tuple[str, list[str]]] = {
    "12060": ("Atlanta-Sandy Springs-Alpharetta, GA",     ["13"]),
    "16740": ("Charlotte-Concord-Gastonia, NC-SC",        ["37", "45"]),
    "34980": ("Nashville-Davidson-Murfreesboro-Franklin, TN", ["47"]),
    "26420": ("Houston-The Woodlands-Sugar Land, TX",     ["48"]),
    "19100": ("Dallas-Fort Worth-Arlington, TX",          ["48"]),
    "38060": ("Phoenix-Mesa-Chandler, AZ",                ["04"]),
    "19740": ("Denver-Aurora-Lakewood, CO",               ["08"]),
    "33460": ("Minneapolis-St. Paul-Bloomington, MN-WI",  ["27", "55"]),
    "12580": ("Baltimore-Columbia-Towson, MD",            ["24"]),
    "28140": ("Kansas City, MO-KS",                      ["29", "20"]),
    "41860": ("San Francisco-Oakland-Berkeley, CA",       ["06"]),
    "33100": ("Miami-Fort Lauderdale-Pompano Beach, FL",  ["12"]),
    "35380": ("New Orleans-Metairie, LA",                 ["22"]),
    "40060": ("Richmond, VA",                            ["51"]),
    "16980": ("Chicago-Naperville-Elgin, IL-IN-WI",      ["17", "18", "55"]),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeoLevelSpec:
    """Resolved specification for fetching one geography level."""
    geo_level: str              # our internal name e.g. "tract"
    for_clause: str             # Census API for= value
    in_clause: Optional[str]    # Census API in= value (None if omitted)
    is_national: bool           # True if response needs post-fetch filtering
    # Peer metro context — set when fetching tract/county for a peer metro
    peer_cbsa_code: Optional[str] = None   # e.g. "16740"
    peer_state_fips: Optional[str] = None  # e.g. "37" (one state at a time)
    peer_metro_name: Optional[str] = None  # e.g. "Charlotte-Concord..."

    @property
    def cache_key(self) -> str:
        """Unique key for disk caching — includes peer metro if present."""
        if self.peer_cbsa_code and self.peer_state_fips:
            return f"peer_{self.peer_cbsa_code}_{self.peer_state_fips}_{self.geo_level}"
        return self.geo_level


def build_peer_metro_specs(
    cbsa_codes: list[str],
    geo_levels: list[str],
    geography_levels: list[dict],
) -> list[GeoLevelSpec]:
    """Build GeoLevelSpecs for peer metro tract and county fetches.

    For each peer metro × state combination, generates:
      - tract: for=tract:*&in=state:{sf}+county:*
      - county: for=county:*&in=state:{sf}

    Only generates specs for geo levels the dataset actually supports.

    Args:
        cbsa_codes: List of CBSA codes to fetch (e.g. ["16740", "34980"])
        geo_levels: Which levels to fetch for peers (typically ["tract", "county"])
        geography_levels: Stage 1 geography_levels for this table

    Returns:
        List of GeoLevelSpec with peer metro context set
    """
    supported = {g["name"] for g in geography_levels}
    specs = []

    for cbsa_code in cbsa_codes:
        metro_info = DEFAULT_PEER_METROS.get(cbsa_code)
        if not metro_info:
            logger.warning("Unknown peer metro CBSA: %s — skipping", cbsa_code)
            continue

        metro_name, state_fips_list = metro_info

        for state_fips in state_fips_list:
            for geo_level in geo_levels:
                # Only generate if the dataset supports this level
                if geo_level == "tract" and "tract" not in supported:
                    continue
                if geo_level == "county" and "county" not in supported:
                    continue

                if geo_level == "tract":
                    for_clause = "tract:*"
                    in_clause = f"state:{state_fips}+county:*"
                elif geo_level == "county":
                    for_clause = "county:*"
                    in_clause = f"state:{state_fips}"
                else:
                    continue

                specs.append(GeoLevelSpec(
                    geo_level=geo_level,
                    for_clause=for_clause,
                    in_clause=in_clause,
                    is_national=False,
                    peer_cbsa_code=cbsa_code,
                    peer_state_fips=state_fips,
                    peer_metro_name=metro_name,
                ))

    return specs


@dataclass
class FetchResult:
    """Result of one (table, geo_level) fetch attempt."""
    geo_level: str
    status: str             # success | failed_no_data | failed_api_error | skipped
    row_count: int = 0
    error_message: Optional[str] = None
    raw_path: Optional[Path] = None
    jsonl_path: Optional[Path] = None


@dataclass
class LongFormatRow:
    """One row in the long-format output."""
    source: str
    year: int
    dataset_path: str
    table_id: str
    variable_id: str
    geo_level: str
    geo_id: str
    state_fips: Optional[str]
    county_fips: Optional[str]
    tract_ce: Optional[str]
    place_fips: Optional[str]
    zcta5: Optional[str]
    msa_code: Optional[str]
    estimate: Optional[float]
    moe: Optional[float]
    annotation: Optional[str]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "year": self.year,
            "dataset_path": self.dataset_path,
            "table_id": self.table_id,
            "variable_id": self.variable_id,
            "geo_level": self.geo_level,
            "geo_id": self.geo_id,
            "state_fips": self.state_fips,
            "county_fips": self.county_fips,
            "tract_ce": self.tract_ce,
            "place_fips": self.place_fips,
            "zcta5": self.zcta5,
            "msa_code": self.msa_code,
            "estimate": self.estimate,
            "moe": self.moe,
            "annotation": self.annotation,
        }


# ---------------------------------------------------------------------------
# Geography level resolution
# ---------------------------------------------------------------------------

def build_msa_for_clause(cbsa_ids: Optional[list[str]]) -> str:
    """Build the MSA `for=` clause for the Census API.

    If ``cbsa_ids`` is None or empty, returns the nationwide wildcard
    (all 925 CBSAs). Otherwise returns a comma-separated list of the
    specified CBSA GEOIDs — this is how we scope to e.g. top-100-by-pop
    + all GA CBSAs.
    """
    base = "metropolitan+statistical+area%2Fmicropolitan+statistical+area"
    if not cbsa_ids:
        return f"{base}:*"
    return f"{base}:" + ",".join(sorted(set(cbsa_ids)))


def select_top_cbsas(
    conn,
    top_n: int = 100,
    include_state_fips: Optional[list[str]] = None,
) -> list[str]:
    """Return the union of (top N CBSAs by population) + (all CBSAs that
    contain at least one county in any state in ``include_state_fips``).

    Requires ``admin_geographies.population`` to be populated for CBSAs
    (see ``scripts.geography.populate_cbsa_population``). Returns an
    empty list if the gazetteer doesn't have population data — caller
    should fall back to the nationwide wildcard and log a warning.
    """
    # Top-N MSAs by population. Micros are excluded — any metro ranked
    # top-100 is by definition an MSA (micros max out well below).
    top_rows = conn.execute("""
        SELECT geoid FROM admin_geographies
        WHERE geo_type = 'msa'
          AND population IS NOT NULL
          AND population > 0
        ORDER BY population DESC
        LIMIT ?
    """, (top_n,)).fetchall()
    top_ids = {r[0] for r in top_rows}

    state_ids: set[str] = set()
    if include_state_fips:
        placeholders = ",".join("?" * len(include_state_fips))
        rows = conn.execute(
            f"SELECT DISTINCT cbsa_geoid FROM cbsa_counties "
            f"WHERE state_fips IN ({placeholders})",
            include_state_fips,
        ).fetchall()
        state_ids = {r[0] for r in rows}

    return sorted(top_ids | state_ids)


def resolve_supported_geo_levels(
    geography_levels: list[dict],
    msa_cbsa_ids: Optional[list[str]] = None,
) -> list[GeoLevelSpec]:
    """Given Stage 1 geography_levels, return specs for all target levels
    that this dataset actually supports.

    state_national and us are handled specially:
    - state_national is emitted whenever 'state' is supported
    - us is emitted whenever 'us' is supported
    Both use the same Census geography name as their base level but with
    different for= clauses, so they can't be in CENSUS_NAME_TO_GEO_LEVEL
    directly without ambiguity.

    Args:
        geography_levels: The geography_levels list from stage1_metadata.json
        msa_cbsa_ids: Optional list of CBSA GEOIDs to restrict MSA fetches
            to (e.g. top-100 by population + all GA CBSAs). When None, the
            MSA for-clause is the nationwide wildcard.

    Returns:
        List of GeoLevelSpec in fetch priority order
    """
    supported_census_names = {g["name"] for g in geography_levels}

    specs = []
    for geo_level in TARGET_GEO_LEVELS:
        # state_national and us are derived from base Census names
        if geo_level == "state_national":
            if "state" in supported_census_names:
                specs.append(GeoLevelSpec(
                    geo_level="state_national",
                    for_clause=GEO_LEVEL_FOR_CLAUSE["state_national"],
                    in_clause=None,
                    is_national=False,  # returns all states, no filtering needed
                ))
            continue

        if geo_level == "us":
            if "us" in supported_census_names:
                specs.append(GeoLevelSpec(
                    geo_level="us",
                    for_clause=GEO_LEVEL_FOR_CLAUSE["us"],
                    in_clause=None,
                    is_national=False,
                ))
            continue

        # Standard levels — look up Census name
        census_name = None
        for cname, internal in CENSUS_NAME_TO_GEO_LEVEL.items():
            if internal == geo_level and cname in supported_census_names:
                census_name = cname
                break

        if census_name is None:
            continue

        # MSA geo level supports a per-run CBSA filter to skip the ~900
        # CBSAs that aren't in our peer-comparison scope.
        if geo_level == "msa" and msa_cbsa_ids:
            for_clause = build_msa_for_clause(msa_cbsa_ids)
            # With a specific CBSA list the response only contains those
            # CBSAs, so we no longer treat it as "national needs filtering".
            is_national = False
        else:
            for_clause = GEO_LEVEL_FOR_CLAUSE[geo_level]
            is_national = geo_level in NATIONAL_GEO_LEVELS

        specs.append(GeoLevelSpec(
            geo_level=geo_level,
            for_clause=for_clause,
            in_clause=GEO_LEVEL_IN_CLAUSE[geo_level],
            is_national=is_national,
        ))

    return specs


# ---------------------------------------------------------------------------
# API response parsing
# ---------------------------------------------------------------------------

def parse_census_response(
    raw_response: list,
    table_id: str,
) -> tuple[list[str], list[list]]:
    """Split a raw Census API response into headers and data rows.

    The Census API returns a list where element [0] is the header row
    and elements [1:] are data rows.

    Returns:
        (headers, data_rows)
    """
    if not raw_response or len(raw_response) < 2:
        return [], []

    headers = raw_response[0]
    data_rows = raw_response[1:]
    return headers, data_rows


def classify_variable(var_id: str, headers: list[str]) -> str:
    """Classify a variable column as estimate, moe, or annotation."""
    if var_id.endswith("_E") or var_id.endswith("E"):
        # Could be estimate suffix pattern
        pass
    upper = var_id.upper()
    if upper.endswith("MA") or upper.endswith("_MA"):
        return "moe_annotation"
    if upper.endswith("EA") or upper.endswith("_EA"):
        return "estimate_annotation"
    if upper.endswith("NA") or upper.endswith("_NA"):
        return "annotation"
    if upper.endswith("M") or upper.endswith("_M"):
        return "moe"
    if upper.endswith("E") or upper.endswith("_E"):
        return "estimate"
    # Decennial estimates use the _N / N suffix (no MOE). Annotation ('NA')
    # is handled above so this matches only bare estimates.
    if upper.endswith("N") or upper.endswith("_N"):
        return "estimate"
    return "other"


import re as _re

# Known geography column names returned by the Census API. Referenced by
# both unpivot_response (to skip) and validate_variable_coverage (to
# keep the "actual" set honest when no table-prefix filter applies).
GEO_COLS = {
    "state", "county", "tract", "place", "NAME", "name", "GEO_ID",
    "zip code tabulation area",
    "metropolitan statistical area/micropolitan statistical area",
    "school district (unified)", "school district (elementary)",
    "school district (secondary)",
    "block group", "congressional district",
    "us", "region", "division",
    "county subdivision", "consolidated city",
    "public use microdata area",
    "american indian area/alaska native area/hawaiian home land",
}

# Column-name suffixes we always exclude from value columns.
# Uppercase for case-insensitive match.
_NON_VALUE_SUFFIXES = (
    "_LABEL", "LABEL",        # descriptive label columns (ABS, ecn)
    "_F",                     # flag (ABS, economic censuses)
    "_FLAG",
    "_SA",                    # annotated standard error
    "_EA", "EA",              # ACS estimate annotation
    "_MA", "MA",              # ACS MOE annotation
    "_NA", "NA",              # decennial annotation
)

# Static column-name blacklist: classification / grouping cols that aren't
# values but happen to be numeric-looking (NAICS codes, grouping IDs).
_NON_VALUE_EXACT_NAMES = frozenset({
    "GEO_ID", "NAME",
    # NAICS variants
    "NAICS2022", "NAICS2017", "NAICS2012", "NAICS2007", "NAICS2002", "NAICS",
    # Common grouping dimensions in ABS / economic
    "SEX", "RACE_GROUP", "ETH_GROUP", "VET_GROUP",
    "EMPSZES", "EMPSZFI", "LFO", "YEAR", "YIBSZFI",
})

# Pattern for ACS MOE columns: TABLE_NUMM (e.g. B01001_001M).
_ACS_MOE_RE = _re.compile(r"^[A-Z]+\d+_\d+M$")


def _is_estimate_suffix(var_id_upper: str) -> bool:
    """True when a variable ID is a value column (not metadata / annotation).

    Design: negative list of known non-value patterns rather than a positive
    whitelist of estimate suffixes. This lets new datasets (ABS, economic,
    decennial variants) "just work" without per-dataset shims. Values that
    slip through as non-numeric are caught at row-parse time in
    ``unpivot_response``.

    Handled explicitly:
      - ACS annotations (EA, MA) and MOEs (TABLE_###M pattern) → False
      - Decennial annotations (NA) → False
      - ABS flags (_F), labels (_LABEL, LABEL), standard errors (_S, _SA) → False
      - Classification dimensions (NAICS, SEX, RACE_GROUP, …) → False
    Everything else returns True and the unpivot will attempt to parse.
    """
    up = var_id_upper
    if up in _NON_VALUE_EXACT_NAMES:
        return False
    # Exclude _S (standard error) only when it's a real suffix marker, not
    # part of a word (e.g., "RECEIPTS" ends in "S" but that's not a _S var).
    # Require an underscore before for ABS-style naming.
    if up.endswith("_S"):
        return False
    for suffix in _NON_VALUE_SUFFIXES:
        if up.endswith(suffix):
            return False
    # ACS MOE pattern — structural match so we don't false-positive on
    # variables that just happen to end in M.
    if _ACS_MOE_RE.match(up):
        return False
    return True


def _moe_column_for_estimate(est_col: str, headers_set: set) -> Optional[str]:
    """Return the matching margin-of-error column name if one exists.

    ACS estimates end in 'E' and pair with an 'M' column (e.g., B01001_001E
    ↔ B01001_001M). Decennial estimates end in 'N' and have no MOE. For
    any other convention (ABS, economic) there is no standard MOE column.
    """
    if not est_col:
        return None
    if est_col[-1].upper() == "E":
        candidate = est_col[:-1] + "M"
        return candidate if candidate in headers_set else None
    return None


def _column_looks_numeric(
    col: str,
    headers: list[str],
    data_rows: list,
    sample_size: int = 20,
) -> bool:
    """Sample up to `sample_size` non-null values from a column and test
    whether any parse as float. Used to reject string-like columns (labels,
    codes with embedded commas) that slip through the name-based filter.

    Census API returns numeric values as strings (e.g. '8220274') so we
    always go through float() rather than type inspection.
    """
    try:
        col_idx = headers.index(col)
    except ValueError:
        return False
    seen = 0
    for row in data_rows:
        if seen >= sample_size:
            break
        if col_idx >= len(row):
            continue
        val = row[col_idx]
        if val is None or val == "":
            continue
        seen += 1
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            # Every row so far non-numeric — but we might just have hit
            # a stretch of nulls; keep looking until sample_size of
            # actual values have been probed.
            return False
    # All sampled values were null/empty — can't tell, give benefit of doubt.
    return seen == 0


def extract_geo_identifiers(
    row_dict: dict,
    geo_level: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract geo_id and component FIPS codes from a data row dict.

    Returns:
        (geo_id, state_fips, county_fips, tract_ce, place_fips, zcta5, msa_code)
    """
    state = row_dict.get("state", "").zfill(2) if row_dict.get("state") else None
    county = row_dict.get("county", "").zfill(3) if row_dict.get("county") else None
    tract = row_dict.get("tract", "").zfill(6) if row_dict.get("tract") else None
    place = row_dict.get("place", "").zfill(5) if row_dict.get("place") else None
    zcta = row_dict.get("zip code tabulation area") or row_dict.get("zip+code+tabulation+area")
    msa = row_dict.get(
        "metropolitan statistical area/micropolitan statistical area"
    ) or row_dict.get("metropolitan+statistical+area%2Fmicropolitan+statistical+area")

    # Build full concatenated geo_id
    if geo_level == "tract" and state and county and tract:
        geo_id = f"{state}{county}{tract}"
    elif geo_level == "county" and state and county:
        geo_id = f"{state}{county}"
    elif geo_level in ("state", "state_national") and state:
        geo_id = state
    elif geo_level == "us":
        geo_id = "1"
    elif geo_level == "place" and state and place:
        geo_id = f"{state}{place}"
    elif geo_level == "zcta" and zcta:
        geo_id = str(zcta).zfill(5)
    elif geo_level == "msa" and msa:
        geo_id = str(msa).zfill(5)
    elif geo_level == "school_district_unified":
        sd = (row_dict.get("school district (unified)", "") or "").zfill(5)
        geo_id = f"{state}{sd}" if state else sd
    else:
        parts = [v for v in [state, county, tract, place] if v]
        geo_id = "".join(parts) if parts else "unknown"

    return (
        geo_id,
        state if geo_level not in ("zcta", "msa", "us") else None,
        county if geo_level == "tract" else None,
        tract if geo_level == "tract" else None,
        place if geo_level == "place" else None,
        str(zcta).zfill(5) if zcta else None,
        str(msa).zfill(5) if msa else None,
    )


def unpivot_response(
    raw_response: list,
    year: int,
    dataset_path: str,
    table_id: str,
    geo_level: str,
    ga_zcta_set: Optional[set[str]] = None,
    peer_cbsa_code: Optional[str] = None,
    stage1_variables: Optional[list] = None,
) -> list[LongFormatRow]:
    """Unpivot a wide Census API response into long-format rows.

    Variable identification is driven by the Stage 1 metadata variable list
    rather than regex pattern matching against the table_id. This handles
    all Census naming conventions (B-tables, S-tables, DP-tables, RHFS, etc.)
    correctly.

    Args:
        raw_response: Raw Census API response (list of lists)
        year: Census data year
        dataset_path: e.g. "acs/acs5"
        table_id: e.g. "B17001"
        geo_level: Our internal geo level name
        ga_zcta_set: If provided, filter ZCTA rows to only these ZCTAs
        peer_cbsa_code: If set, tag rows with this CBSA code in msa_code field
        stage1_variables: Variable list from stage1_metadata.json — either
                         list of strings ["B17001_001E", ...] or list of dicts
                         [{"var_id": "B17001_001E"}, ...]. When provided, only
                         these variables are processed. When None, falls back
                         to extracting all non-geography columns from the response.

    Returns:
        List of LongFormatRow objects
    """
    headers, data_rows = parse_census_response(raw_response, table_id)
    if not headers or not data_rows:
        return []

    # Derive source label from dataset_path
    source = dataset_path.replace("/", "_").lstrip("_")

    # Build the set of known variable IDs from Stage 1 metadata
    # This is the authoritative source — no regex guessing
    known_var_ids: set[str] = set()
    if stage1_variables:
        for var in stage1_variables:
            if isinstance(var, dict):
                vid = var.get("var_id", "")
            elif isinstance(var, str):
                vid = var
            else:
                continue
            if vid:
                known_var_ids.add(vid)

    # Build set of headers present in the response for fast lookup
    header_set = set(headers)

    # Identify estimate variables and their MOE pairs
    # Strategy: if we have Stage 1 metadata, use it as the authoritative list.
    # If not, fall back to treating all non-geography columns as variables.
    estimate_vars: dict[str, Optional[str]] = {}

    if known_var_ids:
        # Stage 1 metadata available — use it to identify estimate columns.
        # Estimate suffix is 'E' for ACS and 'N' for decennial; MOE only
        # applies to ACS.
        for vid in known_var_ids:
            if not _is_estimate_suffix(vid.upper()):
                continue
            if vid in header_set:
                estimate_vars[vid] = _moe_column_for_estimate(vid, header_set)
    else:
        # No Stage 1 metadata — fallback: scan headers directly. For
        # unfamiliar datasets (ABS, economic) the name-based filter is
        # permissive; we follow up with a content probe that rejects
        # columns whose first populated value isn't numeric.
        logger.warning(
            "stage1_variables missing for %s (%s / %s) — using header-based "
            "fallback with content probe",
            table_id, dataset_path, geo_level,
        )
        for col in headers:
            if col in GEO_COLS or col.lower() in GEO_COLS:
                continue
            if not _is_estimate_suffix(col.upper()):
                continue
            if not _column_looks_numeric(col, headers, data_rows):
                continue
            estimate_vars[col] = _moe_column_for_estimate(col, header_set)

    rows = []
    for data_row in data_rows:
        row_dict = dict(zip(headers, data_row))

        (geo_id, state_fips, county_fips, tract_ce,
         place_fips, zcta5, msa_code) = extract_geo_identifiers(
            row_dict, geo_level
        )

        # Filter ZCTAs to Georgia if filter set provided
        if geo_level == "zcta" and ga_zcta_set is not None:
            if zcta5 not in ga_zcta_set:
                continue

        for est_col, moe_col in estimate_vars.items():
            raw_est = row_dict.get(est_col)
            raw_moe = row_dict.get(moe_col) if moe_col else None

            # Parse estimate
            estimate = None
            annotation = None
            if raw_est is not None:
                try:
                    estimate = float(raw_est)
                except (ValueError, TypeError):
                    annotation = str(raw_est) if raw_est else None

            # Parse MOE
            moe = None
            if raw_moe is not None:
                try:
                    moe = float(raw_moe)
                    if moe < 0:
                        moe = None
                except (ValueError, TypeError):
                    pass

            rows.append(LongFormatRow(
                source=source,
                year=year,
                dataset_path=dataset_path,
                table_id=table_id,
                variable_id=est_col,
                geo_level=geo_level,
                geo_id=geo_id,
                state_fips=state_fips,
                county_fips=county_fips,
                tract_ce=tract_ce,
                place_fips=place_fips,
                zcta5=zcta5,
                msa_code=peer_cbsa_code if peer_cbsa_code else msa_code,
                estimate=estimate,
                moe=moe,
                annotation=annotation,
            ))

    return rows


# ---------------------------------------------------------------------------
# ZCTA filter helpers
# ---------------------------------------------------------------------------

def load_georgia_zctas(raw_dir: Path) -> set[str]:
    """Load the set of Georgia ZCTA codes from the ARC statewide GeoJSON.

    Filters using the raw API response data rather than the gazetteer,
    keeping this extensible for other states later.

    Falls back to an empty set (no filtering) if the file isn't found,
    which means all ZCTAs are kept.
    """
    # Try ARC statewide GeoJSON first
    arc_files = list((raw_dir / "geo" / "raw").glob(
        "GeographicBoundaries2024statewide_*.geojson"
    ))
    if arc_files:
        try:
            import json as _json
            with arc_files[0].open() as f:
                data = _json.load(f)
            zctas = set()
            for feat in data["features"]:
                props = feat.get("properties", {})
                if props.get("SumLev") == "ZCTA":
                    geoid = str(props.get("GEOID", "")).strip()
                    if geoid:
                        zctas.add(geoid.zfill(5))
            logger.info("Loaded %d Georgia ZCTAs from ARC file", len(zctas))
            return zctas
        except Exception as e:
            logger.warning("Could not load ZCTAs from ARC file: %s", e)

    # Fallback: ZCTA-place relationship file
    rel_file = raw_dir / "geo" / "raw" / "zcta_place_rel_2020.txt"
    if rel_file.exists():
        try:
            import csv as _csv
            zctas = set()
            with rel_file.open() as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    if row.get("STATE", "").zfill(2) == GEORGIA_FIPS:
                        zcta = row.get("ZCTA5", "").strip()
                        if zcta:
                            zctas.add(zcta.zfill(5))
            logger.info("Loaded %d Georgia ZCTAs from ZCTA-place rel file", len(zctas))
            return zctas
        except Exception as e:
            logger.warning("Could not load ZCTAs from rel file: %s", e)

    logger.warning(
        "No Georgia ZCTA filter file found — keeping all ZCTAs from API response"
    )
    return set()  # Empty set = no filtering


# ---------------------------------------------------------------------------
# Variable coverage validation
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Result of comparing expected vs actual variables in an API response."""
    table_id: str
    geo_level: str
    year: int
    dataset_path: str
    expected_count: int
    actual_count: int
    missing_vars: list[str]       # in stage1 but not in API response
    extra_vars: list[str]         # in API response but not in stage1
    coverage_pct: float           # actual / expected * 100

    @property
    def is_complete(self) -> bool:
        return len(self.missing_vars) == 0

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "geo_level": self.geo_level,
            "year": self.year,
            "dataset_path": self.dataset_path,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "missing_vars": self.missing_vars,
            "extra_vars": self.extra_vars,
            "coverage_pct": round(self.coverage_pct, 2),
        }


def validate_variable_coverage(
    raw_response: list,
    table_id: str,
    stage1_variables: list[dict],
    geo_level: str,
    year: int,
    dataset_path: str,
) -> CoverageReport:
    """Compare expected variables (from Stage 1) against actual API response.

    Args:
        raw_response: Raw Census API response (list of lists, first row is headers)
        table_id: e.g. "B17001"
        stage1_variables: The 'variables' list from stage1_metadata.json
        geo_level: e.g. "tract"
        year: Census data year
        dataset_path: e.g. "acs/acs5"

    Returns:
        CoverageReport with missing/extra variable lists and coverage percentage
    """
    # Build expected set — value columns from Stage 1 metadata.
    # The per-table metadata file only contains this table's variables,
    # so no table-prefix filter is needed (and would break datasets like
    # ABS/CBP/geoinfo whose variables don't carry the table_id prefix).
    expected_vars = set()
    for var in stage1_variables:
        # Handle both formats:
        #   dict format: {"var_id": "B17001_001E", ...}
        #   string format: "B17001_001E"
        if isinstance(var, dict):
            var_id = var.get("var_id", "")
        elif isinstance(var, str):
            var_id = var
        else:
            continue

        if not var_id:
            continue
        if _is_estimate_suffix(var_id.upper()):
            expected_vars.add(var_id)

    # Build actual set from response headers — same permissive test.
    if not raw_response:
        actual_vars = set()
    else:
        headers = raw_response[0]
        actual_vars = set()
        for col in headers:
            if col in GEO_COLS or col.lower() in GEO_COLS:
                continue
            if _is_estimate_suffix(col.upper()):
                actual_vars.add(col)

    missing = sorted(expected_vars - actual_vars)
    extra = sorted(actual_vars - expected_vars)

    coverage_pct = (
        len(actual_vars) / len(expected_vars) * 100
        if expected_vars else 100.0
    )

    return CoverageReport(
        table_id=table_id,
        geo_level=geo_level,
        year=year,
        dataset_path=dataset_path,
        expected_count=len(expected_vars),
        actual_count=len(actual_vars),
        missing_vars=missing,
        extra_vars=extra,
        coverage_pct=coverage_pct,
    )

def write_jsonl(rows: list[LongFormatRow], path: Path) -> int:
    """Write long-format rows to a JSONL file. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    return len(rows)


def append_jsonl(rows: list[LongFormatRow], path: Path) -> int:
    """Append long-format rows to an existing JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# JSONL shard path helper
# ---------------------------------------------------------------------------

def jsonl_shard_path(
    curated_dir: Path,
    year: int,
    dataset_path: str,
    geo_level: str,
) -> Path:
    """Return the JSONL shard path for a (year, dataset, geo_level) combination.

    Sharding by (year, dataset, geo_level) keeps files manageable and
    allows partial BigQuery loads. Each shard contains all tables for
    that combination.

    Example:
        curated/bigquery_loads/census_values/
            2022__acs_acs5__tract.jsonl
            2022__acs_acs5__county.jsonl
            2020__dec_pl__tract.jsonl
    """
    dataset_slug = dataset_path.replace("/", "_")
    filename = f"{year}__{dataset_slug}__{geo_level}.jsonl"
    return curated_dir / "bigquery_loads" / "census_values" / filename
