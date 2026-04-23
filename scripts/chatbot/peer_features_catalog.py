"""Peer-feature catalog: 108 ACS5 features organized across 19 axes.

Each entry names a Census table, the variable(s) that feed it, and a
`derivation` code telling the fetcher how to combine them:

    raw         → value = vars[0]
    ratio       → value = vars[0] / vars[1]
    sum_ratio   → value = sum(vars[:-1]) / vars[-1]

The `tools/fetch_peer_features` driver reads this catalog, issues one
Census API call per unique (table, geo_level) pair, then applies each
feature's derivation to the returned rows. Output lands in
`data/metadata/peer_features.sqlite`.

Features are grouped into AXES so PeerRetriever can return
axis-specific peer sets ("econ peers", "demo peers", ...) rather than
a single collapsed score.

Keep this file editable — growing the catalog is a design tuning
knob. Adding a feature: define it, add its name to one or more axes.
Nothing else needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Derivation = Literal["raw", "ratio", "sum_ratio"]


@dataclass(frozen=True)
class Feature:
    """One peer-index feature."""

    name: str                       # snake_case identifier
    description: str
    table: str                      # Census table ID
    variables: tuple[str, ...]      # ACS5 variable IDs (order matters)
    derivation: Derivation
    # True if we expect this feature to be NULL/suppressed for small
    # places (<5-10k pop). The peer ranker treats these as "ignore
    # this feature for this candidate" instead of failing.
    sparse_small_places: bool = False


# ---------------------------------------------------------------------------
# Feature catalog — 108 entries. See docs/END_TO_END.md "Phase 3" section.
# ---------------------------------------------------------------------------

FEATURES: dict[str, Feature] = {}


def _add(*features: Feature) -> None:
    for f in features:
        if f.name in FEATURES:
            raise ValueError(f"duplicate feature: {f.name}")
        FEATURES[f.name] = f


# Axis 1: Size + household base (5)
_add(
    Feature("total_population", "Total population",
            "B01003", ("B01003_001E",), "raw"),
    Feature("total_households", "Total households",
            "B11001", ("B11001_001E",), "raw"),
    Feature("pct_non_family_households",
            "% non-family households",
            "B11001", ("B11001_007E", "B11001_001E"), "ratio"),
    Feature("avg_household_size", "Average household size",
            "B25010", ("B25010_001E",), "raw"),
    Feature("pct_group_quarters", "% population in group quarters",
            "B26001", ("B26001_001E", "B01003_001E"), "ratio"),
)

# Axis 2: Age structure (9)
_add(
    Feature("median_age", "Median age", "B01002", ("B01002_001E",), "raw"),
    Feature("pct_under_5", "% under age 5",
            "B01001", ("B01001_003E", "B01001_027E", "B01001_001E"),
            "sum_ratio"),
    Feature("pct_5_to_17", "% age 5–17", "B01001",
            ("B01001_004E", "B01001_005E", "B01001_006E",
             "B01001_028E", "B01001_029E", "B01001_030E",
             "B01001_001E"),
            "sum_ratio"),
    Feature("pct_18_to_24", "% age 18–24", "B01001",
            ("B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
             "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",
             "B01001_001E"),
            "sum_ratio"),
    Feature("pct_25_to_44", "% age 25–44", "B01001",
            ("B01001_011E", "B01001_012E", "B01001_013E", "B01001_014E",
             "B01001_035E", "B01001_036E", "B01001_037E", "B01001_038E",
             "B01001_001E"),
            "sum_ratio"),
    Feature("pct_45_to_64", "% age 45–64", "B01001",
            ("B01001_015E", "B01001_016E", "B01001_017E", "B01001_018E",
             "B01001_019E",
             "B01001_039E", "B01001_040E", "B01001_041E", "B01001_042E",
             "B01001_043E",
             "B01001_001E"),
            "sum_ratio"),
    Feature("pct_65_to_74", "% age 65–74", "B01001",
            ("B01001_020E", "B01001_021E", "B01001_022E",
             "B01001_044E", "B01001_045E", "B01001_046E",
             "B01001_001E"),
            "sum_ratio"),
    Feature("pct_75_plus", "% age 75+", "B01001",
            ("B01001_023E", "B01001_024E", "B01001_025E",
             "B01001_047E", "B01001_048E", "B01001_049E",
             "B01001_001E"),
            "sum_ratio"),
    # Dependency ratio ((under 18 + 65+) / 18-64) is derived at rank-time
    # from the bracket features above; no separate API pull.
)

# Axis 3: Race / ethnicity / origin (10)
_add(
    Feature("pct_white_alone", "% White alone",
            "B02001", ("B02001_002E", "B02001_001E"), "ratio"),
    Feature("pct_black_alone", "% Black / African American alone",
            "B02001", ("B02001_003E", "B02001_001E"), "ratio"),
    Feature("pct_aian_alone", "% American Indian / Alaska Native alone",
            "B02001", ("B02001_004E", "B02001_001E"), "ratio"),
    Feature("pct_asian_alone", "% Asian alone",
            "B02001", ("B02001_005E", "B02001_001E"), "ratio"),
    Feature("pct_nhpi_alone", "% Native Hawaiian / Pacific Islander alone",
            "B02001", ("B02001_006E", "B02001_001E"), "ratio",
            sparse_small_places=True),
    Feature("pct_other_race_alone", "% Some other race alone",
            "B02001", ("B02001_007E", "B02001_001E"), "ratio"),
    Feature("pct_two_or_more_races", "% Two or more races",
            "B02001", ("B02001_008E", "B02001_001E"), "ratio"),
    Feature("pct_hispanic", "% Hispanic / Latino (any race)",
            "B03003", ("B03003_003E", "B03003_001E"), "ratio"),
    Feature("pct_hispanic_mexican", "% Hispanic Mexican",
            "B03001", ("B03001_004E", "B03001_001E"), "ratio"),
    Feature("pct_hispanic_puerto_rican_cuban", "% Hispanic Puerto Rican + Cuban",
            "B03001", ("B03001_005E", "B03001_006E", "B03001_001E"),
            "sum_ratio"),
)

# Axis 4: Nativity / citizenship (4)
_add(
    Feature("pct_native_born", "% native-born",
            "B05002", ("B05002_002E", "B05002_001E"), "ratio"),
    Feature("pct_foreign_born", "% foreign-born",
            "B05002", ("B05002_013E", "B05002_001E"), "ratio"),
    Feature("pct_naturalized_of_foreign_born",
            "Of foreign-born: % naturalized citizens",
            "B05002", ("B05002_014E", "B05002_013E"), "ratio"),
    Feature("pct_recent_arrivals", "% of foreign-born arrived in past 10 years",
            "B05005", ("B05005_004E", "B05005_013E", "B05005_001E"),
            "sum_ratio", sparse_small_places=True),
)

# Axis 5: Language / linguistic isolation (4)
_add(
    Feature("pct_speak_spanish_home", "% speak Spanish at home (5+)",
            "B16001", ("B16001_003E", "B16001_001E"), "ratio"),
    Feature("pct_speak_other_indo_european",
            "% speak other Indo-European language at home (5+)",
            "B16001", ("B16001_060E", "B16001_001E"), "ratio"),
    Feature("pct_speak_asian_pacific",
            "% speak Asian / Pacific Islander language at home (5+)",
            "B16001", ("B16001_117E", "B16001_001E"), "ratio"),
    Feature("pct_limited_english",
            "% with limited English proficiency (speak English less than 'very well')",
            "B16004",
            ("B16004_007E", "B16004_008E", "B16004_012E", "B16004_013E",
             "B16004_017E", "B16004_018E", "B16004_022E", "B16004_023E",
             "B16004_029E", "B16004_030E", "B16004_034E", "B16004_035E",
             "B16004_039E", "B16004_040E", "B16004_044E", "B16004_045E",
             "B16004_051E", "B16004_052E", "B16004_056E", "B16004_057E",
             "B16004_061E", "B16004_062E", "B16004_066E", "B16004_067E",
             "B16004_001E"),
            "sum_ratio"),
)

# Axis 6: Economic core (5)
_add(
    Feature("median_household_income", "Median household income",
            "B19013", ("B19013_001E",), "raw"),
    Feature("median_family_income", "Median family income",
            "B19113", ("B19113_001E",), "raw"),
    Feature("per_capita_income", "Per-capita income",
            "B19301", ("B19301_001E",), "raw"),
    Feature("mean_household_income", "Mean household income",
            "B19025", ("B19025_001E",), "raw"),
    Feature("gini_index", "Gini inequality index",
            "B19083", ("B19083_001E",), "raw",
            sparse_small_places=True),
)

# Axis 7: Income distribution shape (8)
_add(
    Feature("pct_income_under_10k", "% households under $10k",
            "B19001", ("B19001_002E", "B19001_001E"), "ratio"),
    Feature("pct_income_10_to_25k", "% households $10k–$25k",
            "B19001", ("B19001_003E", "B19001_004E", "B19001_005E",
                       "B19001_001E"), "sum_ratio"),
    Feature("pct_income_25_to_50k", "% households $25k–$50k",
            "B19001", ("B19001_006E", "B19001_007E", "B19001_008E",
                       "B19001_009E", "B19001_010E", "B19001_001E"),
            "sum_ratio"),
    Feature("pct_income_50_to_75k", "% households $50k–$75k",
            "B19001", ("B19001_011E", "B19001_012E", "B19001_001E"),
            "sum_ratio"),
    Feature("pct_income_75_to_100k", "% households $75k–$100k",
            "B19001", ("B19001_013E", "B19001_001E"), "ratio"),
    Feature("pct_income_100_to_150k", "% households $100k–$150k",
            "B19001", ("B19001_014E", "B19001_015E", "B19001_001E"),
            "sum_ratio"),
    Feature("pct_income_150_to_200k", "% households $150k–$200k",
            "B19001", ("B19001_016E", "B19001_001E"), "ratio"),
    Feature("pct_income_200k_plus", "% households $200k+",
            "B19001", ("B19001_017E", "B19001_001E"), "ratio"),
)

# Axis 8: Poverty depth (3 — B17002 unavailable at place level in 2023
# ACS5, so deep/near-poverty ratio features are dropped; the core
# poverty_rate + child_poverty_rate still carry the axis.)
_add(
    Feature("poverty_rate", "% below poverty line (all ages)",
            "B17001", ("B17001_002E", "B17001_001E"), "ratio"),
    Feature("child_poverty_rate", "% children under 18 below poverty",
            "B17006", ("B17006_002E", "B17006_001E"), "ratio"),
    Feature("senior_poverty_rate", "% age 65+ below poverty",
            "B17001",
            ("B17001_015E", "B17001_016E", "B17001_029E", "B17001_030E",
             "B17001_044E", "B17001_045E", "B17001_058E", "B17001_059E",
             "B17001_044E"),  # denominator: pop 65+ for whom poverty determined
            "sum_ratio", sparse_small_places=True),
)

# Axis 9: Employment / labor (7)
_add(
    Feature("unemployment_rate", "% unemployed (civilian labor force)",
            "B23025", ("B23025_005E", "B23025_003E"), "ratio"),
    Feature("labor_force_participation",
            "% labor force participation (16+)",
            "B23025", ("B23025_002E", "B23025_001E"), "ratio"),
    Feature("pct_mgmt_professional",
            "% in management / professional occupations",
            "C24010", ("C24010_003E", "C24010_039E", "C24010_001E"),
            "sum_ratio"),
    Feature("pct_service_occupation", "% in service occupations",
            "C24010", ("C24010_019E", "C24010_055E", "C24010_001E"),
            "sum_ratio"),
    Feature("pct_production_transportation",
            "% in production / transportation / material moving",
            "C24010", ("C24010_031E", "C24010_067E", "C24010_001E"),
            "sum_ratio"),
    Feature("pct_manufacturing", "% employed in manufacturing",
            "C24050", ("C24050_005E", "C24050_001E"), "ratio",
            sparse_small_places=True),
    Feature("pct_health_education", "% employed in health + education",
            "C24050", ("C24050_013E", "C24050_001E"), "ratio",
            sparse_small_places=True),
)

# Axis 10: Education (6)
_add(
    Feature("pct_no_hs_diploma", "% without HS diploma (25+)",
            "B15003",
            ("B15003_002E", "B15003_003E", "B15003_004E", "B15003_005E",
             "B15003_006E", "B15003_007E", "B15003_008E", "B15003_009E",
             "B15003_010E", "B15003_011E", "B15003_012E", "B15003_013E",
             "B15003_014E", "B15003_015E", "B15003_016E",
             "B15003_001E"),
            "sum_ratio"),
    Feature("pct_hs_only", "% HS graduate only (25+)",
            "B15003", ("B15003_017E", "B15003_018E", "B15003_001E"),
            "sum_ratio"),
    Feature("pct_some_college", "% some college, no degree (25+)",
            "B15003", ("B15003_019E", "B15003_020E", "B15003_001E"),
            "sum_ratio"),
    Feature("pct_associates", "% associate's degree (25+)",
            "B15003", ("B15003_021E", "B15003_001E"), "ratio"),
    Feature("pct_bachelors_plus", "% bachelor's degree or higher (25+)",
            "B15003",
            ("B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
             "B15003_001E"),
            "sum_ratio"),
    Feature("pct_graduate_prof", "% graduate or professional degree (25+)",
            "B15003", ("B15003_024E", "B15003_025E", "B15003_001E"),
            "sum_ratio"),
)

# Axis 11: Food security / benefits (2)
_add(
    Feature("pct_snap_households",
            "% households receiving SNAP",
            "B22001", ("B22001_002E", "B22001_001E"), "ratio"),
    Feature("pct_public_assistance_income",
            "% households with public assistance income",
            "B19057", ("B19057_002E", "B19057_001E"), "ratio"),
)

# Axis 12: Housing (13)
_add(
    Feature("median_home_value", "Median owner-occupied home value",
            "B25077", ("B25077_001E",), "raw"),
    Feature("median_gross_rent", "Median gross rent",
            "B25064", ("B25064_001E",), "raw"),
    Feature("median_owner_costs_w_mortgage",
            "Median monthly owner costs (with mortgage)",
            "B25088", ("B25088_002E",), "raw"),
    Feature("median_owner_costs_wo_mortgage",
            "Median monthly owner costs (without mortgage)",
            "B25088", ("B25088_003E",), "raw"),
    Feature("pct_owner_occupied", "% owner-occupied housing units",
            "B25003", ("B25003_002E", "B25003_001E"), "ratio"),
    Feature("pct_vacant_units", "% vacant housing units",
            "B25002", ("B25002_003E", "B25002_001E"), "ratio"),
    Feature("pct_crowded_units",
            "% occupied units with >1 person per room",
            "B25014",
            ("B25014_005E", "B25014_006E", "B25014_007E",
             "B25014_011E", "B25014_012E", "B25014_013E",
             "B25014_001E"),
            "sum_ratio"),
    Feature("pct_single_family_detached",
            "% housing units that are single-family detached",
            "B25024", ("B25024_002E", "B25024_001E"), "ratio"),
    Feature("pct_multifamily_5plus",
            "% housing units in 5+ unit structures",
            "B25024",
            ("B25024_007E", "B25024_008E", "B25024_009E", "B25024_001E"),
            "sum_ratio"),
    Feature("pct_mobile_homes", "% mobile / manufactured homes",
            "B25024", ("B25024_010E", "B25024_001E"), "ratio"),
    Feature("median_year_built", "Median year structure built",
            "B25035", ("B25035_001E",), "raw"),
    Feature("pct_pre_1940_housing", "% housing built before 1940",
            "B25034", ("B25034_010E", "B25034_011E", "B25034_001E"),
            "sum_ratio"),
    Feature("pct_cost_burdened",
            "% households with housing cost burden (≥30% of income)",
            "B25106",
            ("B25106_006E", "B25106_010E", "B25106_014E", "B25106_018E",
             "B25106_022E",
             "B25106_028E", "B25106_032E", "B25106_036E", "B25106_040E",
             "B25106_044E",
             "B25106_001E"),
            "sum_ratio"),
)

# Axis 13: Commute / transit access (8)
_add(
    Feature("pct_drove_alone", "% drove alone to work",
            "B08301", ("B08301_003E", "B08301_001E"), "ratio"),
    Feature("pct_carpooled", "% carpooled to work",
            "B08301", ("B08301_004E", "B08301_001E"), "ratio"),
    Feature("pct_public_transit", "% public-transit commute",
            "B08301", ("B08301_010E", "B08301_001E"), "ratio"),
    Feature("pct_walked_to_work", "% walked to work",
            "B08301", ("B08301_019E", "B08301_001E"), "ratio"),
    Feature("pct_worked_from_home", "% worked from home",
            "B08301", ("B08301_021E", "B08301_001E"), "ratio"),
    Feature("median_travel_time", "Median travel time to work (minutes)",
            "B08303", ("B08303_001E",), "raw"),
    Feature("pct_long_commute", "% with 60+ minute commute",
            "B08303",
            ("B08303_012E", "B08303_013E", "B08303_001E"),
            "sum_ratio"),
    Feature("pct_zero_vehicle_households",
            "% households with no vehicle",
            "B25044",
            ("B25044_003E", "B25044_010E", "B25044_001E"),
            "sum_ratio"),
)

# Axis 14: Digital access (3)
_add(
    Feature("pct_with_computer", "% households with a computer",
            "B28003", ("B28003_002E", "B28003_001E"), "ratio"),
    Feature("pct_broadband", "% households with broadband subscription",
            "B28002", ("B28002_007E", "B28002_001E"), "ratio"),
    Feature("pct_no_internet", "% households without internet access",
            "B28002", ("B28002_013E", "B28002_001E"), "ratio"),
)

# Axis 15: Health insurance (3)
_add(
    Feature("pct_uninsured", "% population without health insurance (any age)",
            "B27010",
            ("B27010_017E", "B27010_033E", "B27010_050E", "B27010_066E",
             "B27010_001E"),
            "sum_ratio"),
    Feature("pct_uninsured_under_19", "% uninsured children under 19",
            "B27010", ("B27010_017E", "B27010_002E"), "ratio"),
    Feature("pct_uninsured_working_age",
            "% uninsured age 19–64",
            "B27010",
            ("B27010_033E", "B27010_050E",
             "B27010_018E", "B27010_034E"),
            "sum_ratio"),
)

# Axis 16: Disability (5)
_add(
    Feature("pct_with_disability", "% population with any disability",
            "B18101",
            ("B18101_004E", "B18101_007E", "B18101_010E", "B18101_013E",
             "B18101_016E", "B18101_019E",
             "B18101_023E", "B18101_026E", "B18101_029E", "B18101_032E",
             "B18101_035E", "B18101_038E",
             "B18101_001E"),
            "sum_ratio"),
    Feature("pct_ambulatory_difficulty",
            "% with ambulatory difficulty",
            "B18105",
            ("B18105_004E", "B18105_007E", "B18105_010E", "B18105_013E",
             "B18105_016E",
             "B18105_020E", "B18105_023E", "B18105_026E", "B18105_029E",
             "B18105_032E",
             "B18105_001E"),
            "sum_ratio", sparse_small_places=True),
    Feature("pct_cognitive_difficulty", "% with cognitive difficulty",
            "B18104",
            ("B18104_004E", "B18104_007E", "B18104_010E", "B18104_013E",
             "B18104_016E",
             "B18104_020E", "B18104_023E", "B18104_026E", "B18104_029E",
             "B18104_032E",
             "B18104_001E"),
            "sum_ratio", sparse_small_places=True),
    Feature("pct_independent_living_difficulty",
            "% with independent-living difficulty (18+)",
            "B18107",
            ("B18107_004E", "B18107_007E", "B18107_010E",
             "B18107_014E", "B18107_017E", "B18107_020E",
             "B18107_001E"),
            "sum_ratio", sparse_small_places=True),
    Feature("pct_self_care_difficulty",
            "% with self-care difficulty",
            "B18106",
            ("B18106_004E", "B18106_007E", "B18106_010E", "B18106_013E",
             "B18106_016E",
             "B18106_020E", "B18106_023E", "B18106_026E", "B18106_029E",
             "B18106_032E",
             "B18106_001E"),
            "sum_ratio", sparse_small_places=True),
)

# Axis 17: Family structure (5)
_add(
    Feature("pct_married_couple_with_children",
            "% married-couple families with children under 18",
            "B11003", ("B11003_003E", "B11003_001E"), "ratio"),
    Feature("pct_single_mother_families",
            "% single-mother families with children under 18",
            "B11003", ("B11003_016E", "B11003_001E"), "ratio"),
    Feature("pct_single_father_families",
            "% single-father families with children under 18",
            "B11003", ("B11003_010E", "B11003_001E"), "ratio"),
    Feature("pct_grandparent_households",
            "% households with grandparents responsible for grandchildren",
            "B10051", ("B10051_002E", "B10051_001E"), "ratio",
            sparse_small_places=True),
    Feature("pct_households_with_children",
            "% households with children under 18",
            "B11005", ("B11005_002E", "B11005_001E"), "ratio"),
)

# Axis 18: Residential stability (4) — all from B07003 (Geographical
# Mobility by Sex), which is the table actually published at place
# level in ACS5. Earlier draft used B07001 (age breakdown) which has a
# different variable numbering; B07003 keeps all four rates consistent.
_add(
    Feature("pct_same_house_1yr", "% lived in same house 1 year ago",
            "B07003", ("B07003_004E", "B07003_001E"), "ratio"),
    Feature("pct_moved_within_county",
            "% moved within same county in past year",
            "B07003", ("B07003_007E", "B07003_001E"), "ratio"),
    Feature("pct_moved_from_different_state",
            "% moved from different state in past year",
            "B07003", ("B07003_013E", "B07003_001E"), "ratio",
            sparse_small_places=True),
    Feature("pct_moved_from_abroad", "% moved from abroad in past year",
            "B07003", ("B07003_016E", "B07003_001E"), "ratio",
            sparse_small_places=True),
)

# Axis 19: Veterans (1 — B21007 unavailable at place level in 2023
# ACS5, so veteran_poverty_rate is dropped until we find a working
# replacement.)
_add(
    Feature("pct_veterans", "% veterans (civilian population 18+)",
            "B21001", ("B21001_002E", "B21001_001E"), "ratio"),
)


# ---------------------------------------------------------------------------
# Axis groupings — used by PeerRetriever to pick axis-specific peer sets.
# Axes reference features by name; features can appear in multiple axes.
# ---------------------------------------------------------------------------

AXES: dict[str, dict] = {
    "size": {
        "description": "Population, household count, density of group quarters",
        "features": [
            "total_population", "total_households",
            "pct_non_family_households", "avg_household_size",
            "pct_group_quarters",
        ],
    },
    "age_structure": {
        "description": "Age distribution shape",
        "features": [
            "median_age", "pct_under_5", "pct_5_to_17", "pct_18_to_24",
            "pct_25_to_44", "pct_45_to_64", "pct_65_to_74", "pct_75_plus",
        ],
    },
    "race_ethnicity": {
        "description": "Race and ethnicity composition",
        "features": [
            "pct_white_alone", "pct_black_alone", "pct_aian_alone",
            "pct_asian_alone", "pct_nhpi_alone", "pct_other_race_alone",
            "pct_two_or_more_races", "pct_hispanic",
            "pct_hispanic_mexican", "pct_hispanic_puerto_rican_cuban",
        ],
    },
    "nativity": {
        "description": "Native-born vs foreign-born, recency of arrival",
        "features": [
            "pct_native_born", "pct_foreign_born",
            "pct_naturalized_of_foreign_born", "pct_recent_arrivals",
        ],
    },
    "language": {
        "description": "Home language and English proficiency",
        "features": [
            "pct_speak_spanish_home", "pct_speak_other_indo_european",
            "pct_speak_asian_pacific", "pct_limited_english",
        ],
    },
    "economic": {
        "description": "Income, poverty, unemployment, inequality",
        "features": [
            "median_household_income", "median_family_income",
            "per_capita_income", "mean_household_income", "gini_index",
            "poverty_rate", "child_poverty_rate", "senior_poverty_rate",
            "unemployment_rate", "labor_force_participation",
        ],
    },
    "income_distribution": {
        "description": "Shape of the household income distribution",
        "features": [
            "pct_income_under_10k", "pct_income_10_to_25k",
            "pct_income_25_to_50k", "pct_income_50_to_75k",
            "pct_income_75_to_100k", "pct_income_100_to_150k",
            "pct_income_150_to_200k", "pct_income_200k_plus",
        ],
    },
    "employment": {
        "description": "Occupational and industry mix",
        "features": [
            "pct_mgmt_professional", "pct_service_occupation",
            "pct_production_transportation",
            "pct_manufacturing", "pct_health_education",
            "labor_force_participation", "unemployment_rate",
        ],
    },
    "education": {
        "description": "Educational attainment",
        "features": [
            "pct_no_hs_diploma", "pct_hs_only", "pct_some_college",
            "pct_associates", "pct_bachelors_plus", "pct_graduate_prof",
        ],
    },
    "food_benefits": {
        "description": "SNAP + public assistance participation",
        "features": [
            "pct_snap_households", "pct_public_assistance_income",
            "poverty_rate", "child_poverty_rate",
        ],
    },
    "housing": {
        "description": "Value, cost, tenure, and stock characteristics",
        "features": [
            "median_home_value", "median_gross_rent",
            "median_owner_costs_w_mortgage", "median_owner_costs_wo_mortgage",
            "pct_owner_occupied", "pct_vacant_units", "pct_crowded_units",
            "pct_single_family_detached", "pct_multifamily_5plus",
            "pct_mobile_homes", "median_year_built", "pct_pre_1940_housing",
            "pct_cost_burdened",
        ],
    },
    "commute_transit": {
        "description": "Commute mode and travel time",
        "features": [
            "pct_drove_alone", "pct_carpooled", "pct_public_transit",
            "pct_walked_to_work", "pct_worked_from_home",
            "median_travel_time", "pct_long_commute",
            "pct_zero_vehicle_households",
        ],
    },
    "digital_access": {
        "description": "Computer and broadband access",
        "features": [
            "pct_with_computer", "pct_broadband", "pct_no_internet",
        ],
    },
    "health_insurance": {
        "description": "Health-insurance coverage",
        "features": [
            "pct_uninsured", "pct_uninsured_under_19",
            "pct_uninsured_working_age",
        ],
    },
    "disability": {
        "description": "Disability prevalence and type",
        "features": [
            "pct_with_disability", "pct_ambulatory_difficulty",
            "pct_cognitive_difficulty",
            "pct_independent_living_difficulty",
            "pct_self_care_difficulty",
        ],
    },
    "family_structure": {
        "description": "Household composition and child-raising context",
        "features": [
            "pct_married_couple_with_children", "pct_single_mother_families",
            "pct_single_father_families", "pct_grandparent_households",
            "pct_households_with_children",
        ],
    },
    "residential_stability": {
        "description": "Housing turnover and migration inflows",
        "features": [
            "pct_same_house_1yr", "pct_moved_within_county",
            "pct_moved_from_different_state", "pct_moved_from_abroad",
        ],
    },
    "veterans": {
        "description": "Veteran population share",
        "features": ["pct_veterans"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tables_used() -> set[str]:
    """Every distinct ACS5 table the catalog pulls from."""
    return {f.table for f in FEATURES.values()}


def variables_for_table(table: str) -> set[str]:
    """Every distinct variable ID this catalog needs from one table."""
    vs: set[str] = set()
    for f in FEATURES.values():
        if f.table == table:
            vs.update(f.variables)
    return vs


def validate() -> list[str]:
    """Return any catalog integrity problems — axes referencing missing
    features, duplicate variable refs, etc. Returns [] if clean."""
    issues: list[str] = []
    for axis_name, axis in AXES.items():
        for fname in axis["features"]:
            if fname not in FEATURES:
                issues.append(
                    f"axis {axis_name!r} references missing feature {fname!r}"
                )
    return issues
