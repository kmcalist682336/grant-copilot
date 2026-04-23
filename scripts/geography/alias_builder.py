"""Build the geo_aliases table from loaded administrative geographies.

For each geography in admin_geographies, generates:
  - Exact name match
  - Common abbreviations and variations
  - Suffix stripping (e.g., "Fulton" for "Fulton County")
  - Known aliases from the curated alias list

Also loads curated aliases for Atlanta neighborhoods, road corridors,
and informal regional names.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suffix normalization rules
# These strip legal suffixes so "Fulton" matches "Fulton County"
# ---------------------------------------------------------------------------

COUNTY_SUFFIXES = [
    " County", " Parish", " Borough", " Census Area",
    " City and Borough", " Municipality", " Consolidated Government",
    " Unified Government", " Metropolitan Government",
]

PLACE_SUFFIXES = [
    " city", " town", " village", " borough", " township",
    " CDP", " municipality",
]

STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT",
    "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

# ---------------------------------------------------------------------------
# Curated aliases — things that can't be derived algorithmically
# ---------------------------------------------------------------------------

# Format: (alias, target_id_pattern, target_type, alias_type)
# target_id_pattern uses {state_fips} and {county_fips} substitution

CURATED_STATE_ALIASES = [
    # Georgia
    ("GA", "13", "admin", "abbreviation"),
    ("the Peach State", "13", "admin", "common"),

    # Common state abbreviations are handled algorithmically from STATE_ABBREVIATIONS
]

CURATED_COUNTY_ALIASES = [
    # Georgia counties with common informal names
    # (alias, county_geoid, alias_type)
    ("DeKalb", "13089", "common"),
    ("Dekalb County", "13089", "common"),
    ("Dek", "13089", "abbreviation"),
    ("Gwinnett", "13135", "common"),
    ("Gwinett", "13135", "misspelling"),
    ("Gwinette", "13135", "misspelling"),
    ("Fulton", "13121", "common"),
    ("Cobb", "13067", "common"),
    ("Clayton", "13063", "common"),
    ("Cherokee", "13057", "common"),
    ("Forsyth", "13117", "common"),
    ("Hall", "13139", "common"),
    ("Henry", "13151", "common"),
    ("Fayette", "13113", "common"),
    ("Paulding", "13223", "common"),
    ("Douglas", "13097", "common"),
    ("Rockdale", "13247", "common"),
    ("Newton", "13217", "common"),
    ("Barrow", "13013", "common"),
    ("Walton", "13297", "common"),
    ("Coweta", "13077", "common"),
    ("Carroll", "13045", "common"),
    ("Spalding", "13255", "common"),
    ("Butts", "13035", "common"),
    ("Jackson", "13157", "common"),        # important ambiguity case
    ("Dawson", "13085", "common"),
    ("Pickens", "13227", "common"),
    ("Bartow", "13015", "common"),
    ("Haralson", "13143", "common"),
    ("Heard", "13149", "common"),
    ("Lamar", "13171", "common"),
    ("Meriwether", "13199", "common"),
    ("Pike", "13231", "common"),
    ("Upson", "13293", "common"),
    ("Jasper", "13159", "common"),
    ("Morgan", "13211", "common"),
    ("Putnam", "13233", "common"),
    ("Greene", "13133", "common"),
]

CURATED_PLACE_ALIASES = [
    # Atlanta area cities and common references
    # (alias, place_geoid, alias_type)
    ("ATL", "1304000", "abbreviation"),
    ("the ATL", "1304000", "common"),
    ("City of Atlanta", "1304000", "exact"),
    ("Atlanta GA", "1304000", "common"),
    ("ITP Atlanta", "1304000", "common"),   # "inside the perimeter"
    ("Midtown Atlanta", "1304000", "common"),  # maps to city, not just the neighborhood
    ("Downtown Atlanta", "1304000", "common"),

    ("Decatur", "1321380", "common"),
    ("City of Decatur", "1321380", "exact"),
    ("Sandy Springs", "1367284", "common"),
    ("Dunwoody", "1324600", "common"),
    ("Brookhaven", "1310788", "common"),
    ("Chamblee", "1313688", "common"),
    ("Doraville", "1322908", "common"),
    ("Roswell", "1365508", "common"),
    ("Alpharetta", "1300988", "common"),
    ("Marietta", "1349756", "common"),
    ("Smyrna", "1371276", "common"),
    ("Kennesaw", "1343960", "common"),
    ("Acworth", "1300124", "common"),
    ("Canton", "1313372", "common"),
    ("Woodstock", "1381908", "common"),
    ("Lawrenceville", "1344340", "common"),
    ("Duluth", "1323908", "common"),
    ("Norcross", "1356648", "common"),
    ("Peachtree City", "1360132", "common"),
    ("Newnan", "1355748", "common"),
    ("Griffin", "1336948", "common"),
    ("McDonough", "1350172", "common"),
    ("Stockbridge", "1372812", "common"),
    ("College Park", "1317428", "common"),
    ("East Point", "1325264", "common"),
    ("Union City", "1379672", "common"),
    ("Morrow", "1353428", "common"),
    ("Jonesboro", "1342340", "common"),

    # Major Georgia cities outside metro
    ("Savannah", "1369908", "common"),
    ("Augusta", "1303440", "common"),   # Augusta-Richmond County consolidated
    ("Columbus", "1319000", "common"),  # Columbus-Muscogee County
    ("Macon", "1348972", "common"),
    ("Athens", "1302344", "common"),    # Athens-Clarke County
    ("Albany", "1300388", "common"),
]


# ---------------------------------------------------------------------------
# Curated named places (neighborhoods, regions, corridors)
# These are inserted into the named_places table separately
# ---------------------------------------------------------------------------

# ATL neighborhoods with common aliases
# These reference named_places entries (loaded separately from geometries)
ATL_NEIGHBORHOOD_ALIASES = [
    # (neighborhood_name, [aliases])
    ("Buckhead", ["Buckhead Atlanta", "upscale Atlanta", "north Atlanta"]),
    ("Midtown", ["Midtown Atlanta", "arts district"]),
    ("Downtown", ["Downtown Atlanta", "downtown ATL", "the downtown"]),
    ("Old Fourth Ward", ["O4W", "Old 4th Ward", "4th Ward"]),
    ("Grant Park", ["Grant Park Atlanta"]),
    ("Inman Park", ["Inman Park Atlanta"]),
    ("Virginia Highland", ["Va-Hi", "Virginia Highlands"]),
    ("Little Five Points", ["L5P", "Little 5 Points"]),
    ("Decatur (neighborhood)", ["Decatur area"]),
    ("Kirkwood", ["Kirkwood Atlanta"]),
    ("East Atlanta", ["East Atlanta Village", "EAV"]),
    ("West End", ["West End Atlanta"]),
    ("Vine City", ["Vine City Atlanta"]),
    ("English Avenue", ["English Ave"]),
    ("Mechanicsville", ["Mechanicsville Atlanta"]),
    ("Summerhill", ["Summerhill Atlanta"]),
    ("Pittsburgh", ["Pittsburgh Atlanta"]),
    ("Peoplestown", ["Peoplestown Atlanta"]),
    ("Adair Park", ["Adair Park Atlanta"]),
    ("Oakland City", ["Oakland City Atlanta"]),
    ("Westview", ["Westview Atlanta"]),
    ("Cascade Heights", ["Cascade", "Cascade Road area"]),
    ("Adamsville", ["Adamsville Atlanta"]),
    ("Collier Heights", ["Collier Heights Atlanta"]),
    ("Bankhead", ["Bankhead Atlanta", "the Bankhead"]),
    ("Grove Park", ["Grove Park Atlanta"]),
    ("Edgewood", ["Edgewood Atlanta"]),
    ("Reynoldstown", ["Reynoldstown Atlanta"]),
    ("Cabbagetown", ["Cabbagetown Atlanta"]),
    ("Poncey-Highland", ["Ponce Highland", "Poncey Highland"]),
    ("North Highland", ["North Highland Atlanta"]),
    ("Candler Park", ["Candler Park Atlanta"]),
    ("Lake Claire", ["Lake Claire Atlanta"]),
    ("Druid Hills", ["Druid Hills Atlanta"]),
    ("Toco Hills", ["Toco Hills area"]),
    ("Emory", ["Emory area", "around Emory", "near Emory"]),
    ("Tucker", ["Tucker GA", "Tucker area"]),
    ("Stone Mountain", ["Stone Mountain area", "Stone Mountain village"]),
    ("Clarkston", ["Clarkston GA", "Clarkston area"]),
    ("Avondale Estates", ["Avondale"]),
    ("Conley", ["Conley GA"]),
    ("Panthersville", ["Panthersville area"]),
    ("Lithonia", ["Lithonia GA", "Lithonia area"]),
    ("South DeKalb", ["South Dekalb", "S. DeKalb"]),
    ("North Fulton", ["North Fulton County", "north Fulton"]),
    ("South Fulton", ["South Fulton County", "south Fulton", "SW Atlanta"]),
    ("Sandy Springs (neighborhood)", ["Sandy Springs area"]),
    ("Perimeter Center", ["the Perimeter", "Perimeter area"]),
    ("Cumberland", ["Cumberland area", "Cumberland Galleria area"]),
    ("Vinings", ["Vinings GA", "Vinings area"]),
    ("Mableton", ["Mableton GA", "Mableton area"]),
    ("Austell", ["Austell GA"]),
    ("Lithia Springs", ["Lithia Springs GA"]),
    ("Powder Springs", ["Powder Springs GA"]),
    ("Hiram", ["Hiram GA"]),
    ("Villa Rica", ["Villa Rica GA"]),
    ("Carrollton", ["Carrollton GA"]),
    ("Douglasville", ["Douglasville GA"]),
    ("Peachtree Corners", ["Peachtree Corners GA", "Peachtree Corners area"]),
    ("Suwanee", ["Suwanee GA", "Suwanee area"]),
    ("Buford", ["Buford GA", "Buford area"]),
    ("Sugar Hill", ["Sugar Hill GA"]),
    ("Gainesville", ["Gainesville GA", "Gainesville area"]),
    ("Flowery Branch", ["Flowery Branch GA"]),
]

# Road corridor aliases
ROAD_ALIASES = [
    # (road_id, [aliases])
    ("I-20", ["Interstate 20", "I20", "i-20", "Highway 20 Atlanta"]),
    ("I-75", ["Interstate 75", "I75", "i-75"]),
    ("I-85", ["Interstate 85", "I85", "i-85"]),
    ("I-285", ["Interstate 285", "I285", "i-285", "the Perimeter highway",
               "285", "perimeter interstate"]),
    ("I-575", ["Interstate 575", "I575", "i-575"]),
    ("I-675", ["Interstate 675", "I675", "i-675"]),
    ("GA-400", ["Georgia 400", "GA 400", "400", "the 400"]),
    ("US-78", ["Highway 78", "US 78", "Stone Mountain Freeway"]),
    ("US-29", ["Highway 29", "US 29"]),
    ("US-19", ["Highway 19", "US 19"]),
    ("US-41", ["Highway 41", "Cobb Parkway", "US 41"]),
    ("US-23", ["Highway 23", "Buford Highway", "BuHi"]),
    ("SR-316", ["State Route 316", "GA 316", "316", "University Parkway"]),
    ("SR-20", ["State Route 20", "GA 20"]),
]

# Informal regional names → county lists
REGIONAL_ALIASES = [
    # (region_name, [aliases])
    ("ATL Metro", ["Atlanta metro", "greater Atlanta", "the metro",
                   "metro Atlanta", "Atlanta metropolitan area",
                   "Atlanta MSA"]),
    ("ITP", ["inside the perimeter", "inside 285", "inside I-285"]),
    ("OTP", ["outside the perimeter", "outside 285", "outside I-285",
             "suburbs", "the suburbs"]),
    ("South Atlanta", ["south ATL", "SWATS", "southwest Atlanta",
                       "southwest ATL"]),
    ("North Atlanta", ["north ATL", "north side"]),
    ("BeltLine area", ["Atlanta BeltLine", "the BeltLine", "Beltline corridor"]),
    ("I-20 corridor", ["along I-20", "I-20 corridor", "the I-20 corridor",
                       "east-west corridor"]),
    ("I-85 corridor", ["along I-85", "I-85 corridor", "the I-85 corridor",
                       "northeast corridor", "Buford Highway corridor"]),
    ("I-75 corridor", ["along I-75", "I-75 corridor", "northwest corridor"]),
    ("GA-400 corridor", ["along 400", "GA-400 corridor", "north Fulton corridor"]),
    ("Rural Georgia", ["rural GA", "rural areas", "non-metro Georgia",
                       "outside the metro"]),
    ("South Georgia", ["south GA", "south Georgia", "SW Georgia",
                       "SE Georgia", "deep south Georgia"]),
    ("North Georgia", ["north GA", "north Georgia", "north Georgia mountains",
                       "the mountains"]),
    ("Middle Georgia", ["central Georgia", "middle GA"]),
    ("coastal Georgia", ["the coast", "Georgia coast", "coastal GA"]),
]


# ---------------------------------------------------------------------------
# Alias insertion
# ---------------------------------------------------------------------------

def _insert_alias(
    conn: sqlite3.Connection,
    alias: str,
    target_id: str,
    target_type: str,
    target_name: str,
    alias_type: str,
    state_fips: Optional[str] = None,
    confidence: float = 1.0,
) -> None:
    """Insert a single alias, ignoring duplicates."""
    try:
        conn.execute(
            "INSERT INTO geo_aliases "
            "(alias, target_id, target_type, target_name, "
            " alias_type, state_fips, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alias.strip(), target_id, target_type, target_name,
             alias_type, state_fips, confidence),
        )
    except sqlite3.IntegrityError:
        pass


def build_admin_aliases(conn: sqlite3.Connection) -> int:
    """Generate aliases for all administrative geographies."""
    total = 0

    cur = conn.execute(
        "SELECT geoid, name, geo_type, state_fips FROM admin_geographies"
    )
    rows = cur.fetchall()

    with conn:
        for row in rows:
            geoid = row["geoid"]
            name = row["name"]
            geo_type = row["geo_type"]
            state_fips = row["state_fips"]

            aliases_for_row = _generate_admin_aliases(
                geoid, name, geo_type, state_fips
            )
            for alias, alias_type, confidence in aliases_for_row:
                _insert_alias(
                    conn, alias, geoid, "admin", name,
                    alias_type, state_fips, confidence,
                )
                total += 1

    logger.info("Generated %d admin aliases", total)
    return total


def _generate_admin_aliases(
    geoid: str,
    name: str,
    geo_type: str,
    state_fips: str,
) -> list[tuple[str, str, float]]:
    """Return list of (alias, alias_type, confidence) for one geography."""
    aliases = []

    # Always include exact name
    aliases.append((name, "exact", 1.0))

    if geo_type == "state":
        # State abbreviation
        abbrev = STATE_ABBREVIATIONS.get(name)
        if abbrev:
            aliases.append((abbrev, "abbreviation", 1.0))
            aliases.append((f"state of {name}", "common", 0.9))

    elif geo_type == "county":
        # Strip " County" suffix → "Fulton"
        short = name
        for suffix in COUNTY_SUFFIXES:
            if name.endswith(suffix):
                short = name[:-len(suffix)]
                aliases.append((short, "common", 0.9))
                break
        # "Fulton County, Georgia" / "Fulton County, GA"
        # (state name appended is handled by resolution scoping, not aliases)

    elif geo_type == "place":
        # Strip city/town suffix
        short = name
        for suffix in PLACE_SUFFIXES:
            if name.lower().endswith(suffix.lower()):
                short = name[:-(len(suffix))]
                aliases.append((short, "common", 0.9))
                break
        # Add "city of X" form
        aliases.append((f"City of {name}", "common", 0.95))

    elif geo_type == "tract":
        # Tracts are rarely referenced by name — skip extra aliases
        pass

    return aliases


def load_curated_county_aliases(conn: sqlite3.Connection) -> int:
    """Load hand-curated county aliases."""
    total = 0
    with conn:
        for alias, geoid, alias_type in CURATED_COUNTY_ALIASES:
            # Get the county name for denormalization
            cur = conn.execute(
                "SELECT name FROM admin_geographies WHERE geoid = ?", (geoid,)
            )
            row = cur.fetchone()
            if not row:
                logger.warning("County %s not found in admin_geographies", geoid)
                continue

            state_fips = geoid[:2]
            _insert_alias(
                conn, alias, geoid, "admin", row["name"],
                alias_type, state_fips, 0.95,
            )
            total += 1

    logger.info("Loaded %d curated county aliases", total)
    return total


def load_curated_place_aliases(conn: sqlite3.Connection) -> int:
    """Load hand-curated place aliases."""
    total = 0
    with conn:
        for alias, geoid, alias_type in CURATED_PLACE_ALIASES:
            cur = conn.execute(
                "SELECT name FROM admin_geographies WHERE geoid = ?", (geoid,)
            )
            row = cur.fetchone()
            if not row:
                logger.warning(
                    "Place %s not found — will add with placeholder name", geoid
                )
                name = alias  # Use alias as name
            else:
                name = row["name"]

            state_fips = geoid[:2]
            _insert_alias(
                conn, alias, geoid, "admin", name,
                alias_type, state_fips, 0.95,
            )
            total += 1

    logger.info("Loaded %d curated place aliases", total)
    return total


def load_road_aliases(conn: sqlite3.Connection) -> int:
    """Load road aliases from the curated ROAD_ALIASES list."""
    total = 0
    with conn:
        for road_id, aliases in ROAD_ALIASES:
            # Check road exists
            cur = conn.execute(
                "SELECT name FROM roads WHERE road_id = ?", (road_id,)
            )
            row = cur.fetchone()
            if not row:
                logger.warning(
                    "Road %s not in roads table — skipping aliases", road_id
                )
                continue

            road_name = row["name"]
            # Primary name
            _insert_alias(
                conn, road_name, road_id, "road", road_name,
                "exact", "13", 1.0,
            )
            for alias in aliases:
                _insert_alias(
                    conn, alias, road_id, "road", road_name,
                    "common", "13", 0.9,
                )
                total += 1

    logger.info("Loaded %d road aliases", total)
    return total


def load_named_place_aliases(conn: sqlite3.Connection) -> int:
    """Load aliases for ATL neighborhoods and regional names."""
    total = 0
    with conn:
        # Neighborhood aliases
        for neighborhood_name, aliases in ATL_NEIGHBORHOOD_ALIASES:
            cur = conn.execute(
                "SELECT place_id FROM named_places WHERE name = ?",
                (neighborhood_name,),
            )
            row = cur.fetchone()
            if not row:
                logger.debug(
                    "Neighborhood '%s' not in named_places — skipping",
                    neighborhood_name,
                )
                continue
            place_id = row["place_id"]

            _insert_alias(
                conn, neighborhood_name, place_id, "named_place",
                neighborhood_name, "exact", "13", 1.0,
            )
            for alias in aliases:
                _insert_alias(
                    conn, alias, place_id, "named_place",
                    neighborhood_name, "common", "13", 0.9,
                )
                total += 1

        # Regional aliases
        for region_name, aliases in REGIONAL_ALIASES:
            cur = conn.execute(
                "SELECT place_id FROM named_places WHERE name = ?",
                (region_name,),
            )
            row = cur.fetchone()
            if not row:
                logger.debug(
                    "Region '%s' not in named_places — skipping", region_name
                )
                continue
            place_id = row["place_id"]

            _insert_alias(
                conn, region_name, place_id, "named_place",
                region_name, "exact", "13", 1.0,
            )
            for alias in aliases:
                _insert_alias(
                    conn, alias, place_id, "named_place",
                    region_name, "common", "13", 0.9,
                )
                total += 1

    logger.info("Loaded %d named place aliases", total)
    return total


def rebuild_fts_index(conn: sqlite3.Connection) -> None:
    """Rebuild the FTS5 index from current geo_aliases content."""
    logger.info("Rebuilding FTS index")
    conn.execute("INSERT INTO geo_fts(geo_fts) VALUES('rebuild')")
    conn.commit()
    logger.info("FTS index rebuilt")
