"""Resolver tests — exercise every strategy against the real gazetteer.

Focus areas:
  - Per-ref_type happy paths (admin, neighborhood, corridor, region,
    landmark_area, national, relative)
  - GA-bias tiebreaker (Fulton County should resolve to GA, not AR/OH)
  - Non-GA admin falls back to state/msa for data_level_available
  - Empty-FTS and gazetteer-gap paths return empty, not crash
  - `resolve_intent` with non-compound multi-ref (comparisons)
"""
from __future__ import annotations

from scripts.chatbot.models import (
    ExtractedGeoRef,
    ExtractedIntent,
    CompoundSpec,
)
from scripts.chatbot.geo_resolver import resolve_intent, resolve_ref


# ---------------------------------------------------------------------------
# Administrative
# ---------------------------------------------------------------------------

def test_administrative_ga_bias_on_fulton_county(gazetteer_db):
    """Fulton County must resolve to GA (13121), not an out-of-state Fulton."""
    ref = ExtractedGeoRef(
        text="Fulton County", ref_type="administrative",
        qualifier="in", scope_hint="Georgia",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_id == "13121"
    assert top.geo_level == "county"
    assert top.data_level_available in ("tract", "county")
    assert top.confidence >= 0.7


def test_administrative_state(gazetteer_db):
    ref = ExtractedGeoRef(text="Georgia", ref_type="administrative")
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_id == "13"
    assert top.geo_level == "state"
    assert top.confidence >= 0.8


def test_administrative_city_atlanta(gazetteer_db):
    ref = ExtractedGeoRef(
        text="Atlanta", ref_type="administrative", scope_hint="Georgia",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert "Atlanta" in top.display_name
    assert top.geo_level == "place"


def test_administrative_non_ga_data_level_note(gazetteer_db):
    """A non-GA admin resolution should carry the 'only state/MSA' note."""
    ref = ExtractedGeoRef(
        text="Charlotte", ref_type="administrative",
        scope_hint="North Carolina",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    # Either we resolve to a NC Charlotte, or the note is attached.
    if top.data_level_available != "tract":
        assert any(
            "outside Georgia" in n.lower()
            or "only state" in n.lower()
            for n in top.assumption_notes
        )


# ---------------------------------------------------------------------------
# MSA preference for cities outside GA (post-Phase-C principal-city aliases)
# ---------------------------------------------------------------------------

def test_houston_resolves_to_msa_not_ga_county(gazetteer_db):
    """'Houston, TX' must resolve to Houston MSA, not Houston County (GA).

    Pre-fix bug: FTS only knew the MSA by its full name, so "Houston"
    fell through to the same-named GA county. Migration added a
    principal_city alias and the FTS ORDER BY now prefers text-exact
    alias matches.
    """
    ref = ExtractedGeoRef(
        text="Houston", ref_type="administrative", scope_hint="Texas",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_level == "msa", (
        f"expected MSA, got {top.geo_level} ({top.display_name})"
    )
    assert "Houston" in top.display_name
    # Houston MSA is in CBSA 26420; data tier should be 'msa'.
    assert top.geo_id == "26420"
    assert top.data_level_available == "msa"


def test_chicago_resolves_to_msa(gazetteer_db):
    """No Chicago city exists in our places gazetteer (GA-only); resolve
    to the Chicago MSA via principal_city alias."""
    ref = ExtractedGeoRef(
        text="Chicago", ref_type="administrative", scope_hint="Illinois",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_level == "msa"
    assert "Chicago" in top.display_name


def test_atlanta_still_resolves_to_city_not_msa(gazetteer_db):
    """Regression guard: 'Atlanta' must keep resolving to Atlanta city,
    not the same-named MSA (which now also has a principal_city alias)."""
    ref = ExtractedGeoRef(
        text="Atlanta", ref_type="administrative", scope_hint="Georgia",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_level == "place", (
        f"expected place (city), got {top.geo_level} ({top.display_name})"
    )
    assert top.geo_id == "1304000"


def test_fulton_county_still_ga(gazetteer_db):
    """Regression guard: GA-bias tiebreak still works for shared county
    names after the FTS ORDER BY change."""
    ref = ExtractedGeoRef(
        text="Fulton County", ref_type="administrative",
        scope_hint="Georgia",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    assert out[0].geo_id == "13121"


# ---------------------------------------------------------------------------
# Neighborhood
# ---------------------------------------------------------------------------

def test_neighborhood_buckhead(gazetteer_db):
    ref = ExtractedGeoRef(
        text="Buckhead", ref_type="neighborhood", scope_hint="Atlanta",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    # SuperDistrict 4 ("Buckhead") should be the primary since it has the
    # widest tract coverage and matches as an exact alias.
    assert "Buckhead" in top.display_name
    assert top.geo_level == "neighborhood"
    assert len(top.tract_geoids) > 5
    assert top.confidence >= 0.7


def test_neighborhood_midtown(gazetteer_db):
    ref = ExtractedGeoRef(text="Midtown", ref_type="neighborhood", scope_hint="Atlanta")
    out = resolve_ref(ref, gazetteer_db)
    assert out
    assert "Midtown" in out[0].display_name or "Midtown" in out[0].geo_id
    assert out[0].tract_geoids


# ---------------------------------------------------------------------------
# Corridor
# ---------------------------------------------------------------------------

def test_corridor_i85_along(gazetteer_db):
    ref = ExtractedGeoRef(text="I-85", ref_type="corridor", qualifier="along")
    out = resolve_ref(ref, gazetteer_db)
    assert out
    top = out[0]
    assert top.geo_level == "corridor"
    assert top.geo_id == "I-85"
    assert len(top.tract_geoids) > 50


def test_corridor_ga400_near_uses_buffer(gazetteer_db):
    ref = ExtractedGeoRef(
        text="GA-400", ref_type="corridor", qualifier="near",
        proximity_tier="very_local",
    )
    out = resolve_ref(ref, gazetteer_db)
    assert out
    assert out[0].geo_id == "GA-400"
    assert any("buffer" in n.lower() for n in out[0].assumption_notes)


# ---------------------------------------------------------------------------
# National
# ---------------------------------------------------------------------------

def test_national_hardcoded(gazetteer_db):
    ref = ExtractedGeoRef(text="US", ref_type="national")
    out = resolve_ref(ref, gazetteer_db)
    assert len(out) == 1
    assert out[0].geo_id == "1"
    assert out[0].geo_level == "us"
    assert out[0].confidence == 1.0


# ---------------------------------------------------------------------------
# Empty & edge
# ---------------------------------------------------------------------------

def test_empty_geo_refs_returns_empty(gazetteer_db):
    intent = ExtractedIntent(geo_refs=[], intent_type="context")
    assert resolve_intent(intent, gazetteer_db) == []


def test_unknown_text_returns_empty_gracefully(gazetteer_db):
    ref = ExtractedGeoRef(
        text="xyzzzq notarealplace", ref_type="administrative",
    )
    out = resolve_ref(ref, gazetteer_db)
    # Should return either nothing or low-confidence noise — never crash.
    for r in out:
        assert 0.0 <= r.confidence <= 1.0


# ---------------------------------------------------------------------------
# Multi-ref comparison
# ---------------------------------------------------------------------------

def test_comparison_georgia_vs_us(gazetteer_db):
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="Georgia", ref_type="administrative"),
            ExtractedGeoRef(text="US", ref_type="national"),
        ],
        intent_type="comparison",
        comparison_implied=True,
        national_comparison_implied=True,
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 2
    assert out[0].geo_level == "state"
    assert out[1].geo_level == "us"


# ---------------------------------------------------------------------------
# Gold-set sweep — tracks resolver behavior across the entire fixture set.
# Reported as a single aggregated pass/fail to make it easy to see
# regressions at a glance.
# ---------------------------------------------------------------------------

def _ref_from_gold(ref_dict: dict) -> ExtractedGeoRef:
    return ExtractedGeoRef(**{
        k: v for k, v in ref_dict.items()
        if k in {"text", "ref_type", "qualifier",
                 "proximity_tier", "scope_hint"}
    })


def _intent_from_gold(entry: dict) -> ExtractedIntent:
    exp = entry["expected"]
    return ExtractedIntent(
        geo_refs=[_ref_from_gold(r) for r in exp.get("geo_refs", [])],
        compounds=[CompoundSpec(**c) for c in exp.get("compounds", [])],
        intent_type=exp.get("intent_type", "exact"),
        comparison_implied=exp.get("comparison_implied", False),
        national_comparison_implied=exp.get(
            "national_comparison_implied", False
        ),
    )


def test_gold_sweep(gazetteer_db, gold_entries):
    """Run every gold entry through the resolver, categorize failures.

    Failures split into two buckets:
      - data_gap: the resolver returned no resolution because the entity
        isn't in the gazetteer (e.g., Piedmont Park has no named_places
        row). These are ingestion gaps, not resolver bugs.
      - logic:    the resolver returned something but at the wrong
        geo_level or otherwise clearly off.

    Only `logic` failures gate the test (>10% of eligible entries hard-fails).
    `data_gap` failures are reported but don't fail — they're trackable
    signal for future gazetteer improvements.
    """
    data_gaps: list[str] = []
    logic_fails: list[str] = []
    eligible = 0

    for i, entry in enumerate(gold_entries):
        exp = entry["expected"]
        query = entry["query"]

        if not exp.get("geo_refs"):
            continue
        if exp.get("resolved") == []:
            continue
        eligible += 1

        intent = _intent_from_gold(entry)
        resolved = resolve_intent(intent, gazetteer_db)

        if not resolved:
            data_gaps.append(f"[{i}] {query!r}")
            continue

        exp_res = exp.get("resolved", [])
        if exp_res:
            want_level = exp_res[0].get("geo_level")
            got_level = resolved[0].geo_level
            if want_level and want_level != got_level:
                # Close-kin tolerance: neighborhood<->region are fuzzy.
                if {want_level, got_level} == {"neighborhood", "region"}:
                    continue
                # place <-> county for same-name GA admins: soft-pass
                # (e.g., "Macon" is both a city and a county in GA).
                if {want_level, got_level} == {"place", "county"}:
                    continue
                # MSA <-> place: "Athens GA" could resolve to either.
                if {want_level, got_level} == {"place", "msa"}:
                    continue
                logic_fails.append(
                    f"[{i}] {query!r}: geo_level want={want_level!r} "
                    f"got={got_level!r}"
                )

    # Print the report whether we pass or fail
    summary = [
        f"Gold sweep: {eligible} eligible entries",
        f"  data gaps (no resolution): {len(data_gaps)}",
        f"  logic mismatches:          {len(logic_fails)}",
    ]
    if data_gaps:
        summary.append("\n  Data gaps:")
        summary.extend(f"    {line}" for line in data_gaps[:20])
        if len(data_gaps) > 20:
            summary.append(f"    ... and {len(data_gaps) - 20} more")
    if logic_fails:
        summary.append("\n  Logic mismatches:")
        summary.extend(f"    {line}" for line in logic_fails)
    print("\n".join(summary))

    # Gate only on logic failures
    if len(logic_fails) > 0.10 * eligible:
        raise AssertionError(
            f"{len(logic_fails)}/{eligible} logic failures — resolver is miscategorizing geo_level"
        )
