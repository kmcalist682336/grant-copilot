"""Composer tests — intersect, directional, difference, and multi-compound."""
from __future__ import annotations

from scripts.chatbot.models import (
    ExtractedGeoRef,
    ExtractedIntent,
    CompoundSpec,
)
from scripts.chatbot.geo_resolver import resolve_intent


def test_intersect_i85_in_downtown(gazetteer_db):
    """I-85 tracts intersected with Downtown should be non-empty at 2mi."""
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="I-85", ref_type="corridor",
                            qualifier="near",
                            proximity_tier="very_local"),
            ExtractedGeoRef(text="downtown Atlanta", ref_type="region",
                            scope_hint="Atlanta"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="intersect")],
        intent_type="proximity",
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 1
    result = out[0]
    assert result.geo_level == "corridor"
    assert "Downtown" in result.display_name or "downtown" in result.display_name.lower()
    assert len(result.tract_geoids) > 0
    assert any("filtered to tracts within" in n.lower()
               for n in result.assumption_notes)
    # Confidence was penalized vs either piece alone
    assert result.confidence < 1.0


def test_intersect_empty_surface_note(gazetteer_db):
    """An intersect that produces zero tracts should surface a note, not crash."""
    # Deliberately nonsense filter — "Savannah" doesn't overlap I-85.
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="I-85", ref_type="corridor",
                            qualifier="near"),
            ExtractedGeoRef(text="Savannah", ref_type="administrative"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="intersect")],
        intent_type="proximity",
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 1
    result = out[0]
    # Savannah is a city admin → tracts derived via _ensure_tracts. Either
    # the intersection is non-empty (Savannah ∩ I-85 buffer) or empty with
    # a "no tracts" note.
    assert result.data_level_available in ("none", "tract")
    if not result.tract_geoids:
        assert any("no tracts" in n.lower() or "widen" in n.lower()
                   for n in result.assumption_notes)


def test_directional_north_of_buckhead(gazetteer_db):
    """North slice of Buckhead: should yield fewer tracts than all Buckhead."""
    # Control — full Buckhead
    base_intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                                    scope_hint="Atlanta")],
    )
    base = resolve_intent(base_intent, gazetteer_db)
    assert base
    n_all = len(base[0].tract_geoids)

    # Directional — north side
    dir_intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                                    scope_hint="Atlanta")],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=0,
                                op="directional", direction="n")],
    )
    out = resolve_intent(dir_intent, gazetteer_db)
    assert len(out) == 1
    result = out[0]
    n_north = len(result.tract_geoids)
    # Should be a proper subset (some but not all)
    assert 0 < n_north < n_all
    assert "side" in result.display_name.lower()
    assert any("filtered to tracts on the n side" in n.lower()
               for n in result.assumption_notes)


def test_directional_south_of_buckhead_complements_north(gazetteer_db):
    """N and S directional slices of the same anchor should be disjoint."""
    def run(direction: str) -> set[str]:
        intent = ExtractedIntent(
            geo_refs=[ExtractedGeoRef(
                text="Buckhead", ref_type="neighborhood",
                scope_hint="Atlanta",
            )],
            compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=0,
                                    op="directional", direction=direction)],
        )
        out = resolve_intent(intent, gazetteer_db)
        assert out
        return set(out[0].tract_geoids)

    north = run("n")
    south = run("s")
    # They must not overlap — a tract's centroid can't be both N and S of anchor
    assert not (north & south)
    # And the union should be a subset of the full Buckhead tract set
    # (some tracts near the anchor latitude may fall in neither slice)
    both = north | south
    assert len(both) > 0


def test_compound_primary_missing_returns_empty(gazetteer_db):
    """If the primary ref can't be resolved, the compound returns nothing."""
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="xyzzzq", ref_type="administrative"),
            ExtractedGeoRef(text="Atlanta", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="intersect")],
    )
    out = resolve_intent(intent, gazetteer_db)
    # Primary failed to resolve → no compound result
    assert out == []


# ---------------------------------------------------------------------------
# Difference op
# ---------------------------------------------------------------------------

def test_difference_dekalb_minus_decatur(gazetteer_db):
    """DeKalb County minus the city of Decatur: should drop Decatur's
    tracts but keep the rest of DeKalb."""
    from scripts.chatbot.composer import _ensure_tracts

    # Resolve the two pieces standalone to know the derived tract sets.
    dekalb = resolve_intent(
        ExtractedIntent(geo_refs=[ExtractedGeoRef(
            text="DeKalb County", ref_type="administrative",
            scope_hint="Georgia",
        )]),
        gazetteer_db,
    )[0]
    decatur = resolve_intent(
        ExtractedIntent(geo_refs=[ExtractedGeoRef(
            text="Decatur", ref_type="administrative",
            scope_hint="Georgia",
        )]),
        gazetteer_db,
    )[0]
    dekalb_tracts, _ = _ensure_tracts(dekalb, gazetteer_db)
    decatur_tracts, _ = _ensure_tracts(decatur, gazetteer_db)
    assert len(dekalb_tracts) > 50  # sanity — DeKalb has ~150 tracts
    assert len(decatur_tracts) > 0  # sanity — Decatur has tracts

    # Difference — minus Decatur
    diff_intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="DeKalb County", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="Decatur", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="difference")],
    )
    out = resolve_intent(diff_intent, gazetteer_db)
    assert len(out) == 1
    result = out[0]
    assert "excluding" in result.display_name.lower()
    assert "Decatur" in result.display_name
    # Strict subset of derived DeKalb. Should be smaller iff Decatur shares
    # at least one tract with DeKalb (which it does — Decatur is in DeKalb).
    assert set(result.tract_geoids).issubset(set(dekalb_tracts))
    if set(decatur_tracts) & set(dekalb_tracts):
        assert len(result.tract_geoids) < len(dekalb_tracts)
    assert any("excluded" in n.lower() for n in result.assumption_notes)


def test_difference_nonoverlapping_filter_is_noop_with_note(gazetteer_db):
    """Subtracting a filter that doesn't overlap shouldn't change tract count
    but should attach an explanatory note."""
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="DeKalb County", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="Savannah", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="difference")],
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 1
    result = out[0]
    # Savannah is hundreds of miles away — no overlap.
    assert any("no effect" in n.lower() or "did not overlap" in n.lower()
               for n in result.assumption_notes)


# ---------------------------------------------------------------------------
# Multi-compound — the user-reported failure case
# ---------------------------------------------------------------------------

def test_multi_compound_three_operands(gazetteer_db):
    """Buckhead + SW Atlanta + (DeKalb minus Decatur) = three results.

    This is the regression that prompted the compounds-as-list refactor.
    Before the fix, the resolver returned only the first compound and
    silently dropped Buckhead and the difference operand.
    """
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="Buckhead", ref_type="neighborhood",
                            scope_hint="Atlanta"),
            ExtractedGeoRef(text="Atlanta", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="DeKalb county", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="Decatur", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[
            CompoundSpec(primary_ref_idx=1, filter_ref_idx=1,
                         op="directional", direction="sw"),
            CompoundSpec(primary_ref_idx=2, filter_ref_idx=3,
                         op="difference"),
        ],
        intent_type="comparison",
        comparison_implied=True,
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 3, (
        "expected 3 operands (Buckhead, SW Atlanta, DeKalb-Decatur); "
        f"got {len(out)}: {[r.display_name for r in out]}"
    )

    # Output order follows geo_ref order: Buckhead first (free ref at idx 0),
    # SW Atlanta second (compound at idx 1), DeKalb-Decatur third (compound
    # at idx 2). Decatur (idx 3) is consumed as the filter side.
    names = [r.display_name for r in out]
    assert "Buckhead" in names[0]
    assert "sw side" in names[1].lower() and "Atlanta" in names[1]
    assert "excluding" in names[2].lower() and "Decatur" in names[2]


def test_filter_only_ref_not_emitted_standalone(gazetteer_db):
    """A ref that's only used as a compound's filter must not appear as
    its own standalone resolution."""
    intent = ExtractedIntent(
        geo_refs=[
            ExtractedGeoRef(text="DeKalb County", ref_type="administrative",
                            scope_hint="Georgia"),
            ExtractedGeoRef(text="Decatur", ref_type="administrative",
                            scope_hint="Georgia"),
        ],
        compounds=[CompoundSpec(primary_ref_idx=0, filter_ref_idx=1,
                                op="difference")],
    )
    out = resolve_intent(intent, gazetteer_db)
    assert len(out) == 1
    # Decatur was the filter — it shouldn't surface as its own row.
    for r in out:
        assert r.display_name != "Decatur"
