"""Tests for the Tier 1 ConceptMap loader + coverage validation."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.chatbot.concept_map import (
    ConceptEntry, ConceptMap, ConceptVariables,
    GeoLevelOverride, ResolvedConcept,
)


# ---------------------------------------------------------------------------
# ConceptVariables validation
# ---------------------------------------------------------------------------

def test_variables_value_only_ok():
    v = ConceptVariables(value="B19013_001E")
    assert v.value == "B19013_001E"


def test_variables_numerator_requires_denominator():
    with pytest.raises(ValidationError):
        ConceptVariables(numerator="A_002E")  # no denominator


def test_variables_denominator_without_numerator_rejected():
    with pytest.raises(ValidationError):
        ConceptVariables(denominator="A_001E")


def test_variables_at_least_one_role_required():
    with pytest.raises(ValidationError):
        ConceptVariables()


def test_numerator_string_normalized_to_list():
    v = ConceptVariables(numerator="A_002E", denominator="A_001E")
    assert v.numerator == ["A_002E"]


def test_numerator_list_kept_as_list():
    v = ConceptVariables(
        numerator=["A_002E", "A_003E"], denominator="A_001E",
    )
    assert v.numerator == ["A_002E", "A_003E"]


def test_components_only_ok():
    v = ConceptVariables(components=["A_002E", "A_003E"])
    assert v.components == ["A_002E", "A_003E"]


def test_all_variable_ids_flattens_every_role():
    v = ConceptVariables(
        value="A_001E",
        numerator=["A_002E", "A_003E"],
        denominator="A_010E",
        components=["A_020E"],
    )
    assert set(v.all_variable_ids()) == {
        "A_001E", "A_002E", "A_003E", "A_010E", "A_020E",
    }


# ---------------------------------------------------------------------------
# ConceptMap construction + lookup
# ---------------------------------------------------------------------------

def _entry(canonical: str, aliases: list[str], **kw) -> ConceptEntry:
    return ConceptEntry(
        canonical=canonical,
        aliases=aliases,
        table_id=kw.get("table_id", "B19013"),
        dataset=kw.get("dataset", "acs/acs5"),
        variables=ConceptVariables(value=kw.get("value", "B19013_001E")),
    )


def test_lookup_matches_canonical():
    cmap = ConceptMap([_entry("median household income", [])])
    e = cmap.lookup("median household income")
    assert e is not None
    assert e.canonical == "median household income"


def test_lookup_matches_alias():
    cmap = ConceptMap([_entry("median household income",
                              ["household income", "median income"])])
    assert cmap.lookup("household income") is not None
    assert cmap.lookup("median income") is not None


def test_lookup_is_case_and_whitespace_insensitive():
    cmap = ConceptMap([_entry("median household income", ["HH income"])])
    assert cmap.lookup("MEDIAN  Household   Income") is not None
    assert cmap.lookup("hh income") is not None


def test_lookup_miss_returns_none():
    cmap = ConceptMap([_entry("median household income", [])])
    assert cmap.lookup("never-defined-concept") is None


def test_duplicate_alias_across_entries_rejected():
    e1 = _entry("median household income", ["income"])
    e2 = _entry("per capita income", ["income"])  # collides
    with pytest.raises(ValueError, match="duplicate lookup key"):
        ConceptMap([e1, e2])


def test_duplicate_canonical_across_entries_rejected():
    e1 = _entry("median household income", [])
    e2 = _entry("median household income", [])
    with pytest.raises(ValueError, match="duplicate lookup key"):
        ConceptMap([e1, e2])


def test_alias_collides_with_canonical_rejected():
    e1 = _entry("median household income", [])
    e2 = _entry("per capita income", ["median household income"])
    with pytest.raises(ValueError, match="duplicate lookup key"):
        ConceptMap([e1, e2])


def test_dunder_methods():
    cmap = ConceptMap([
        _entry("a", []), _entry("b", ["b-alias"]), _entry("c", []),
    ])
    assert len(cmap) == 3
    assert "b-alias" in cmap
    assert "missing" not in cmap
    assert [e.canonical for e in cmap] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# YAML round-trip with the real seed file
# ---------------------------------------------------------------------------

def test_seed_yaml_loads_with_expected_size():
    """Seed is curated at ~177 entries spanning income, poverty,
    demographics, race, housing, education, employment, insurance,
    transit, family, language, digital, food assistance. Bound is wide
    so ordinary additions don't break the test."""
    cmap = ConceptMap.from_yaml()
    assert 150 <= len(cmap) <= 300, (
        f"unexpected concept count: {len(cmap)}"
    )


def test_seed_yaml_has_no_duplicate_lookup_keys():
    """The constructor would raise on duplicates — this is just an
    explicit sanity surface so curators see it as a tested invariant."""
    cmap = ConceptMap.from_yaml()
    keys = cmap.all_keys()
    assert len(keys) == len(set(keys))


def test_seed_yaml_passes_full_coverage_validation(metadata_db):
    """Every entry must support tract + county + msa + state at some
    vintage. Catches drift if a curated table loses a level."""
    ConceptMap.from_yaml(
        metadata_db=metadata_db,
        require_coverage_at=["tract", "county", "msa", "state"],
    )


def test_coverage_validation_raises_for_unknown_table(metadata_db, tmp_path):
    """A YAML with a bogus table_id must fail coverage validation with
    every offender listed in the error."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "concepts:\n"
        "  - canonical: ghost concept\n"
        "    aliases: []\n"
        "    table_id: ZZZ99999\n"
        "    dataset: acs/acs5\n"
        "    variables:\n"
        "      value: ZZZ99999_001E\n"
    )
    with pytest.raises(ValueError, match="ghost concept"):
        ConceptMap.from_yaml(
            bad_yaml, metadata_db=metadata_db,
            require_coverage_at=["tract", "county"],
        )


# ---------------------------------------------------------------------------
# GeoLevelOverride + resolve_for
# ---------------------------------------------------------------------------

def test_override_requires_at_least_one_field():
    with pytest.raises(ValidationError):
        GeoLevelOverride()


def test_override_dataset_only_ok():
    o = GeoLevelOverride(dataset="acs/acs1")
    assert o.dataset == "acs/acs1"
    assert o.table_id is None
    assert o.variables is None


def test_resolve_for_no_override_returns_base():
    e = _entry("median income", [])
    r = e.resolve_for("tract")
    assert isinstance(r, ResolvedConcept)
    assert r.table_id == "B19013"
    assert r.dataset == "acs/acs5"


def test_resolve_for_dataset_override_swaps_dataset_keeps_table():
    e = ConceptEntry(
        canonical="median income", aliases=[],
        table_id="B19013", dataset="acs/acs5",
        variables=ConceptVariables(value="B19013_001E"),
        by_geo_level={"msa": GeoLevelOverride(dataset="acs/acs1")},
    )
    msa = e.resolve_for("msa")
    assert msa.dataset == "acs/acs1"
    assert msa.table_id == "B19013"      # inherited
    assert msa.variables.value == "B19013_001E"

    tract = e.resolve_for("tract")        # no override → base
    assert tract.dataset == "acs/acs5"


def test_resolve_for_full_override_swaps_everything():
    e = ConceptEntry(
        canonical="x", aliases=[],
        table_id="B19013", dataset="acs/acs5",
        variables=ConceptVariables(value="B19013_001E"),
        by_geo_level={
            "state": GeoLevelOverride(
                table_id="S1901",
                dataset="acs/acs5/subject",
                variables=ConceptVariables(value="S1901_C01_012E"),
            ),
        },
    )
    r = e.resolve_for("state")
    assert (r.table_id, r.dataset, r.variables.value) == (
        "S1901", "acs/acs5/subject", "S1901_C01_012E",
    )


def test_per_level_validation_catches_bad_override(metadata_db, tmp_path):
    """If an override points at a (table, dataset) that doesn't have
    coverage at the override's level, validation must reject it."""
    bad_yaml = tmp_path / "bad_override.yaml"
    bad_yaml.write_text(
        "concepts:\n"
        "  - canonical: bad override concept\n"
        "    aliases: []\n"
        "    table_id: B19013\n"
        "    dataset: acs/acs5\n"
        "    variables:\n"
        "      value: B19013_001E\n"
        "    by_geo_level:\n"
        "      msa:\n"
        "        table_id: ZZZ99999\n"   # bogus
    )
    with pytest.raises(ValueError, match="bad override concept.*msa"):
        ConceptMap.from_yaml(
            bad_yaml, metadata_db=metadata_db,
            require_coverage_at=["tract", "county", "msa"],
        )


def test_per_level_validation_passes_when_override_has_coverage(
    metadata_db, tmp_path,
):
    """ACS5→ACS1 override at MSA must pass since ACS1 covers MSA."""
    yaml_path = tmp_path / "ok_override.yaml"
    yaml_path.write_text(
        "concepts:\n"
        "  - canonical: median income with msa override\n"
        "    aliases: []\n"
        "    table_id: B19013\n"
        "    dataset: acs/acs5\n"
        "    variables:\n"
        "      value: B19013_001E\n"
        "    by_geo_level:\n"
        "      msa:\n"
        "        dataset: acs/acs1\n"
        "      state:\n"
        "        dataset: acs/acs1\n"
    )
    cmap = ConceptMap.from_yaml(
        yaml_path, metadata_db=metadata_db,
        require_coverage_at=["tract", "county", "msa", "state"],
    )
    assert len(cmap) == 1
