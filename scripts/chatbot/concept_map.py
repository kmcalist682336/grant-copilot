"""Tier 1: curated concept→Census-table map.

This is the fast path between Tier 0 query expansion and Tier 2 FTS
metadata search. About 50 hand-curated entries cover the common ACS
concepts (income, poverty, housing, education, employment, demographics)
that drive the bulk of user questions. A hit here means we know exactly
which table + variables to fetch — no LLM, no FTS scoring required.

YAML schema (see config/concept_map.yaml for the seed list)::

    concepts:
      - canonical: "median household income"
        aliases: ["household income", "median income"]
        table_id: "B19013"
        dataset: "acs/acs5"
        variables:
          value: "B19013_001E"
        units: "dollars"
        notes: "..."

      - canonical: "poverty rate"
        aliases: ["poverty"]
        table_id: "B17001"
        dataset: "acs/acs5"
        variables:
          numerator: "B17001_002E"
          denominator: "B17001_001E"
        units: "percent"

Variables roles:
  value        — single estimate (e.g., median income, total population).
  numerator    — single var or list[str] summed in the numerator.
  denominator  — single var; required whenever numerator is set.
  components   — list[str] read together for breakdowns (e.g., race shares).

Public API:
    ConceptVariables, ConceptEntry  — Pydantic records
    ConceptMap                       — loader + lookup
    DEFAULT_PATH                     — config/concept_map.yaml
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "concept_map.yaml"


Units = Literal[
    "dollars", "percent", "count", "years", "minutes", "index", "ratio",
]


# ---------------------------------------------------------------------------
# Pydantic
# ---------------------------------------------------------------------------

class ConceptVariables(BaseModel):
    """Which Census variables to pull, and how to combine them."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[str] = Field(
        default=None,
        description="Single estimate variable (e.g., 'B19013_001E').",
    )
    numerator: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Summed in the percentage numerator. str or list[str].",
    )
    denominator: Optional[str] = Field(
        default=None,
        description="Universe total; required when numerator is set.",
    )
    components: Optional[list[str]] = Field(
        default=None,
        description="Variables read together for a breakdown (e.g., race "
                    "shares). Distinct from numerator/denominator.",
    )

    @model_validator(mode="after")
    def _check_role_consistency(self) -> "ConceptVariables":
        roles_set = sum(
            1 for v in (self.value, self.numerator, self.components)
            if v is not None
        )
        if roles_set == 0:
            raise ValueError(
                "ConceptVariables needs at least one of: value, "
                "numerator, components."
            )
        if self.numerator is not None and self.denominator is None:
            raise ValueError(
                "ConceptVariables.numerator requires a denominator."
            )
        if self.numerator is None and self.denominator is not None:
            raise ValueError(
                "ConceptVariables.denominator was set without numerator."
            )
        # Normalize numerator to list for downstream consumers.
        if isinstance(self.numerator, str):
            object.__setattr__(self, "numerator", [self.numerator])
        return self

    def all_variable_ids(self) -> list[str]:
        """Flatten every variable_id this entry references."""
        out: list[str] = []
        if self.value:
            out.append(self.value)
        if isinstance(self.numerator, list):
            out.extend(self.numerator)
        if self.denominator:
            out.append(self.denominator)
        if self.components:
            out.extend(self.components)
        return out


class GeoLevelOverride(BaseModel):
    """Sparse override of one or more base fields for a specific geo level.

    Used inside ``ConceptEntry.by_geo_level``. Common case: same table,
    different dataset (e.g., ACS5 at tract → ACS1 at MSA/state for
    currency). Any field left None inherits from the entry's base.
    """

    model_config = ConfigDict(extra="forbid")

    table_id: Optional[str] = None
    dataset: Optional[str] = None
    variables: Optional[ConceptVariables] = None

    @model_validator(mode="after")
    def _at_least_one_change(self) -> "GeoLevelOverride":
        if self.table_id is None and self.dataset is None and self.variables is None:
            raise ValueError(
                "GeoLevelOverride must change at least one of "
                "table_id, dataset, variables (otherwise omit the entry)."
            )
        return self


class ResolvedConcept(BaseModel):
    """Effective (table, dataset, variables) after applying any overrides
    for a specific geo level. Returned by ``ConceptEntry.resolve_for``.
    """

    model_config = ConfigDict(extra="ignore")

    table_id: str
    dataset: str
    variables: ConceptVariables


class ConceptEntry(BaseModel):
    """One curated concept entry."""

    model_config = ConfigDict(extra="forbid")

    canonical: str = Field(
        ..., description="Canonical name. Shown to users in citations.",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate phrasings the lookup will match.",
    )
    table_id: str = Field(
        ..., description="Census table_id (e.g., 'B19013').",
    )
    dataset: str = Field(
        default="acs/acs5",
        description="Census dataset; defaults to ACS 5-year for tract coverage.",
    )
    variables: ConceptVariables
    by_geo_level: dict[str, GeoLevelOverride] = Field(
        default_factory=dict,
        description=(
            "Sparse per-geo-level overrides. Keys are geo levels "
            "(tract, county, msa, state, ...). Each value may change "
            "table_id, dataset, and/or variables. Levels not listed "
            "use the base config."
        ),
    )
    universe: Optional[str] = Field(
        default=None,
        description="Stat universe (e.g., 'Households', 'People 25+').",
    )
    units: Optional[Units] = None
    notes: Optional[str] = None

    def lookup_keys(self) -> list[str]:
        """All terms (canonical + aliases) this entry should match on."""
        return [self.canonical, *self.aliases]

    def resolve_for(self, geo_level: str) -> ResolvedConcept:
        """Return the effective (table_id, dataset, variables) for the
        requested geo_level after applying any sparse override."""
        ovr = self.by_geo_level.get(geo_level)
        if ovr is None:
            return ResolvedConcept(
                table_id=self.table_id,
                dataset=self.dataset,
                variables=self.variables,
            )
        return ResolvedConcept(
            table_id=ovr.table_id or self.table_id,
            dataset=ovr.dataset or self.dataset,
            variables=ovr.variables or self.variables,
        )


class _ConceptFile(BaseModel):
    """Top-level YAML wrapper."""
    model_config = ConfigDict(extra="forbid")
    concepts: list[ConceptEntry]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _normalize_lookup(s: str) -> str:
    """Casing + whitespace normalization for lookup keys.

    Intentionally simple — Tier 1 is for exact-or-alias hits; fuzzier
    matching is Tier 0/2 territory.
    """
    return " ".join(s.lower().split())


class ConceptMap:
    """In-memory index over ConceptEntry rows.

    Keys are case-insensitive and whitespace-normalized. Construction
    rejects duplicate keys across entries so the seed YAML can't quietly
    shadow itself.
    """

    def __init__(self, entries: list[ConceptEntry]):
        self._entries: list[ConceptEntry] = list(entries)
        self._index: dict[str, ConceptEntry] = {}
        for entry in self._entries:
            for key in entry.lookup_keys():
                norm = _normalize_lookup(key)
                if not norm:
                    raise ValueError(
                        f"empty lookup key under concept "
                        f"{entry.canonical!r}"
                    )
                if norm in self._index:
                    other = self._index[norm].canonical
                    raise ValueError(
                        f"duplicate lookup key {norm!r}: appears under "
                        f"{other!r} and {entry.canonical!r}"
                    )
                self._index[norm] = entry

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Optional[Path] = None,
        *,
        metadata_db: Optional[sqlite3.Connection] = None,
        require_coverage_at: Optional[list[str]] = None,
    ) -> "ConceptMap":
        """Load + validate the concept_map YAML.

        Args:
            path: YAML location (default: ``config/concept_map.yaml``).
            metadata_db: Optional metadata search DB. When given alongside
                ``require_coverage_at``, every entry's
                ``(table_id, dataset)`` is checked for at least one year
                with successful coverage at every listed geo_level. Bad
                entries surface as a single ``ValueError`` listing them
                all — the seed file can't drift away from reality.
            require_coverage_at: Geo levels every entry must support
                (e.g., ``["tract", "county", "msa"]``).
        """
        path = Path(path) if path is not None else DEFAULT_PATH
        with path.open() as f:
            doc = yaml.safe_load(f) or {}
        parsed = _ConceptFile.model_validate(doc)
        cmap = cls(parsed.concepts)
        if metadata_db is not None and require_coverage_at:
            cmap.validate_coverage(metadata_db, require_coverage_at)
        return cmap

    # ------------------------------------------------------------------
    # Coverage validation (against tools/import_v1_coverage.py output)
    # ------------------------------------------------------------------

    def validate_coverage(
        self,
        metadata_db: sqlite3.Connection,
        require_coverage_at: list[str],
    ) -> None:
        """Raise ValueError if any entry's effective (table, dataset) for
        any required geo_level lacks successful coverage at that level
        in some year.

        Per-level: applies ``by_geo_level`` overrides, so a concept whose
        base is acs/acs5 + B19013 but whose msa override switches to
        acs/acs1 must have acs/acs1+B19013 coverage at msa specifically.

        Lists every offender at once so curators don't play whack-a-mole.
        """
        # Lazy import keeps concept_map importable when metadata_search
        # isn't around (loader-only unit tests).
        from scripts.chatbot.metadata_search import find_supported_years

        bad: list[str] = []
        for entry in self._entries:
            for level in require_coverage_at:
                eff = entry.resolve_for(level)
                years = find_supported_years(
                    metadata_db, eff.table_id, eff.dataset, [level],
                )
                if not years:
                    overridden = level in entry.by_geo_level
                    suffix = " (via override)" if overridden else ""
                    bad.append(
                        f"{entry.canonical!r} @ {level}: "
                        f"{eff.dataset}/{eff.table_id} has NO year "
                        f"with coverage{suffix}"
                    )
        if bad:
            joined = "\n  ".join(bad)
            raise ValueError(
                f"concept_map has {len(bad)} (entry, geo_level) pairs "
                f"without required coverage:\n  {joined}"
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, concept: str) -> Optional[ConceptEntry]:
        """Return the entry whose canonical or any alias matches ``concept``
        (case-insensitive, whitespace-normalized). None on miss."""
        return self._index.get(_normalize_lookup(concept))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ConceptEntry]:
        return iter(self._entries)

    def __contains__(self, concept: str) -> bool:
        return _normalize_lookup(concept) in self._index

    def all_keys(self) -> list[str]:
        """Every canonical+alias key (already normalized). Used by tests
        and the Phase J seeder."""
        return list(self._index.keys())
