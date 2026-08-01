"""Curated metric recipes for record-level datasets.

This module is intentionally parallel to ``concept_map.py`` rather than a
replacement for it. Census concept-map entries resolve concepts to Census API
variables. Record metric entries resolve concepts to deterministic record
queries: selected variables, denominator filters, numerator-only filters, and
aggregation operation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.chatbot.census_caller import RecordFilter, RecordFilterOperator
from scripts.chatbot.models import ExtractedConcept


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "record_metric_map.yaml"

RecordOperation = Literal[
    "value", "count", "sum", "average", "median", "percentage",
]
RecordUnits = Literal["dollars", "percent", "count", "years", "ratio", "index"]


def _key(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


class RecordMetricMeasure(BaseModel):
    """The real record variable used to compute a metric."""

    model_config = ConfigDict(extra="forbid")

    text: str
    canonical_hint: Optional[str] = None
    variable_id: str

    def concept(self, *, dataset: str) -> ExtractedConcept:
        return ExtractedConcept(
            text=self.text,
            canonical_hint=self.canonical_hint or self.text,
            dataset_hint=dataset,
        )


class RecordMetricPredicate(BaseModel):
    """A deterministic predicate used by a record metric recipe."""

    model_config = ConfigDict(extra="forbid")

    variable_id: str
    dimension: str
    canonical_hint: Optional[str] = None
    operator: RecordFilterOperator = "equals"
    value: Any = None

    @model_validator(mode="after")
    def _check_value_shape(self) -> "RecordMetricPredicate":
        if self.operator in {"equals", "not_equals"} and self.value is None:
            raise ValueError(f"{self.operator} requires a value")
        if self.operator in {"in", "not_in"}:
            if isinstance(self.value, (str, bytes)) or self.value is None:
                raise ValueError(f"{self.operator} requires a non-string list")
            values = list(self.value)
            if not values:
                raise ValueError(f"{self.operator} requires at least one value")
            object.__setattr__(self, "value", values)
        if self.operator in {"is_null", "is_not_null"}:
            object.__setattr__(self, "value", None)
        return self

    def concept(self, *, dataset: str) -> ExtractedConcept:
        return ExtractedConcept(
            text=self.dimension,
            canonical_hint=self.canonical_hint or self.dimension,
            dataset_hint=dataset,
        )

    def record_filter(self) -> RecordFilter:
        return RecordFilter(
            variable_id=self.variable_id,
            operator=self.operator,
            value=self.value,
        )


class RecordMetricRecipe(BaseModel):
    """One curated metric definition for a record-level dataset."""

    model_config = ConfigDict(extra="forbid")

    canonical: str
    aliases: list[str] = Field(default_factory=list)
    dataset: str
    table_id: str
    operation: RecordOperation
    measure: RecordMetricMeasure
    record_filters: list[RecordMetricPredicate] = Field(default_factory=list)
    numerator_filters: list[RecordMetricPredicate] = Field(default_factory=list)
    denominator_description: Optional[str] = None
    units: Optional[RecordUnits] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_percentage_shape(self) -> "RecordMetricRecipe":
        if self.operation == "percentage" and not self.numerator_filters:
            raise ValueError("percentage record metrics need numerator_filters")
        return self

    def lookup_keys(self) -> list[str]:
        return [self.canonical, *self.aliases]

    def measure_concept(self) -> ExtractedConcept:
        return self.measure.concept(dataset=self.dataset)

    def record_filter_objects(self) -> list[RecordFilter]:
        return [predicate.record_filter() for predicate in self.record_filters]

    def numerator_filter_objects(self) -> list[RecordFilter]:
        return [predicate.record_filter() for predicate in self.numerator_filters]


class RecordMetricMap:
    """Lookup table for deterministic record-level metric recipes."""

    def __init__(self, recipes: list[RecordMetricRecipe]) -> None:
        self.recipes = recipes
        self._lookup: dict[str, RecordMetricRecipe] = {}
        for recipe in recipes:
            for key in recipe.lookup_keys():
                normalized = _key(key)
                if normalized:
                    self._lookup[normalized] = recipe

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "RecordMetricMap":
        with path.open(encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
        recipes = [
            RecordMetricRecipe.model_validate(item)
            for item in document.get("metrics", [])
        ]
        return cls(recipes)

    def lookup(self, text: Optional[str]) -> Optional[RecordMetricRecipe]:
        key = _key(text)
        if not key:
            return None
        if key in self._lookup:
            return self._lookup[key]
        matches = [
            recipe for phrase, recipe in self._lookup.items()
            if phrase and phrase in key
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda recipe: max(len(_key(k)) for k in recipe.lookup_keys()),
        )

    def lookup_any(self, texts: list[Optional[str]]) -> Optional[RecordMetricRecipe]:
        for text in texts:
            recipe = self.lookup(text)
            if recipe is not None:
                return recipe
        return None

    def summaries(self) -> list[dict[str, Any]]:
        """Compact descriptions suitable for prompts/debugging."""
        return [
            {
                "canonical": recipe.canonical,
                "aliases": recipe.aliases,
                "dataset": recipe.dataset,
                "table_id": recipe.table_id,
                "operation": recipe.operation,
                "measure_variable_id": recipe.measure.variable_id,
                "record_filters": [p.model_dump(mode="json") for p in recipe.record_filters],
                "numerator_filters": [p.model_dump(mode="json") for p in recipe.numerator_filters],
                "units": recipe.units,
                "notes": recipe.notes,
            }
            for recipe in self.recipes
        ]


@lru_cache(maxsize=1)
def load_default_record_metric_map() -> RecordMetricMap:
    return RecordMetricMap.load(DEFAULT_PATH)
