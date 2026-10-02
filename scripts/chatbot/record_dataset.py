"""Dataset capabilities consumed by the shared record planner.

Load a metadata definition explicitly; unknown datasets never inherit another
dataset's fields, geographic defaults, or year coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import yaml

from scripts.chatbot.record_metric_map import RecordMetricMap, load_default_record_metric_map
from scripts.chatbot.record_variable_aliases import resolve_record_variable_alias
from scripts.chatbot.record_value_registry import resolve_record_filter_value


@dataclass
class RecordDatasetDefinition:
    dataset: str
    table_id: str
    supported_years: tuple[int, ...] = ()
    coverage_level: str = "record"
    geography_column: str = "area_code"
    geography_match: str = "exact"
    geography_levels: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    categories: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_map: RecordMetricMap | None = None
    geography_strategy: str = "identity"
    default_geography_spec: dict[str, Any] | None = None
    ambiguous_terms: dict[str, list[str]] = field(default_factory=dict)
    ambiguous_filter_help: str = "provide a supported, explicit category"
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def vocabulary(self):
        from scripts.chatbot.record_variable_aliases import load_record_variable_aliases
        from scripts.chatbot.record_value_registry import load_record_value_registry
        def table(doc):
            return doc.get("datasets", {}).get(self.dataset, {}).get("tables", {}).get(self.table_id, {})
        aliases = {**table(load_record_variable_aliases().document).get("aliases", {}), **self.aliases}
        variables = {**table(load_record_value_registry().document).get("variables", {}), **self.variables}
        for variable, values in self.categories.items():
            variables[variable] = {"values": [
                {"canonical_value": v, "aliases": [k]} for k, v in values.items()]}
        for variable in variables:
            aliases.setdefault(variable, variable)
        return aliases, variables

    def recipes(self):
        source = self.metric_map or load_default_record_metric_map()
        return RecordMetricMap([r for r in source.recipes
                                if r.dataset == self.dataset and r.table_id == self.table_id])

    def variable_alias(self, texts):
        aliases, _ = self.vocabulary()
        for text in texts:
            key = " ".join((text or "").lower().split())
            if key in aliases:
                return aliases[key]
        return resolve_record_variable_alias(dataset=self.dataset, table_id=self.table_id, texts=texts)

    def filter_value(self, variable_id, value):
        if isinstance(value, list):
            result = []
            for item in value:
                normalized = self.filter_value(variable_id, item)
                result.extend(normalized if isinstance(normalized, list) else [normalized])
            return result
        _, definitions = self.vocabulary()
        definition = definitions.get(variable_id, {})
        for entry in definition.get("values", []):
            canonical = entry.get("canonical_value")
            if value == canonical or str(value).strip().lower() in [str(v).strip().lower() for v in entry.get("aliases", [])]:
                return canonical
        if definition.get("values"):
            raise ValueError(f"Unsupported {self.dataset}/{variable_id} category: {value!r}")
        choices = self.categories.get(variable_id)
        if choices is not None:
            key = str(value).strip().lower()
            if key in choices:
                return choices[key]
            if value in choices.values():
                return value
            raise ValueError(f"Unsupported {self.dataset}/{variable_id} category: {value!r}")
        return resolve_record_filter_value(dataset=self.dataset, table_id=self.table_id,
                                           variable_id=variable_id, raw_value=value)

    def metric_recipe(self, analysis, metric_map):
        if analysis.measure is None:
            return None
        texts = [analysis.measure.canonical_hint, analysis.measure.text]
        prefix = {"average": "average", "median": "median", "sum": "total", "count": "count of"}.get(analysis.operation)
        if prefix:
            texts += [f"{prefix} {t}" for t in texts if t]
        direct = metric_map.lookup_any(texts)
        if analysis.operation != "percentage" or (direct is not None and direct.operation == "percentage"):
            return direct
        # Match an extracted outcome against declared recipe predicates, not
        # dataset-specific status strings. Exact sets avoid confusing approval
        # (two outcomes) with origination (one outcome).
        from scripts.chatbot.record_values import _decoded_value
        for item in analysis.filters:
            variable = self.variable_alias([item.dimension.canonical_hint, item.dimension.text])
            if not variable or item.operator not in {"equals", "in"}:
                continue
            value = self.filter_value(variable, _decoded_value(item))
            values = set(map(str, value if isinstance(value, list) else [value]))
            for recipe in metric_map.recipes:
                if recipe.operation != "percentage" or len(recipe.numerator_filters) != 1:
                    continue
                predicate = recipe.numerator_filters[0]
                expected = predicate.value if isinstance(predicate.value, list) else [predicate.value]
                if predicate.variable_id == variable and predicate.operator in {"equals", "in"} and values == set(map(str, expected)):
                    return recipe
        return direct

    def default_geography(self):
        from scripts.chatbot.models import ResolvedGeography
        return ResolvedGeography.model_validate(self.default_geography_spec) if self.default_geography_spec else None

    def interpretation_context(self, query, intent):
        """Small dataset-scoped vocabulary for one interpretation call."""
        aliases, variables = self.vocabulary()
        text = " " + query.lower() + " "
        selected = set()
        for alias, variable in aliases.items():
            if alias.lower() in text:
                selected.add(variable)
        for variable, definition in variables.items():
            for entry in definition.get("values", []):
                terms = [*entry.get("aliases", []), str(entry.get("canonical_value", ""))]
                if any(" " + term.lower() + " " in text for term in terms if term):
                    selected.add(variable)
        recipes = self.recipes()
        matched = []
        for analysis in intent.analyses:
            if analysis.measure is None or analysis.measure.dataset_hint not in {self.dataset, "both", "unknown"}:
                continue
            recipe = self.metric_recipe(analysis, recipes)
            if recipe is not None and recipe not in matched:
                matched.append(recipe)
            for concept in [analysis.measure, *analysis.groupings, *[f.dimension for f in analysis.filters]]:
                variable = self.variable_alias([concept.canonical_hint, concept.text])
                if variable:
                    selected.add(variable)
        for recipe in matched:
            selected.add(recipe.measure.variable_id)
            selected.update(p.variable_id for p in [*recipe.record_filters, *recipe.numerator_filters])
        selected = set(sorted(selected)[:16])
        return {
            "dataset": self.dataset, "table_id": self.table_id,
            "supported_years": list(self.supported_years),
            "geography_column": self.geography_column,
            "variable_aliases": {k: v for k, v in aliases.items() if v in selected},
            "available_fields": {k: {"description": v.get("description", v.get("label", "")),
                                     "aliases": [a for a, target in aliases.items() if target == k],
                                     "values": v.get("values", [])}
                                 for k, v in {**{v: {} for v in aliases.values()}, **variables}.items()},
            "variables": {k: v for k, v in variables.items() if k in selected},
            "metric_recipes": [r.model_dump(mode="json") for r in matched],
        }

    def geography_values(self, geo, geo_db):
        if self.geography_strategy == "tract_prefix":
            from scripts.chatbot.record_geography import _record_geo_prefixes
            values = _record_geo_prefixes(geo, geo_db)
            if not values:
                raise ValueError(f"{self.dataset} geography could not be mapped to tract prefixes")
            return values
        if geo.geo_level not in self.geography_levels:
            raise ValueError(f"{self.dataset} does not support geography level {geo.geo_level!r}")
        return [str(geo.geo_id)]

    def percentage_measure(self, analysis):
        return analysis.measure

    def rate_label(self, filters):
        for recipe in self.recipes().recipes:
            if recipe.numerator_filter_objects() == filters:
                return recipe.canonical
        return None

    def ambiguous_filter(self, filter_item):
        dim = (filter_item.dimension.canonical_hint or filter_item.dimension.text).lower()
        value = (filter_item.normalized_value_hint or filter_item.value_text or "").lower()
        return any(term in dim and value in values for term, values in self.ambiguous_terms.items())


_ADAPTERS: dict[str, RecordDatasetDefinition] = {}
_BUILTINS_LOADED = False


def register_record_dataset(adapter: RecordDatasetDefinition):
    if not adapter.dataset or adapter.dataset in {"unknown", "both", "census"}:
        raise ValueError("A concrete record dataset ID is required")
    if adapter.geography_match not in {"exact", "prefix"}:
        raise ValueError("geography_match must be exact or prefix")
    if adapter.geography_strategy not in {"identity", "tract_prefix"}:
        raise ValueError("Unsupported geography strategy")
    _ADAPTERS[adapter.dataset] = adapter


def load_record_datasets(path=None):
    """Register dataset definitions from YAML; no per-dataset import or code."""
    from scripts.chatbot.record_metric_map import RecordMetricRecipe
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "config/record_datasets.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    definitions = []
    for dataset, spec in document["datasets"].items():
        spec = dict(spec)
        recipes = spec.pop("metrics", None)
        if recipes is not None:
            spec["metric_map"] = RecordMetricMap([
                RecordMetricRecipe.model_validate({**r, "dataset": dataset, "table_id": spec["table_id"]})
                for r in recipes])
        definition = RecordDatasetDefinition(dataset=dataset, **spec)
        definitions.append(definition)
    for definition in definitions:
        register_record_dataset(definition)
    return definitions


def get_record_dataset(dataset: str) -> RecordDatasetDefinition:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        load_record_datasets()
        _BUILTINS_LOADED = True
    try:
        return _ADAPTERS[dataset]
    except KeyError:
        raise ValueError(f"No record adapter/definition registered for {dataset!r}") from None


def record_dataset_definitions() -> dict[str, RecordDatasetDefinition]:
    """Snapshot of configured record sources available to dataset selection."""
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        load_record_datasets()
        _BUILTINS_LOADED = True
    return dict(_ADAPTERS)


# Compatibility for callers using the old name; this is one shared data class,
# not a requirement to implement or subclass an adapter for each dataset.
RecordDatasetAdapter = RecordDatasetDefinition
