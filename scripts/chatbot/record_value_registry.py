"""Value registry for record-level categorical filters.

Semantic routing answers "which variable is this?"  Record-level filtering also
needs a second answer: "which stored value inside that variable should be used
in SQL?"  This module keeps that mapping dataset/table/variable scoped so new
record-level datasets can bring their own allowed values without hardcoding
them in the planner.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "config" / "record_value_registry.json"


def _key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _value_key(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    return _key(value)


class RecordValueRegistry:
    """Resolve user-facing filter phrases to stored categorical values."""

    def __init__(self, document: dict[str, Any]):
        self.document = document
        self._alias_index: dict[tuple[str, str, str, str], Any] = {}
        self._canonical_index: dict[tuple[str, str, str, str], Any] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        datasets = self.document.get("datasets", {})
        for dataset, dataset_doc in datasets.items():
            for table_id, table_doc in dataset_doc.get("tables", {}).items():
                for variable_id, variable_doc in table_doc.get("variables", {}).items():
                    for entry in variable_doc.get("values", []):
                        canonical = entry.get("canonical_value")
                        if canonical is None:
                            continue
                        self._canonical_index[
                            (dataset, table_id, variable_id, _value_key(canonical))
                        ] = canonical
                        for alias in entry.get("aliases", []):
                            self._alias_index[
                                (dataset, table_id, variable_id, _key(alias))
                            ] = canonical

    def has_variable(
        self,
        *,
        dataset: str,
        table_id: str,
        variable_id: str,
    ) -> bool:
        return bool(
            self.document
            .get("datasets", {})
            .get(dataset, {})
            .get("tables", {})
            .get(table_id, {})
            .get("variables", {})
            .get(variable_id)
        )

    def allowed_values(
        self,
        *,
        dataset: str,
        table_id: str,
        variable_id: str,
    ) -> list[Any]:
        variable_doc = (
            self.document
            .get("datasets", {})
            .get(dataset, {})
            .get("tables", {})
            .get(table_id, {})
            .get("variables", {})
            .get(variable_id, {})
        )
        return [
            entry.get("canonical_value")
            for entry in variable_doc.get("values", [])
            if entry.get("canonical_value") is not None
        ]

    def resolve(
        self,
        *,
        dataset: str,
        table_id: str,
        variable_id: str,
        raw_value: Any,
    ) -> Any:
        """Return the canonical stored value for a routed filter variable.

        Exact canonical-value matches win first, then aliases.  Unknown values
        pass through unchanged so unsupported values are visible in downstream
        plans instead of being silently rewritten.
        """
        if isinstance(raw_value, list):
            return [
                self.resolve(
                    dataset=dataset,
                    table_id=table_id,
                    variable_id=variable_id,
                    raw_value=item,
                )
                for item in raw_value
            ]

        canonical_key = (
            dataset,
            table_id,
            variable_id,
            _value_key(raw_value),
        )
        if canonical_key in self._canonical_index:
            return self._canonical_index[canonical_key]

        alias_key = (
            dataset,
            table_id,
            variable_id,
            _key(raw_value),
        )
        return self._alias_index.get(alias_key, raw_value)


@lru_cache(maxsize=4)
def load_record_value_registry(
    path: Optional[str | Path] = None,
) -> RecordValueRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    with registry_path.open(encoding="utf-8") as file:
        document = json.load(file)
    return RecordValueRegistry(document)


def resolve_record_filter_value(
    *,
    dataset: str,
    table_id: str,
    variable_id: str,
    raw_value: Any,
    registry: Optional[RecordValueRegistry] = None,
) -> Any:
    registry = registry or load_record_value_registry()
    return registry.resolve(
        dataset=dataset,
        table_id=table_id,
        variable_id=variable_id,
        raw_value=raw_value,
    )
