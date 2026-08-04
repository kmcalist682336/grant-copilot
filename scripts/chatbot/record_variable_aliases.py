"""Dataset-scoped aliases for record-level variable routing.

These aliases are guardrails for high-value dimensions where semantic routing
can confuse similar fields, such as HMDA primary applicant vs co-applicant
race/sex/ethnicity.  They are intentionally external to ``record_planner`` so
future record-level datasets can define their own aliases.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIASES_PATH = REPO_ROOT / "config" / "record_variable_aliases.json"


def _key(value: Optional[str]) -> str:
    return " ".join((value or "").strip().lower().split())


class RecordVariableAliases:
    def __init__(self, document: dict):
        self.document = document
        self._index: dict[tuple[str, str, str], str] = {}
        self._build_index()

    def _build_index(self) -> None:
        datasets = self.document.get("datasets", {})
        for dataset, dataset_doc in datasets.items():
            for table_id, table_doc in dataset_doc.get("tables", {}).items():
                aliases = table_doc.get("aliases", {})
                for alias, variable_id in aliases.items():
                    self._index[(dataset, table_id, _key(alias))] = variable_id

    def resolve(
        self,
        *,
        dataset: str,
        table_id: str,
        text: Optional[str],
    ) -> Optional[str]:
        key = _key(text)
        if not key:
            return None
        return self._index.get((dataset, table_id, key))

    def resolve_any(
        self,
        *,
        dataset: str,
        table_id: str,
        texts: list[Optional[str]],
    ) -> Optional[str]:
        for text in texts:
            variable_id = self.resolve(
                dataset=dataset,
                table_id=table_id,
                text=text,
            )
            if variable_id is not None:
                return variable_id
        return None


@lru_cache(maxsize=4)
def load_record_variable_aliases(
    path: Optional[str | Path] = None,
) -> RecordVariableAliases:
    aliases_path = Path(path) if path is not None else DEFAULT_ALIASES_PATH
    with aliases_path.open(encoding="utf-8") as file:
        document = json.load(file)
    return RecordVariableAliases(document)


def resolve_record_variable_alias(
    *,
    dataset: str,
    table_id: str,
    texts: list[Optional[str]],
    aliases: Optional[RecordVariableAliases] = None,
) -> Optional[str]:
    aliases = aliases or load_record_variable_aliases()
    return aliases.resolve_any(
        dataset=dataset,
        table_id=table_id,
        texts=texts,
    )
