"""Shared parsing of extracted scalar and list filter values."""
import json
from typing import Any
from scripts.chatbot.models import ExtractedFilter

def _decoded_value(filter_item: ExtractedFilter) -> Any:
    """Parse extracted scalar/list values without dataset-specific aliases.

    LLM extraction sometimes returns list-valued normalized hints as a JSON
    string (for example ``["Loan originated", "Application approved but not
    accepted"]``).  Normalize that before building RecordFilter objects so
    DuckDB sees a real iterable for IN predicates, not one literal string.
    """
    raw_value = (
        filter_item.normalized_value_hint
        or filter_item.value_text
        or ""
    )
    if isinstance(raw_value, list):
        return [str(value).strip() for value in raw_value]
    raw = str(raw_value).strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(value).strip() for value in parsed]
    return raw

