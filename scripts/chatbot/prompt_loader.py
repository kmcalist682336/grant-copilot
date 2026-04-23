"""Load and assemble the chatbot extraction prompt.

The YAML template at ``prompts/v1/chatbot/geo_extraction.yaml`` carries the
system prompt and few-shot examples. The JSON schema is NOT stored in the
YAML — it's derived from ``scripts.chatbot.models.ExtractedIntent`` at
runtime so prompt and contract cannot drift.

Public API:
    load_prompt_template(path)   -> dict with 'system', 'few_shots'
    render_system_prompt(template, schema) -> str ready for the LLM
    load_and_render(path=None)   -> (system_prompt_str, response_schema_dict)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from scripts.chatbot.models import extracted_intent_json_schema


DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts" / "v1" / "geo_extraction.yaml"
)


def load_prompt_template(path: Path) -> dict[str, Any]:
    """Load the raw YAML template. Validates that required sections exist."""
    with path.open() as f:
        doc = yaml.safe_load(f)

    for key in ("system", "few_shots"):
        if key not in doc:
            raise ValueError(
                f"Prompt template at {path} is missing required key {key!r}"
            )
    if not isinstance(doc["few_shots"], list) or not doc["few_shots"]:
        raise ValueError("few_shots must be a non-empty list")
    return doc


def render_system_prompt(
    template: dict[str, Any],
    response_schema: dict[str, Any],
) -> str:
    """Assemble the final system prompt string the LLM will see.

    Format:
        <system text>

        === JSON SCHEMA ===
        <pretty-printed schema>

        === EXAMPLES ===
        Input: "..."
        Output: {...}

        ...
    """
    parts: list[str] = []
    parts.append(template["system"].rstrip())
    parts.append("")
    parts.append("=== JSON SCHEMA ===")
    parts.append(json.dumps(response_schema, indent=2))
    parts.append("")
    parts.append("=== EXAMPLES ===")
    for shot in template["few_shots"]:
        parts.append(f"Input: {json.dumps(shot['input'])}")
        parts.append(
            f"Output: {json.dumps(shot['output'], ensure_ascii=False)}"
        )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def load_and_render(
    path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    """Load the prompt template and return (system_prompt, response_schema).

    The response_schema is what the Vertex client passes as
    ``response_schema`` / ``response_json_schema`` to force structured
    output. The system_prompt is the full text (including schema and
    few-shots) the LLM sees as instructions.
    """
    template = load_prompt_template(path or DEFAULT_PROMPT_PATH)
    schema = extracted_intent_json_schema()
    return render_system_prompt(template, schema), schema


def _main_dump() -> int:
    """CLI: render the prompt and print it. Useful for sanity-checking."""
    prompt, schema = load_and_render()
    print(f"=== Rendered system prompt ({len(prompt)} chars) ===\n")
    print(prompt)
    print(f"\n=== Schema has {len(schema.get('$defs', {}))} $defs ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main_dump())
