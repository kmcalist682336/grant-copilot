"""Prompt loading, rendering, and versioning.

Prompts live in prompts/v1/ as plain text files with {placeholder} variables.
This module handles:
  - Loading prompt templates from disk
  - Rendering templates with table/variable data
  - Computing content hashes for provenance tracking
  - Validating that all required placeholders are filled
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum number of sample variable labels to include in prompts.
# Long variable lists help the model understand table structure but
# add tokens. 20 is a good balance for most tables.
MAX_SAMPLE_LABELS = 20

# Maximum characters for any single field injected into a prompt.
# Prevents runaway context from very long universe statements etc.
MAX_FIELD_CHARS = 800


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class PromptLoader:
    """Loads and renders prompt templates from the prompts directory."""

    def __init__(self, prompts_dir: Path, version: str = "v1"):
        self.prompts_dir = prompts_dir / version
        self.version = version
        self._cache: dict[str, str] = {}

    def load(self, prompt_name: str) -> str:
        """Load a prompt template by name, with caching."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]

        path = self.prompts_dir / f"{prompt_name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {path}\n"
                f"Expected at: {self.prompts_dir / prompt_name}.txt"
            )
        template = path.read_text(encoding="utf-8")
        self._cache[prompt_name] = template
        logger.debug("Loaded prompt template: %s", prompt_name)
        return template

    def render(self, prompt_name: str, **kwargs) -> str:
        """Render a prompt template with the provided keyword arguments.

        Truncates field values that exceed MAX_FIELD_CHARS to avoid
        runaway context lengths. Logs a warning when truncation occurs.
        """
        template = self.load(prompt_name)

        # Truncate long values
        safe_kwargs = {}
        for key, value in kwargs.items():
            value_str = str(value) if value is not None else ""
            if len(value_str) > MAX_FIELD_CHARS:
                logger.debug(
                    "Truncating field '%s' from %d to %d chars for prompt '%s'",
                    key, len(value_str), MAX_FIELD_CHARS, prompt_name,
                )
                value_str = value_str[:MAX_FIELD_CHARS] + "..."
            safe_kwargs[key] = value_str

        try:
            rendered = template.format(**safe_kwargs)
        except KeyError as e:
            raise ValueError(
                f"Prompt '{prompt_name}' references undefined placeholder {e}. "
                f"Available keys: {sorted(safe_kwargs.keys())}"
            )
        return rendered

    def hash(self, prompt_name: str) -> str:
        """Compute a sha256 hash of the raw prompt template for provenance."""
        template = self.load(prompt_name)
        return hashlib.sha256(template.encode()).hexdigest()[:16]

    @classmethod
    def from_repo_root(cls, version: str = "v1") -> "PromptLoader":
        repo_root = _repo_root()
        prompts_dir = repo_root / "prompts"
        return cls(prompts_dir=prompts_dir, version=version)


# ---------------------------------------------------------------------------
# Table context builders
# ---------------------------------------------------------------------------

def build_sample_labels(variable_inventory: list[dict], n: int = MAX_SAMPLE_LABELS) -> str:
    """Build a formatted sample of variable labels for prompt injection.

    Selects a representative sample of ESTIMATE variables only (not MOE or
    annotations): headline variables first, then a spread across the depth
    range so the model sees the table's structure.
    """
    if not variable_inventory:
        return "(no variables)"

    # Filter to estimates only — MOE and annotation variables add noise
    estimates = [
        v for v in variable_inventory
        if v.get("kind") in ("estimate", "percent_estimate")
    ]
    if not estimates:
        # Fallback: use everything if somehow no estimates are present
        estimates = variable_inventory

    # Prioritize: totals, then shallow variables, then deeper ones
    totals = [v for v in estimates if v.get("is_total")]
    shallow = [v for v in estimates if not v.get("is_total") and v.get("depth", 0) <= 2]
    deep = [v for v in estimates if v.get("depth", 0) > 2]

    selected = []
    selected.extend(totals[:3])
    remaining = n - len(selected)
    shallow_take = min(len(shallow), remaining // 2 + 1)
    selected.extend(shallow[:shallow_take])
    remaining = n - len(selected)
    selected.extend(deep[:remaining])

    # Deduplicate preserving order
    seen = set()
    unique = []
    for v in selected:
        vid = v.get("var_id")
        if vid not in seen:
            seen.add(vid)
            unique.append(v)

    lines = []
    for v in unique[:n]:
        label = v.get("label_raw", "").replace("!!", " → ").strip(" →")
        var_id = v.get("var_id", "")
        lines.append(f"  {var_id}: {label}")

    total_estimate_count = len(estimates)
    if total_estimate_count > n:
        lines.append(f"  ... ({total_estimate_count - n} more variables)")

    return "\n".join(lines)


def build_flat_sample_labels(variables: dict[str, dict], n: int = MAX_SAMPLE_LABELS) -> str:
    """Build sample labels for flat/microdata datasets."""
    if not variables:
        return "(no variables)"
    lines = []
    for var_id, meta in list(variables.items())[:n]:
        label = meta.get("label", "")
        lines.append(f"  {var_id}: {label}")
    if len(variables) > n:
        lines.append(f"  ... ({len(variables) - n} more variables)")
    return "\n".join(lines)
