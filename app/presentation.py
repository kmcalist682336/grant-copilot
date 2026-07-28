"""Hot-reloading presentation config.

Owns config/presentation.yaml — the four-layer dial described in that
file's header. Re-read from disk on every access so a student's edit
takes effect on the next query (or the next Re-render) with no restart.

A broken edit must never take the app down: a malformed or missing file
falls back to DEFAULTS and reports the parse error, which the UI shows
as a banner instead of a stack trace.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_PATH = REPO_ROOT / "config" / "presentation.yaml"


# Mirrors config/presentation.yaml. Every key the code reads must have a
# default here, so a partial file (student deleted a block) still works.
DEFAULTS: dict[str, Any] = {
    "bundle": {
        # Per-key overrides of the defaults registered in BUNDLE_SPECS.
        # Keys absent here fall back to the spec; keys with no spec at
        # all (an artifact a new dataset contributed) stay off until
        # someone opts in.
        "send": {
            "peer_contexts": False,
            "concept_resolutions": False,
        },
        "max_values_sent": None,
    },
    "formatting": {
        "normalize_whitespace": True,
        "strip_markdown_fences": True,
        "prose_max_words": None,
        "max_key_findings": 6,
        "max_caveats": 5,
    },
    "lints": {
        "forbid_unsourced_numbers": True,
        "require_year_mentioned": True,
        "forbid_peer_names": True,
        "prose_word_range": [60, 320],
    },
    "sections": {
        "prose": True,
        "key_findings": True,
        "caveats": True,
        "citations": True,
        "peer_contexts": True,
        "followups": True,
        "anomaly_flags": True,
        "values_table": True,
        "upstream": True,
        "trace": True,
        "metrics": True,
    },
}


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge one level of nesting — enough for this schema, and it
    keeps an unknown key from silently wiping a whole block."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    return out


class PresentationConfig:
    """Parsed presentation config plus any error encountered loading it."""

    def __init__(self, data: dict[str, Any], error: Optional[str] = None):
        self.data = data
        self.error = error

    @property
    def bundle_options(self) -> dict[str, Any]:
        """Options consumed by ``render_bundle_payload`` — the ``send``
        map plus ``max_values_sent``."""
        return self.data.get("bundle", {})

    @property
    def formatting(self) -> dict[str, Any]:
        return self.data.get("formatting", {})

    @property
    def lints(self) -> dict[str, Any]:
        return self.data.get("lints", {})

    @property
    def sections(self) -> dict[str, Any]:
        return self.data.get("sections", {})

    def shows(self, section: str) -> bool:
        return bool(self.sections.get(section, True))


def load(path: Optional[Path] = None) -> PresentationConfig:
    """Read the presentation config fresh from disk.

    Never raises. On any problem, returns DEFAULTS with ``error`` set so
    the caller can surface the message in the UI.
    """
    p = path or PRESENTATION_PATH
    try:
        with p.open() as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        return PresentationConfig(copy.deepcopy(DEFAULTS))
    except Exception as e:
        logger.warning("presentation.yaml is unparseable (%s); using defaults", e)
        return PresentationConfig(
            copy.deepcopy(DEFAULTS),
            error=f"presentation.yaml could not be parsed: {e}",
        )

    if raw is None:
        return PresentationConfig(copy.deepcopy(DEFAULTS))
    if not isinstance(raw, dict):
        return PresentationConfig(
            copy.deepcopy(DEFAULTS),
            error="presentation.yaml must be a mapping at the top level.",
        )
    return PresentationConfig(_merge(DEFAULTS, raw))


def read_text(path: Optional[Path] = None) -> str:
    """Raw file text, for the editor pane."""
    p = path or PRESENTATION_PATH
    try:
        return p.read_text()
    except FileNotFoundError:
        return ""


def write_text(text: str, path: Optional[Path] = None) -> Optional[str]:
    """Validate then save. Returns an error string, or None on success.

    Validation before write is deliberate — a student who saves invalid
    YAML should get an error in the editor, not a silently broken app.
    """
    try:
        parsed = yaml.safe_load(text)
    except Exception as e:
        return f"Invalid YAML: {e}"
    if parsed is not None and not isinstance(parsed, dict):
        return "Top level must be a mapping (key: value)."
    (path or PRESENTATION_PATH).write_text(text)
    return None
