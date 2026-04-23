"""Grant-narrative frame registry.

Loads ``config/frames.yaml`` into typed ``Frame`` objects. The
GrantFrameMatcher classifies user queries into one of these; the
Expander reads each frame's ``required_additional_concepts`` to know
which other concepts to pull; the Synthesizer reads the frame's
``rhetorical_target`` to shape the prose.

Public API:
    Frame               — one entry
    FrameRegistry       — the indexed collection
    load_default_frames — FrameRegistry from config/frames.yaml
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config" / "frames.yaml"
)


class Frame(BaseModel):
    """One curated grant-narrative frame."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        ..., description="Stable ID referenced from code.",
    )
    description: str = Field(
        ..., description="One-line hint shown to the LLM classifier.",
    )
    triggers: list[str] = Field(
        default_factory=list,
        description="Human-readable trigger phrases for the prompt's "
                    "example list.",
    )
    required_additional_concepts: list[str] = Field(
        default_factory=list,
        description="Canonical concept names the Expander adds when "
                    "this frame fires.",
    )
    rhetorical_target: str = Field(
        default="",
        description="Synthesizer hint — the prose shape to target.",
    )
    standard_caveats: list[str] = Field(
        default_factory=list,
        description="Frame-level caveats always surfaced alongside the "
                    "answer.",
    )


class FrameRegistry:
    """Indexed collection of frames."""

    def __init__(self, frames: list[Frame]):
        self.frames = frames
        self._by_name: dict[str, Frame] = {f.name: f for f in frames}
        if "specific_lookup" not in self._by_name:
            raise ValueError(
                "frames.yaml must define a 'specific_lookup' frame "
                "(used as the default classification)"
            )
        if "other" not in self._by_name:
            raise ValueError(
                "frames.yaml must define an 'other' catch-all frame"
            )

    def __len__(self) -> int:
        return len(self.frames)

    def get(self, name: str) -> Frame:
        """Return the frame with ``name``. Falls back to 'other' for
        unknown names so a misbehaving LLM output doesn't crash."""
        return self._by_name.get(name) or self._by_name["other"]

    def names(self) -> list[str]:
        return [f.name for f in self.frames]

    def describe_for_prompt(self) -> str:
        """Render a compact frame inventory for the classifier prompt.

        Format: one section per frame with name + description + triggers.
        Kept short — we want the LLM to choose among names, not re-read
        a novel per query.
        """
        out: list[str] = []
        for f in self.frames:
            lines = [f"- {f.name}:", f"    {f.description.strip()}"]
            if f.triggers:
                trig = ", ".join(f"{t!r}" for t in f.triggers)
                lines.append(f"    triggers: {trig}")
            out.append("\n".join(lines))
        return "\n\n".join(out)


@functools.lru_cache(maxsize=1)
def load_default_frames() -> FrameRegistry:
    """Load and cache the default frames.yaml registry."""
    with _CONFIG_PATH.open() as f:
        doc = yaml.safe_load(f)
    entries = doc.get("frames") or []
    frames = [Frame.model_validate(e) for e in entries]
    return FrameRegistry(frames)


def invalidate_frame_cache() -> None:
    load_default_frames.cache_clear()
