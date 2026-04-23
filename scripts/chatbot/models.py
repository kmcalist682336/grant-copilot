"""Pydantic contracts for the chatbot intent + resolution pipeline.

These schemas are the fixed contracts between stages:
  - extractor.py  produces ExtractedIntent (geographies + concepts +
                  temporal intent)
  - resolver.py   produces ResolvedGeography per geo operand
  - planner       (Phase H) takes concepts → Tier 1/2/3 → APIPlanCall list
  - orchestrator  wraps everything into QueryResolution

The JSON schema derived from ExtractedIntent is passed to Vertex AI as
the structured-output schema, so any field changes here propagate
directly to the LLM contract.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


RefType = Literal[
    "administrative",
    "neighborhood",
    "corridor",
    "region",
    "relative",
    "landmark_area",
    "national",
]

Qualifier = Literal[
    "along",
    "near",
    "in",
    "around",
    "south_of",
    "north_of",
    "east_of",
    "west_of",
    "within",
]

ProximityTier = Literal["very_local", "local", "regional"]

IntentType = Literal["exact", "proximity", "comparison", "context"]

TemporalIntent = Literal["latest", "change", "trend"]

Direction = Literal["n", "s", "e", "w", "ne", "nw", "se", "sw"]

GeoLevel = Literal[
    "tract",
    "county",
    "state",
    "place",
    "zcta",
    "msa",
    "school_district_unified",
    "neighborhood",
    "corridor",
    "region",
    "us",
]

DataLevelAvailable = Literal[
    "tract", "county", "state", "place", "zcta", "msa",
    "school_district_unified", "us", "none",
]

ClarificationKind = Literal["choose", "open", "data_gap"]


class ExtractedGeoRef(BaseModel):
    """A single geographic reference extracted from a user query."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(
        ..., description="Verbatim span from the user query."
    )
    ref_type: RefType = Field(
        ..., description="Category of the reference."
    )
    qualifier: Optional[Qualifier] = Field(
        default=None,
        description="Spatial qualifier (near, along, in, ...) if present.",
    )
    proximity_tier: Optional[ProximityTier] = Field(
        default=None,
        description=(
            "Tier of proximity when qualifier implies one. Defaults are "
            "applied in the resolver if the LLM leaves this blank."
        ),
    )
    scope_hint: Optional[str] = Field(
        default=None,
        description=(
            "Disambiguation scope from the query (e.g., 'Georgia', "
            "'Atlanta metro'). Not a filter — a rank boost."
        ),
    )


class ExtractedConcept(BaseModel):
    """A single data-dimension concept the user is asking about.

    The Tier 1 planner first tries ``canonical_hint`` (or ``text``) against
    the curated concept_map; on miss it falls through to Tier 2 FTS. The
    ``is_composite`` flag is a hint for the Phase H planner — when True
    the planner consults the Tier 0.5 decomposition cache before lookup.
    """

    model_config = ConfigDict(extra="ignore")

    text: str = Field(
        ..., description="Verbatim span from the user query.",
    )
    canonical_hint: Optional[str] = Field(
        default=None,
        description=(
            "LLM's best guess at a normalized term that should hit the "
            "Tier 1 concept_map (e.g., text='paycheck' → "
            "canonical_hint='median household income'). May be None when "
            "the LLM has no confident normalization."
        ),
    )
    is_composite: bool = Field(
        default=False,
        description=(
            "Hint that this concept is a composite that needs "
            "decomposition (e.g., 'gentrification', 'fresh food access'). "
            "The planner consults the decomposition cache first when set."
        ),
    )


class CompoundSpec(BaseModel):
    """How to combine two (or one) extracted refs into a single resolution.

    A query can carry multiple compounds (one per comparison operand);
    each compound consumes its primary index (and filter index when
    op != 'directional'). Indices not consumed by any compound are
    resolved standalone — see ``ExtractedIntent`` and ``resolve_intent``.
    """

    model_config = ConfigDict(extra="ignore")

    primary_ref_idx: int = Field(
        ...,
        ge=0,
        description="Index into ExtractedIntent.geo_refs for the main geography.",
    )
    filter_ref_idx: int = Field(
        ...,
        ge=0,
        description=(
            "Index for the filtering geography. Equal to primary_ref_idx "
            "when op='directional' (self-filter)."
        ),
    )
    op: Literal["intersect", "directional", "difference"] = Field(
        ...,
        description=(
            "Combination operation. 'intersect' = tracts in both. "
            "'directional' = half-plane filter on primary. "
            "'difference' = primary tracts minus filter tracts."
        ),
    )
    direction: Optional[Direction] = Field(
        default=None,
        description="Compass direction; required when op='directional'.",
    )

    @model_validator(mode="after")
    def _validate_op_fields(self) -> "CompoundSpec":
        if self.op == "directional":
            if self.direction is None:
                raise ValueError("op='directional' requires a direction.")
            if self.primary_ref_idx != self.filter_ref_idx:
                raise ValueError(
                    "op='directional' requires primary_ref_idx == filter_ref_idx."
                )
        else:
            if self.direction is not None:
                raise ValueError(
                    f"op={self.op!r} must not carry a direction."
                )
            if self.primary_ref_idx == self.filter_ref_idx:
                raise ValueError(
                    f"op={self.op!r} requires primary_ref_idx != filter_ref_idx."
                )
        return self


class ExtractedIntent(BaseModel):
    """Structured output of STEP 1 (LLM extraction)."""

    model_config = ConfigDict(extra="ignore")

    geo_refs: list[ExtractedGeoRef] = Field(
        default_factory=list,
        description="Geographic references found in the query.",
    )
    compounds: list[CompoundSpec] = Field(
        default_factory=list,
        description=(
            "Zero or more compound operations over geo_refs. Each compound "
            "consumes its primary (and filter, when op != 'directional') "
            "index and yields a single combined ResolvedGeography. Indices "
            "not consumed by any compound are resolved standalone. Use "
            "multiple entries for queries like 'A vs SW B vs C minus D'."
        ),
    )
    concepts: list[ExtractedConcept] = Field(
        default_factory=list,
        description=(
            "Data-dimension concepts the user is asking about (e.g., "
            "median income, poverty rate, gentrification). Empty for "
            "pure-context queries like 'tell me about Buckhead'."
        ),
    )
    temporal_intent: TemporalIntent = Field(
        default="latest",
        description=(
            "How the planner should treat time. 'latest' = single most "
            "recent vintage. 'change' = compare two specific years "
            "(supplied via `years`). 'trend' = sequence over a span."
        ),
    )
    years: list[int] = Field(
        default_factory=list,
        description=(
            "Optional explicit years pulled from the query "
            "(e.g., 'change from 2020 to 2023' → [2020, 2023]). Empty "
            "means the planner picks defaults based on temporal_intent."
        ),
    )
    intent_type: IntentType = Field(
        default="exact",
        description="Overall intent category of the query.",
    )
    comparison_implied: bool = Field(
        default=False,
        description="User is asking for a comparison between geographies.",
    )
    national_comparison_implied: bool = Field(
        default=False,
        description="User wants a comparison to the national total.",
    )

    @model_validator(mode="after")
    def _validate_compound_indices(self) -> "ExtractedIntent":
        n = len(self.geo_refs)
        for i, c in enumerate(self.compounds):
            if c.primary_ref_idx >= n:
                raise ValueError(
                    f"compounds[{i}].primary_ref_idx={c.primary_ref_idx} "
                    f"out of range; only {n} geo_refs."
                )
            if c.filter_ref_idx >= n:
                raise ValueError(
                    f"compounds[{i}].filter_ref_idx={c.filter_ref_idx} "
                    f"out of range; only {n} geo_refs."
                )
        # A primary index can only be claimed by one compound (it'd otherwise
        # collapse two distinct operations onto the same operand slot).
        primaries = [c.primary_ref_idx for c in self.compounds]
        if len(primaries) != len(set(primaries)):
            raise ValueError(
                "Each geo_ref index may be the primary of at most one compound."
            )
        return self


class ResolvedGeography(BaseModel):
    """Output of STEP 2 (gazetteer resolution) for a single reference or
    compound result."""

    model_config = ConfigDict(extra="ignore")

    geo_id: str = Field(..., description="Primary GEOID or synthetic ID.")
    geo_level: GeoLevel
    geo_type: str = Field(
        ..., description="Gazetteer-specific type (e.g., 'atl_neighborhood')."
    )
    display_name: str

    tract_geoids: list[str] = Field(
        default_factory=list,
        description=(
            "Tracts to aggregate when the resolution spans multiple tracts "
            "(neighborhoods, corridors, landmarks). Empty for pure admin."
        ),
    )
    county_geoid: Optional[str] = Field(
        default=None,
        description="Containing county GEOID when applicable.",
    )

    api_for_clause: str = Field(
        ..., description="Census API 'for=' clause, ready to use."
    )
    api_in_clause: str = Field(
        default="",
        description="Census API 'in=' clause (may be empty).",
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Resolver confidence score."
    )
    assumption_notes: list[str] = Field(default_factory=list)

    data_level_available: DataLevelAvailable = Field(
        ...,
        description=(
            "Finest Census geography level we have Stage 8 data for. "
            "'tract' only for GA; non-GA tops out at 'state' or 'msa'."
        ),
    )

    source_ref: ExtractedGeoRef = Field(
        ..., description="The ExtractedGeoRef this resolution came from."
    )


class ClarificationQuestion(BaseModel):
    """STEP 3 output when confidence is too low to proceed silently."""

    model_config = ConfigDict(extra="ignore")

    kind: ClarificationKind
    text: str = Field(..., description="Question to show the user.")
    options: list[str] = Field(
        default_factory=list,
        description="Populated when kind='choose'.",
    )
    pending_ref_text: str = Field(
        ...,
        description=(
            "The original extracted span this question is about; used to "
            "match the user's next turn back to the pending reference."
        ),
    )


class QueryResolution(BaseModel):
    """Top-level result returned by the orchestrator per user turn."""

    model_config = ConfigDict(extra="ignore")

    session_id: str
    intent: ExtractedIntent
    resolved: list[ResolvedGeography] = Field(default_factory=list)
    clarification_question: Optional[ClarificationQuestion] = None


def extracted_intent_json_schema() -> dict[str, Any]:
    """Return the JSON schema for ExtractedIntent.

    Passed to Vertex AI as ``response_schema`` when calling the extractor.
    Pydantic's default output includes ``$defs`` references; Vertex supports
    those in recent ``google-genai`` versions.
    """
    return ExtractedIntent.model_json_schema()
