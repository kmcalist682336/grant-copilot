"""Stage 5: Topic Tagging.

Two-pass design:

  PASS 1 — Discovery mode (--mode discovery):
    Runs across all tables with the current vocabulary as a preference, not a
    constraint. The model uses existing tags where they fit, and proposes new
    tags when nothing in the vocabulary captures the topic well. All proposals
    are written to data/state/tag_proposals.jsonl for human review.

    After the discovery run, review proposals with:
        python -m scripts.diagnostics.review_tag_proposals

    Promote accepted proposals into config/topic_vocabulary.yaml, then run
    Pass 2.

  PASS 2 — Production mode (--mode production):
    Runs with the locked, finalized vocabulary. No proposals are accepted.
    Unknown tags from the model are logged as warnings and discarded, but in
    practice this should not occur since the vocabulary was built from Pass 1
    outputs.

Output files per table:
  stage5_tags.json — assigned tags with confidence and rationale

Proposal log (discovery mode only):
  data/state/tag_proposals.jsonl — one JSON line per proposal, with frequency
  information accumulated across the full run.
"""
from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml

from scripts.enrichment.llm_client import VLLMClient, LLMCallError
from scripts.enrichment.prompts import PromptLoader, build_sample_labels
from scripts.utils.paths import PipelinePaths
from scripts.utils.state import StateTracker, Status

logger = logging.getLogger(__name__)

STAGE_NAME = "stage5_tags"
DEPENDS_ON_S2 = "stage2_structure"
DEPENDS_ON_S4 = "stage4_enriched"

TaggingMode = Literal["discovery", "production"]


# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------

class TopicVocabulary:
    """Loads and validates the topic tag vocabulary."""

    def __init__(self, vocab_path: Path):
        with vocab_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.l1_domains: set[str] = set(raw["l1_domains"])

        # Flatten l2_subjects dict-of-lists into a flat set
        self.l2_subjects: set[str] = set()
        for entries in raw["l2_subjects"].values():
            self.l2_subjects.update(entries)

        # Formatted strings for prompt injection
        self._valid_l1_str = "\n".join(
            f"  {d}" for d in sorted(self.l1_domains)
        )
        self._valid_tags_str = "\n".join(
            f"  {tag}" for tag in sorted(self.l2_subjects)
        )

    def is_valid_tag(self, tag: str) -> bool:
        return tag in self.l2_subjects

    def valid_l1_for_prompt(self) -> str:
        return self._valid_l1_str

    def valid_tags_for_prompt(self) -> str:
        return self._valid_tags_str

    def filter_valid(self, tags: list[dict]) -> tuple[list[dict], list[str]]:
        """Split tags into (valid_list, invalid_strings)."""
        valid, invalid = [], []
        for t in tags:
            if self.is_valid_tag(t.get("tag", "")):
                valid.append(t)
            else:
                invalid.append(t.get("tag", ""))
        return valid, invalid

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> "TopicVocabulary":
        return cls(config_dir / "topic_vocabulary.yaml")


# ---------------------------------------------------------------------------
# Proposal accumulator
# ---------------------------------------------------------------------------

class ProposalAccumulator:
    """Collects tag proposals across the discovery run and writes them to disk.

    Tracks how many tables proposed each tag so the review script can sort
    by frequency — high-frequency proposals are strong candidates for the
    vocabulary, low-frequency ones may be hallucinations.
    """

    def __init__(self, proposals_path: Path):
        self.proposals_path = proposals_path
        # tag -> {rationale, proposing_tables, count}
        self._by_tag: dict[str, dict] = defaultdict(lambda: {
            "tag": "",
            "proposed_l1": "",
            "proposed_l2": "",
            "rationales": [],
            "proposing_tables": [],
            "count": 0,
        })

    def add(
        self,
        proposals: list[dict],
        year: str,
        dataset_path: str,
        table_id: str,
        table_title: str,
    ) -> None:
        """Record proposals from one table."""
        for p in proposals:
            tag = p.get("tag", "").strip()
            if not tag:
                continue
            entry = self._by_tag[tag]
            entry["tag"] = tag
            entry["proposed_l1"] = p.get("proposed_l1", "")
            entry["proposed_l2"] = p.get("proposed_l2", "")
            entry["count"] += 1
            entry["proposing_tables"].append(
                f"{year}/{dataset_path}/{table_id}"
            )
            # Keep up to 5 unique rationales for review
            rationale = p.get("rationale", "")
            if rationale and rationale not in entry["rationales"]:
                if len(entry["rationales"]) < 5:
                    entry["rationales"].append(rationale)

    def flush(self) -> None:
        """Write accumulated proposals to disk as JSONL."""
        if not self._by_tag:
            return
        self.proposals_path.parent.mkdir(parents=True, exist_ok=True)
        with self.proposals_path.open("w", encoding="utf-8") as f:
            for entry in sorted(
                self._by_tag.values(), key=lambda x: -x["count"]
            ):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(
            "Wrote %d unique tag proposals to %s",
            len(self._by_tag), self.proposals_path,
        )

    @property
    def total_proposals(self) -> int:
        return len(self._by_tag)


# ---------------------------------------------------------------------------
# Flat table deterministic tagging
# ---------------------------------------------------------------------------

_FLAT_DATASET_TAGS: dict[str, list[dict]] = {
    "pdb": [
        {"tag": "Data Quality/Coverage and Universe",
         "confidence": "high",
         "rationale": "Planning database combines Census operational metrics"},
        {"tag": "Demographics/Population Totals",
         "confidence": "medium",
         "rationale": "Contains total population and housing unit counts"},
    ],
    "geoinfo": [
        {"tag": "Geography/Geographic Identifiers",
         "confidence": "high",
         "rationale": "Geographic boundary and identifier reference data"},
    ],
}

_DEFAULT_FLAT_TAGS: list[dict] = [
    {"tag": "Data Quality/Coverage and Universe",
     "confidence": "medium",
     "rationale": "Flat reference dataset with no standard table grouping"},
]


def _deterministic_flat_tags(dataset_path: str) -> list[dict]:
    for key, tags in _FLAT_DATASET_TAGS.items():
        if key in dataset_path:
            return tags
    return _DEFAULT_FLAT_TAGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_context(
    year: str, dataset_path: str, table_id: str, paths: PipelinePaths
) -> Optional[dict]:
    s2 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S2))
    s4 = _load_json(paths.processed_table_stage_file(
        year, dataset_path, table_id, DEPENDS_ON_S4))
    if not s2 or not s4:
        return None
    return {"s2": s2, "s4": s4}


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _tagging_settings(inference_cfg: dict) -> dict:
    return inference_cfg.get("stages", {}).get(
        "stage5_tagging", {
            "temperature": 0.1, "max_tokens": 768,
            "thinking_mode": True, "top_p": 0.9,
        }
    )


# ---------------------------------------------------------------------------
# Core tagging — discovery mode
# ---------------------------------------------------------------------------

async def _tag_discovery(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    vocab: TopicVocabulary,
    inference_cfg: dict,
    accumulator: ProposalAccumulator,
    year: str,
    dataset_path: str,
    table_id: str,
) -> tuple[list[dict], dict]:
    """Discovery mode: use vocab as preference, collect proposals freely."""
    s2, s4 = ctx["s2"], ctx["s4"]
    inventory = s2.get("variable_inventory", [])

    prompt = loader.render(
        "topic_tagging_discovery",
        title=s4.get("title") or "",
        description=s4.get("description") or "",
        universe_plain=s4.get("universe_plain") or "",
        subject_area=s2.get("subject_area") or "",
        measure_type=s2.get("measure_type") or "",
        sample_labels=build_sample_labels(inventory),
        valid_l1=vocab.valid_l1_for_prompt(),
        valid_tags=vocab.valid_tags_for_prompt(),
    )

    resp = await client.generate(
        prompt, "topic_tagging_discovery", _tagging_settings(inference_cfg)
    )

    raw_tags = resp.content.get("tags", [])
    proposals = resp.content.get("proposed_tags", [])

    # Keep only valid tags from the vocabulary portion
    valid_tags, invalid = vocab.filter_valid(raw_tags)

    if invalid:
        logger.debug(
            "%s/%s/%s: model tried to use non-vocab tags as vocabulary tags: %s "
            "(these should have gone in proposed_tags instead)",
            year, dataset_path, table_id, invalid,
        )

    # Record proposals to accumulator
    if proposals:
        accumulator.add(
            proposals, year, dataset_path, table_id,
            table_title=s4.get("title") or "",
        )
        logger.debug(
            "%s/%s/%s: %d new tag proposal(s): %s",
            year, dataset_path, table_id, len(proposals),
            [p.get("tag") for p in proposals],
        )

    # Fallback if model returned no valid vocabulary tags at all
    if not valid_tags:
        subject = s2.get("subject_area", "")
        fallback = vocab.is_valid_tag(subject)
        if fallback:
            valid_tags = [{"tag": subject, "confidence": "low",
                           "rationale": "Fallback from structural subject_area field"}]
        else:
            valid_tags = _DEFAULT_FLAT_TAGS

    prov = resp.provenance_block()
    prov["mode"] = "discovery"
    prov["proposals_count"] = len(proposals)
    return valid_tags, prov


# ---------------------------------------------------------------------------
# Core tagging — production mode
# ---------------------------------------------------------------------------

async def _tag_production(
    ctx: dict,
    client: VLLMClient,
    loader: PromptLoader,
    vocab: TopicVocabulary,
    inference_cfg: dict,
    year: str,
    dataset_path: str,
    table_id: str,
) -> tuple[list[dict], dict]:
    """Production mode: locked vocabulary, log warnings for unknown tags."""
    s2, s4 = ctx["s2"], ctx["s4"]
    inventory = s2.get("variable_inventory", [])

    prompt = loader.render(
        "topic_tagging_production",
        title=s4.get("title") or "",
        description=s4.get("description") or "",
        universe_plain=s4.get("universe_plain") or "",
        subject_area=s2.get("subject_area") or "",
        measure_type=s2.get("measure_type") or "",
        sample_labels=build_sample_labels(inventory),
        valid_tags=vocab.valid_tags_for_prompt(),
    )

    resp = await client.generate(
        prompt, "topic_tagging_production", _tagging_settings(inference_cfg)
    )

    raw_tags = resp.content.get("tags", [])
    valid_tags, invalid = vocab.filter_valid(raw_tags)

    if invalid:
        logger.warning(
            "%s/%s/%s: model returned tags not in locked vocabulary: %s — discarding. "
            "Consider re-running discovery mode if this happens frequently.",
            year, dataset_path, table_id, invalid,
        )

    if not valid_tags:
        subject = s2.get("subject_area", "")
        if vocab.is_valid_tag(subject):
            valid_tags = [{"tag": subject, "confidence": "low",
                           "rationale": "Fallback from structural subject_area field"}]
        else:
            valid_tags = _DEFAULT_FLAT_TAGS

    prov = resp.provenance_block()
    prov["mode"] = "production"
    return valid_tags, prov


# ---------------------------------------------------------------------------
# Per-table orchestration
# ---------------------------------------------------------------------------

async def tag_one_table(
    year: str,
    dataset_path: str,
    table_id: str,
    client: VLLMClient,
    loader: PromptLoader,
    vocab: TopicVocabulary,
    inference_cfg: dict,
    paths: PipelinePaths,
    state: StateTracker,
    mode: TaggingMode = "discovery",
    accumulator: Optional[ProposalAccumulator] = None,
) -> dict:
    """Run Stage 5 tagging for a single table in the specified mode."""
    ctx = _load_context(year, dataset_path, table_id, paths)
    if not ctx:
        msg = f"Missing stage2/stage4 for {year}/{dataset_path}/{table_id}"
        logger.warning(msg)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_NO_DATA, error_message=msg,
        )
        return {"status": "failed", "reason": "missing_prerequisites"}

    analysis_type = ctx["s2"].get("analysis_type", "grouped")
    collected_at = datetime.now(timezone.utc).isoformat()

    try:
        if analysis_type == "flat":
            tags = _deterministic_flat_tags(dataset_path)
            provenance = {
                "method": "deterministic",
                "mode": mode,
                "dataset_path": dataset_path,
            }
        elif mode == "discovery":
            if accumulator is None:
                raise ValueError("accumulator must be provided in discovery mode")
            tags, provenance = await _tag_discovery(
                ctx, client, loader, vocab, inference_cfg, accumulator,
                year, dataset_path, table_id,
            )
        else:  # production
            tags, provenance = await _tag_production(
                ctx, client, loader, vocab, inference_cfg,
                year, dataset_path, table_id,
            )

        output = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "tagging_mode": mode,
            "collected_at": collected_at,
            "year": year,
            "dataset_path": dataset_path,
            "table_id": table_id,
            "tags": tags,
            "tag_count": len(tags),
            "l1_domains": sorted({t["tag"].split("/")[0] for t in tags}),
            "_provenance": provenance,
        }

        out_path = paths.processed_table_stage_file(
            year, dataset_path, table_id, STAGE_NAME
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(out_path, output)

        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME, status=Status.SUCCESS,
            metadata={"tag_count": len(tags), "mode": mode},
        )
        return {"status": "success", "table_id": table_id,
                "tags": tags, "mode": mode}

    except LLMCallError as e:
        logger.error("LLM call failed for %s/%s/%s: %s",
                     year, dataset_path, table_id, e)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_OTHER, error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type="LLMCallError", error_message=str(e),
        )
        return {"status": "failed", "reason": str(e)}

    except Exception as e:
        logger.exception("Unexpected error tagging %s/%s/%s",
                         year, dataset_path, table_id)
        state.set_status(
            year=year, dataset_path=dataset_path, entity_type="table",
            entity_id=table_id, stage=STAGE_NAME,
            status=Status.FAILED_OTHER, error_message=str(e),
        )
        state.log_error(
            year=year, dataset_path=dataset_path, entity_id=table_id,
            stage=STAGE_NAME, error_type=type(e).__name__, error_message=str(e),
            traceback=traceback.format_exc(),
        )
        return {"status": "failed", "reason": str(e)}
