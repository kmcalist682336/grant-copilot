"""Cross-dataset concept selection, independent of any particular record source."""
from dataclasses import dataclass
import math

from scripts.chatbot.record_dataset import record_dataset_definitions


class DatasetSelectionError(ValueError):
    """A source cannot be selected safely; request clarification, not a fallback."""


@dataclass(frozen=True)
class DatasetSelection:
    dataset: str
    reason: str


def is_census_dataset(dataset):
    """Recognize supported Census API families, including their subproducts."""
    return isinstance(dataset, str) and dataset.split("/", 1)[0] in {
        "census", "acs", "dec", "pep", "popest",
    }


def select_dataset(concept, semantic_router, *, cmap=None, requested_dataset=None, min_score=0.05, ambiguity_ratio=0.90):
    """Resolve curated Census concepts before considering record sources.

    Extractor hints are not evidence and never override a winning source.
    Multiple near-tied datasets require clarification, with no favored source.
    """
    definitions = record_dataset_definitions()
    if requested_dataset is not None and requested_dataset not in {*definitions, "census"}:
        raise DatasetSelectionError(f"Requested dataset {requested_dataset!r} is not configured.")
    texts = [t for t in (concept.canonical_hint, concept.text) if t]
    if cmap is not None:
        for text in texts:
            entry = cmap.lookup(text)
            if entry is not None and is_census_dataset(entry.dataset):
                if requested_dataset not in {None, "census"}:
                    raise DatasetSelectionError(
                        f"{concept.text!r} resolves to Census, but the requested "
                        f"source is {requested_dataset!r}. Please clarify the measure or source."
                    )
                return DatasetSelection("census", "curated Census concept mapping")
    exact = {name for name, definition in definitions.items()
             if definition.recipes().lookup_any(texts) is not None}
    if requested_dataset is not None:
        exact &= {requested_dataset}
    if len(exact) > 1:
        raise DatasetSelectionError(f"Metric matches multiple datasets: {', '.join(sorted(exact))}. Please specify the source.")
    if exact:
        return DatasetSelection(next(iter(exact)), "exact declared metric")
    if semantic_router is None:
        # Without semantic evidence, only a unique exact field mapping can
        # establish a record source. Ordinary Census concepts stay downstream.
        aliases = {name for name, definition in definitions.items()
                   if definition.variable_alias(texts) is not None}
        if requested_dataset is not None:
            aliases &= {requested_dataset}
        if len(aliases) == 1:
            return DatasetSelection(next(iter(aliases)), "exact declared field")
        if aliases or requested_dataset not in {None, "census"} or concept.dataset_hint not in {"unknown", "both", "census"}:
            raise DatasetSelectionError("Cannot establish a unique supported dataset without semantic evidence.")
        return DatasetSelection("census", "existing Census concept path; no record match")
    try:
        routed = semantic_router.route(concept.canonical_hint or concept.text, top_k=8)
    except Exception as exc:
        raise DatasetSelectionError("Dataset matching is unavailable; please retry.") from exc
    scores = {}
    for target in [*routed.top_variables, *routed.top_tables]:
        dataset = getattr(target, "target_dataset", None)
        if not dataset:
            continue
        # ACS vintages/products compete as one Census source, not ambiguity
        # between two Census tables. Unknown sources remain competitors.
        if is_census_dataset(dataset):
            dataset = "census"
        score = float(getattr(target, "aggregate_score", 0) or 0)
        if math.isfinite(score):
            scores[dataset] = max(scores.get(dataset, 0), score)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if requested_dataset is not None:
        if scores.get(requested_dataset, 0) < min_score:
            raise DatasetSelectionError(f"No confident match for this measure in requested dataset {requested_dataset!r}.")
        return DatasetSelection(requested_dataset, "user-named source with semantic evidence")
    if not ranked or ranked[0][1] < min_score:
        raise DatasetSelectionError(f"No confident dataset match for {concept.text!r}. Please clarify the metric or source.")
    winner, score = ranked[0]
    if len(ranked) > 1 and ranked[1][1] >= score * ambiguity_ratio:
        raise DatasetSelectionError(f"Ambiguous dataset match for {concept.text!r}: {winner} and {ranked[1][0]}. Please specify the source or clarify the measure.")
    if winner != "census" and winner not in definitions:
        raise DatasetSelectionError(f"Best matching dataset {winner!r} is not configured; no substitute was selected.")
    return DatasetSelection(winner, f"global semantic match (score={score:.3f})")
