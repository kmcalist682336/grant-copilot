"""Demo: natural-language question -> HMDA variable Parquet manifest."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from scripts.chatbot.auth_check import check_auth
from scripts.chatbot.semantic_router import SemanticRouter, VertexEmbedder
from scripts.ingestion.universal_ingestion.demo_hmda_query import (
    PROJECT_ROOT,
    load_registry,
    resolve_files,
)


class QueryComponent(BaseModel):
    """One independently searchable concept in the user's question."""

    search_text: str = Field(
        description="Short generic phrase describing one required variable."
    )
    role: Literal["measure", "filter", "geography_filter"]
    operation: Literal["average", "count", "sum", "ratio"] | None = None
    value_hint: str | None = Field(
        default=None,
        description="Requested value such as female, originated, or Fulton County.",
    )


class QueryAnalysis(BaseModel):
    """Dataset choice and variable-level decomposition."""

    datasets: list[Literal["hmda", "census"]]
    year: int | None = None
    state: str | None = Field(
        default=None, description="Two-letter state abbreviation when known."
    )
    geography_text: str | None = None
    components: list[QueryComponent]


SYSTEM_PROMPT = """
Classify and decompose a data question for a retrieval demo.

Available datasets:
- hmda: individual mortgage applications and loans, applicant attributes,
  property geography, loan terms, lender actions, approvals, and denials.
- census: population, households, demographics, housing, and community
  statistics that are not individual mortgage records.

Choose hmda, census, or both. Spell HMDA exactly as "hmda". For HMDA, return
one component per column needed to answer the question. Components retrieve
files only; calculations happen elsewhere. Use generic searchable phrases,
for example "applicant income", "applicant sex", "action taken on mortgage
application", and "county containing mortgaged property". Put requested
values such as female, originated, or Fulton County in value_hint. Never
invent variable IDs or file paths.
""".strip()


def newest_artifact_dir() -> Path:
    """Find the newest complete artifact pair that contains HMDA cards."""
    candidates = [PROJECT_ROOT / "data" / "metadata"]
    backup_root = PROJECT_ROOT / "data" / "backups"
    if backup_root.is_dir():
        candidates.extend(path for path in backup_root.iterdir() if path.is_dir())
    complete = [
        path for path in candidates
        if (path / "search_index.db").is_file()
        and (path / "embeddings.faiss").is_file()
    ]
    if not complete:
        raise FileNotFoundError(
            "Could not find search_index.db and embeddings.faiss together"
        )
    hmda_complete = []
    for path in complete:
        with sqlite3.connect(path / "search_index.db") as database:
            has_hmda = database.execute(
                "SELECT EXISTS(SELECT 1 FROM cards "
                "WHERE target_dataset = 'hmda')"
            ).fetchone()[0]
        if has_hmda:
            hmda_complete.append(path)
    if not hmda_complete:
        raise FileNotFoundError(
            "Retrieval artifacts were found, but none contain HMDA cards"
        )
    return max(
        hmda_complete,
        key=lambda path: (path / "embeddings.faiss").stat().st_mtime,
    )


def analyze_query(query: str, model: str) -> QueryAnalysis:
    """Use OpenRouter structured output for dataset choice/decomposition."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put it in the project's "
            "uncommitted .env file or export it in the shell."
        )
    completion = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    ).chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format=QueryAnalysis,
    )
    analysis = completion.choices[0].message.parsed
    if analysis is None:
        raise RuntimeError("The model did not return a structured analysis")
    return analysis


def retrieve_hmda_variables(
    analysis: QueryAnalysis,
    router: SemanticRouter,
) -> list[dict]:
    """Choose the best dataset-scoped HMDA variable for each component."""
    selected: dict[str, dict] = {}
    for component in analysis.components:
        result = router.route_dataset(
            component.search_text,
            target_dataset="hmda",
            top_k=5,
        )
        if not result.top_variables:
            raise LookupError(
                f"No HMDA variable matched component: {component.search_text}"
            )
        target = result.top_variables[0]
        variable_id = target.target_variable_id
        evidence = {
            "variable_id": variable_id,
            "role": component.role,
            "operation": component.operation,
            "value_hint": component.value_hint,
            "search_text": component.search_text,
            "matched_card": target.best_hit.text,
            "score": round(target.aggregate_score, 4),
        }
        selected.setdefault(variable_id, evidence)
    return list(selected.values())


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Resolve a natural-language question to HMDA Parquet files."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--state", default="GA")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", "openai/gpt-5.4-mini"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        help="Directory containing search_index.db and embeddings.faiss.",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or newest_artifact_dir()
    analysis = analyze_query(args.query, args.model)
    output: dict = {
        "query": args.query,
        "selected_datasets": analysis.datasets,
        "analysis": analysis.model_dump(),
        "retrieval_artifacts": artifact_dir.as_posix(),
    }

    if "hmda" not in analysis.datasets:
        output["message"] = (
            "The model did not select HMDA; this immediate demo returns only "
            "HMDA variable files."
        )
        print(json.dumps(output, indent=2))
        return
    if analysis.year is None:
        raise ValueError("The query must specify a year")

    year = analysis.year or args.year
    state = (analysis.state or args.state).upper()
    if state != "GA":
        raise ValueError(
            "The local HMDA demo currently contains only state=GA"
        )

    auth = check_auth(verbose=False)
    router = SemanticRouter(
        index_path=artifact_dir / "embeddings.faiss",
        metadata_db_path=artifact_dir / "search_index.db",
        embedder=VertexEmbedder(
            project=auth["project_id"],
            location=auth.get("location", "us-central1"),
        ),
    )
    matches = retrieve_hmda_variables(analysis, router)
    manifest = resolve_files(
        {
            "dataset": "hmda",
            "year": year,
            "variable_ids": [match["variable_id"] for match in matches],
        },
        load_registry(),
    )
    evidence_by_id = {match["variable_id"]: match for match in matches}
    for item in manifest["files"]:
        item["retrieval"] = evidence_by_id[item["variable_id"]]
    output["manifest"] = manifest
    if "census" in analysis.datasets:
        output["note"] = (
            "Census was also selected; Census retrieval is intentionally "
            "outside this HMDA-only demo manifest."
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
