from pathlib import Path

from scripts.chatbot.auth_check import check_auth
from scripts.chatbot.semantic_router import SemanticRouter, VertexEmbedder


BACKUP_DIRECTORY = Path(
    "data/backups/search-index-20260717-172046"
)

DATABASE_PATH = BACKUP_DIRECTORY / "search_index.db"
FAISS_PATH = BACKUP_DIRECTORY / "embeddings.faiss"

EXPECTED_VARIABLES = {
    "c1aa5d4f3f72": "income",
    "6422e2d2aab7": "county_code",
    "6057363dc2e9": "applicant_sex",
    "906bb78b0f70": "action_taken",
}


def create_router() -> SemanticRouter:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Backup SQLite database not found: {DATABASE_PATH}"
        )

    if not FAISS_PATH.is_file():
        raise FileNotFoundError(
            f"Backup FAISS index not found: {FAISS_PATH}"
        )

    auth = check_auth(verbose=False)

    embedder = VertexEmbedder(
        project=auth["project_id"],
        location=auth.get("location", "us-central1"),
    )

    print("Loading the backup FAISS index. This may take a while...")

    return SemanticRouter(
        index_path=FAISS_PATH,
        metadata_db_path=DATABASE_PATH,
        embedder=embedder,
    )


def print_hmda_results(
    router: SemanticRouter,
    query: str,
    top_k: int = 50,
) -> set[str]:
    print(f"\nQUERY: {query}")

    result = router.route(
        query,
        top_k=top_k,
    )

    hmda_results = [
        target
        for target in result.top_variables
        if target.target_dataset == "hmda"
    ]

    if not hmda_results:
        print("No HMDA variables found.")
        return set()

    selected_ids = set()

    for rank, target in enumerate(hmda_results, start=1):
        variable_id = target.target_variable_id

        if variable_id is None:
            continue

        selected_ids.add(variable_id)
        source_name = EXPECTED_VARIABLES.get(
            variable_id,
            "unknown",
        )

        print(
            f"{rank}. "
            f"{source_name} "
            f"({variable_id}) "
            f"score={target.aggregate_score:.4f}"
        )
        print(f"   Best card: {target.best_hit.text}")

    return selected_ids


def main() -> None:
    router = create_router()

    try:
        combined_query = (
            "What is the average income of female mortgage applicants "
            "with originated loans in Fulton County in 2024?"
        )

        print("\n" + "=" * 70)
        print("COMBINED-QUESTION TEST")
        print("=" * 70)

        combined_ids = print_hmda_results(
            router,
            combined_query,
            top_k=50,
        )

        component_queries = [
            "average income of mortgage applicants",
            "female mortgage applicants",
            "originated mortgage loans",
            "mortgage applications in Fulton County",
        ]

        print("\n" + "=" * 70)
        print("DECOMPOSED-QUESTION TEST")
        print("=" * 70)

        decomposed_ids = set()

        for query in component_queries:
            decomposed_ids.update(
                print_hmda_results(
                    router,
                    query,
                    top_k=25,
                )
            )

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print("\nCombined-query variables:")

        if combined_ids:
            for variable_id in sorted(combined_ids):
                print(
                    f"- {EXPECTED_VARIABLES.get(variable_id, 'unknown')} "
                    f"({variable_id})"
                )
        else:
            print("- None")

        print("\nDecomposed-query variables:")

        for variable_id in sorted(decomposed_ids):
            print(
                f"- {EXPECTED_VARIABLES.get(variable_id, 'unknown')} "
                f"({variable_id})"
            )

        expected_ids = set(EXPECTED_VARIABLES)
        missing_ids = expected_ids - decomposed_ids

        print("\nExpected-variable check:")

        for variable_id, source_name in EXPECTED_VARIABLES.items():
            status = (
                "FOUND"
                if variable_id in decomposed_ids
                else "MISSING"
            )

            print(
                f"- {status}: {source_name} "
                f"({variable_id})"
            )

        if missing_ids:
            print(
                "\nThe decomposed search did not find every expected "
                "HMDA variable."
            )
        else:
            print(
                "\nAll expected HMDA variables were retrieved."
            )

    finally:
        router.metadata_db.close()


if __name__ == "__main__":
    main()