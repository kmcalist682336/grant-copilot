"""Quick checks for record-level variable alias routing guardrails."""
from __future__ import annotations

from scripts.chatbot.record_variable_aliases import resolve_record_variable_alias


CASES = [
    ("hmda", "hmda", ["applicant income"], "c1aa5d4f3f72"),
    ("hmda", "hmda", ["loan amount"], "c02eb39025e6"),
    ("hmda", "hmda", ["applicant sex"], "6057363dc2e9"),
    ("hmda", "hmda", ["applicant race"], "38ad9c360a98"),
    ("hmda", "hmda", ["applicant ethnicity"], "b4588a673468"),
    ("hmda", "hmda", ["application status"], "906bb78b0f70"),
]


def main() -> int:
    failures = 0
    for dataset, table_id, texts, expected in CASES:
        got = resolve_record_variable_alias(
            dataset=dataset,
            table_id=table_id,
            texts=texts,
        )
        ok = got == expected
        status = "OK" if ok else "FAIL"
        print(f"{status} {dataset}/{table_id}: {texts!r} -> {got!r}")
        if not ok:
            failures += 1
    if failures:
        print(f"FAILED: {failures} variable-alias check(s)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
