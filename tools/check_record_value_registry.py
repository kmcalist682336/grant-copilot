"""Quick checks for record-level categorical value normalization.

This script intentionally lives outside ``tests/`` because it is a lightweight
developer smoke check for the HMDA demo path.
"""
from __future__ import annotations

from scripts.chatbot.record_value_registry import resolve_record_filter_value


CASES = [
    ("hmda", "hmda", "b4588a673468", "Hispanic", "Hispanic or Latino"),
    ("hmda", "hmda", "b4588a673468", "non-hispanic", "Not Hispanic or Latino"),
    ("hmda", "hmda", "38ad9c360a98", "black", "Black or African American"),
    ("hmda", "hmda", "6057363dc2e9", "women", "Female"),
    ("hmda", "hmda", "906bb78b0f70", "denied", "Application denied"),
    (
        "hmda",
        "hmda",
        "906bb78b0f70",
        "approved",
        ["Loan originated", "Application approved but not accepted"],
    ),
]


def main() -> int:
    failures = 0
    for dataset, table_id, variable_id, raw, expected in CASES:
        got = resolve_record_filter_value(
            dataset=dataset,
            table_id=table_id,
            variable_id=variable_id,
            raw_value=raw,
        )
        ok = got == expected
        status = "OK" if ok else "FAIL"
        print(f"{status} {variable_id}: {raw!r} -> {got!r}")
        if not ok:
            failures += 1
    if failures:
        print(f"FAILED: {failures} value-registry check(s)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
