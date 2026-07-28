"""Tests for the app presentation layer.

Covers the four dials described in config/presentation.yaml:
  layer 1 — payload gating (what the LLM is shown)
  layer 2 — prompt loading / override
  layer 3 — postprocess + lints
  layer 4 — section gating in the public DTO

Reuses the fixture builders from test_synthesizer so the objects under
test are the real Pydantic models, not stand-ins.
"""
from __future__ import annotations

import json

import pytest

from scripts.chatbot.aggregator import AggregatedResult, AggregatedValue
from scripts.chatbot.models import ExtractedConcept
from scripts.chatbot.orchestrator import QueryResponse, StageMetrics
from scripts.chatbot.synthesizer import (
    SynthesizedAnswer, _BUILTIN_SYSTEM_PROMPT, build_synthesis_bundle,
    load_system_prompt, render_bundle_payload,
)

from tests.test_synthesizer import _entry, _geo, _make_plan

from app import lints, postprocess
from app import presentation as pres_mod
from app.promptlab import RunStore, StoredRun, render_run
from app.schemas import to_public


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def scenario():
    """A complete, realistic pipeline result: DeKalb median income."""
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="median income")
    entry = _entry("median household income", "B19013")
    plan = _make_plan(geo, concept, entry, 2023)
    aggregated = AggregatedResult(values=[
        AggregatedValue(
            geo=geo, concept=concept, year=2023,
            value=72483.0, sample_size=98000.0, rows_aggregated=1,
        ),
    ])
    answer = SynthesizedAnswer(
        prose="In 2023, DeKalb County's median household income was $72,500.",
        key_findings=["Median income: $72,500", "Based on 98,000 households"],
        caveats=["ACS 5-year estimates carry margins of error."],
        citations=[],
    )
    resp = QueryResponse(
        query="median income in DeKalb",
        intent=plan.intent,
        resolved_geos=[geo],
        plan=plan,
        aggregated=aggregated,
        answer=answer,
        metrics=StageMetrics(total_s=3.2, synthesize_s=1.1, llm_calls=2),
    )
    return resp, answer, aggregated


@pytest.fixture
def default_pres():
    return pres_mod.load()


# ---------------------------------------------------------------------
# Layer 1 — payload gating
# ---------------------------------------------------------------------

def _bundle(scenario, **kw):
    resp, _, aggregated = scenario
    return build_synthesis_bundle(
        resp.query, resp.intent, resp.plan, aggregated, **kw)


def test_bundle_omits_empty_keys(scenario):
    """`"peer_contexts" in bundle` must be a real test of "did the
    pipeline produce peers", because postprocess branches on it."""
    assert "peer_contexts" not in _bundle(scenario)
    assert "peer_contexts" in _bundle(
        scenario, peer_contexts=[{"axis": "population", "peers": []}])


def test_default_payload_matches_pre_bundle_shape(scenario):
    """With no options the payload keeps the shape the prompt was
    written against."""
    payload = json.loads(render_bundle_payload(_bundle(scenario)))
    assert set(payload) >= {
        "user_query", "intent_summary", "aggregated_values", "fetch_failures",
    }


def test_send_can_suppress_peer_contexts(scenario):
    """The layer-1 dial must keep peers out of the payload — this is what
    replaces 'DO NOT mention peers' prompt instructions."""
    peers = [{"axis": "population", "peers": [{"geo_name": "Travis County"}]}]
    bundle = _bundle(scenario, peer_contexts=peers)

    assert "Travis County" in render_bundle_payload(bundle)

    off = render_bundle_payload(bundle, {"send": {"peer_contexts": False}})
    assert "Travis County" not in off
    assert "peer_contexts" not in json.loads(off)


def test_send_caps_values_and_reports_truncation(scenario):
    """A cap must tell the model it was capped, so it can't claim
    coverage of rows it never saw."""
    resp, _, _ = scenario
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="median income")
    many = AggregatedResult(values=[
        AggregatedValue(geo=geo, concept=concept, year=2000 + i,
                        value=float(i), rows_aggregated=1)
        for i in range(10)
    ])
    bundle = build_synthesis_bundle(resp.query, resp.intent, resp.plan, many)
    out = json.loads(render_bundle_payload(bundle, {"max_values_sent": 4}))
    assert len(out["aggregated_values"]) == 4
    assert out["values_truncated"] == 6


def test_send_can_suppress_fetch_failures(scenario):
    resp, _, _ = scenario
    agg = AggregatedResult(values=[], fetch_failures=["boom"])
    bundle = build_synthesis_bundle(resp.query, resp.intent, resp.plan, agg)
    assert "boom" in render_bundle_payload(bundle)
    assert "boom" not in render_bundle_payload(
        bundle, {"send": {"fetch_failures": False}})


def test_unknown_bundle_keys_are_inspectable_but_not_sent(scenario):
    """The durability property: an artifact a future dataset adds shows
    up in the inventory but does NOT silently start reaching the model
    (which would change answers and per-query cost without review)."""
    from scripts.chatbot.synthesizer import bundle_inventory
    bundle = _bundle(scenario, hmda_loan_context={"denials": 42})

    assert "hmda_loan_context" not in render_bundle_payload(bundle)

    row = next(r for r in bundle_inventory(bundle)
               if r["key"] == "hmda_loan_context")
    assert row["known"] is False
    assert row["sent"] is False
    assert "42" in row["preview"]

    # ...and opting in works without any code change.
    on = render_bundle_payload(bundle, {"send": {"hmda_loan_context": True}})
    assert "42" in on


def test_structural_keys_never_reach_the_model(scenario):
    payload = json.loads(render_bundle_payload(
        _bundle(scenario), {"send": {"plan": True, "aggregated": True}}))
    assert "plan" not in payload
    assert "aggregated" not in payload


# ---------------------------------------------------------------------
# Layer 2 — prompt loading
# ---------------------------------------------------------------------

def test_prompt_loads_from_yaml_and_matches_builtin():
    """The externalized YAML must be byte-identical to the constant it
    replaced, or the refactor silently changed model behavior."""
    assert load_system_prompt() == _BUILTIN_SYSTEM_PROMPT


def test_prompt_falls_back_when_file_missing(tmp_path):
    assert load_system_prompt(tmp_path / "nope.yaml") == _BUILTIN_SYSTEM_PROMPT


def test_prompt_falls_back_when_file_malformed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("system: [this is a list, not a string]\n")
    assert load_system_prompt(bad) == _BUILTIN_SYSTEM_PROMPT


def test_prompt_hot_reloads(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("system: |\n  first\n")
    assert load_system_prompt(p).strip() == "first"
    p.write_text("system: |\n  second\n")
    assert load_system_prompt(p).strip() == "second"


# ---------------------------------------------------------------------
# Layer 3 — postprocess
# ---------------------------------------------------------------------

def test_postprocess_strips_fences_and_normalizes():
    a = SynthesizedAnswer(
        prose="```\nSome   prose\n\n\n\nwith gaps\n```",
        key_findings=["  padded  "], caveats=[], citations=[],
    )
    out = postprocess.clean(a, {}, {"strip_markdown_fences": True,
                                    "normalize_whitespace": True})
    assert "```" not in out.prose
    assert "Some prose" in out.prose
    assert "\n\n\n" not in out.prose
    assert out.key_findings == ["padded"]


def test_postprocess_truncates_on_sentence_boundary():
    prose = "One two three four five. Six seven eight nine ten. Eleven twelve."
    out = postprocess.clean(
        SynthesizedAnswer(prose=prose, key_findings=[], caveats=[], citations=[]),
        {},
        {"prose_max_words": 8},
    )
    assert out.prose.endswith(".")
    assert "Eleven" not in out.prose


def test_postprocess_caps_lists():
    a = SynthesizedAnswer(
        prose="x", key_findings=[f"k{i}" for i in range(10)],
        caveats=[f"c{i}" for i in range(10)], citations=[],
    )
    out = postprocess.clean(a, {}, {"max_key_findings": 3, "max_caveats": 2})
    assert len(out.key_findings) == 3
    assert len(out.caveats) == 2


def test_postprocess_does_not_mutate_input():
    """The trace pane shows the raw answer, so cleanup must not alter it."""
    a = SynthesizedAnswer(prose="```x```", key_findings=[], caveats=[], citations=[])
    postprocess.clean(a, {}, {"strip_markdown_fences": True})
    assert a.prose == "```x```"


# ---------------------------------------------------------------------
# Layer 3 — lints
# ---------------------------------------------------------------------

def _lint_rules(prose, aggregated, **cfg):
    base = {"forbid_unsourced_numbers": True, "require_year_mentioned": True,
            "forbid_peer_names": True}
    base.update(cfg)
    answer = SynthesizedAnswer(
        prose=prose, key_findings=[], caveats=[], citations=[],
    )
    return {l.rule for l in lints.run_all(answer, aggregated, None, base)}


def test_lint_accepts_rounded_sourced_numbers(scenario):
    """The prompt tells the model to round, so $72,483 -> $72,500 must
    not be reported as a hallucination."""
    _, _, aggregated = scenario
    rules = _lint_rules(
        "In 2023, median household income was $72,500.", aggregated,
    )
    assert "forbid_unsourced_numbers" not in rules


def test_lint_catches_invented_number(scenario):
    _, _, aggregated = scenario
    rules = _lint_rules(
        "In 2023, median household income was $95,200.", aggregated,
    )
    assert "forbid_unsourced_numbers" in rules


def test_lint_does_not_flag_years_as_unsourced(scenario):
    """Regression: a loose number regex tokenizes 'In 2023, income' as
    '2023,' which misses the year exemption and flags every date."""
    _, _, aggregated = scenario
    rules = _lint_rules(
        "In 2023, and again in 2019, income was $72,483.", aggregated,
    )
    assert "forbid_unsourced_numbers" not in rules


def test_lint_accepts_percent_rendering_of_a_ratio():
    """A ratio of 0.135 may legitimately be written as 13.5%."""
    geo = _geo("DeKalb County")
    concept = ExtractedConcept(text="poverty rate")
    agg = AggregatedResult(values=[
        AggregatedValue(geo=geo, concept=concept, year=2023,
                        ratio=0.135, rows_aggregated=1),
    ])
    rules = _lint_rules("In 2023 the poverty rate was 13.5%.", agg)
    assert "forbid_unsourced_numbers" not in rules


def test_lint_requires_year(scenario):
    _, _, aggregated = scenario
    assert "require_year_mentioned" in _lint_rules(
        "Median household income was $72,483.", aggregated,
    )


def test_lint_catches_peer_name_leak(scenario):
    _, _, aggregated = scenario
    peers = [{"axis": "pop", "peers": [{"geo_name": "Travis County"}]}]
    answer = SynthesizedAnswer(
        prose="In 2023 income was $72,483, similar to Travis County.",
        key_findings=[], caveats=[], citations=[],
    )
    found = {l.rule for l in lints.run_all(
        answer, aggregated, peers, {"forbid_peer_names": True},
    )}
    assert "forbid_peer_names" in found


def test_lints_never_raise_on_empty_answer(scenario):
    _, _, aggregated = scenario
    assert lints.run_all(None, aggregated, None, {}) == []


# ---------------------------------------------------------------------
# Layer 4 — section gating
# ---------------------------------------------------------------------

def test_sections_off_omits_data_entirely(scenario):
    """A section switched off must be absent from the payload, not
    merely hidden by the frontend."""
    resp, answer, _ = scenario
    out = to_public(
        resp,
        sections={"caveats": False, "values_table": False, "metrics": False},
        answer_override=answer,
    )
    assert out.prose
    assert out.caveats == []
    assert out.values == []
    assert out.metrics is None


def test_sections_on_includes_values_and_metrics(scenario):
    resp, answer, _ = scenario
    out = to_public(resp, sections={}, answer_override=answer)
    assert out.status == "ok"
    assert len(out.values) == 1
    assert out.values[0].geo == "DeKalb County"
    assert out.values[0].value == 72483.0
    assert out.metrics.llm_calls == 2


def test_trace_included_only_when_section_on(scenario):
    resp, answer, _ = scenario
    trace = {"system_prompt": "SYS", "user_payload": "{}", "raw_response": {}}
    on = to_public(resp, sections={"trace": True}, trace=trace,
                   answer_override=answer)
    off = to_public(resp, sections={"trace": False}, trace=trace,
                    answer_override=answer)
    assert on.trace.system_prompt == "SYS"
    assert off.trace is None


def test_status_reflects_failure(scenario):
    resp, _, _ = scenario
    resp.error = "synthesis failed"
    out = to_public(resp, sections={}, answer_override=None)
    assert out.status == "error"
    assert out.error == "synthesis failed"


def test_no_answer_maps_to_no_data(scenario):
    resp, _, _ = scenario
    resp.answer = None
    out = to_public(resp, sections={}, answer_override=None)
    assert out.status == "no_data"


# ---------------------------------------------------------------------
# render_run — the full layer 3 + 4 path used by /rerender
# ---------------------------------------------------------------------

def test_render_run_applies_cleanup_and_lints(scenario, default_pres):
    resp, answer, _ = scenario
    answer.prose = "```\nIn 2023, income was   $95,200.\n```"
    run = StoredRun(run_id="r1", resp=resp, raw_answer=answer,
                    bundle=build_synthesis_bundle(
                        resp.query, resp.intent, resp.plan, resp.aggregated),
                    trace={"system_prompt": "S", "user_payload": "{}"})
    out = render_run(run, default_pres)

    assert "```" not in out.prose            # layer 3 cleanup ran
    assert "  " not in out.prose
    assert any(l.rule == "forbid_unsourced_numbers" for l in out.lints)
    assert out.run_id == "r1"


def test_render_run_survives_broken_postprocess(scenario, default_pres, monkeypatch):
    """A student's bad custom_clean must degrade to the raw answer with a
    warning, not 500 the app."""
    resp, answer, _ = scenario

    def boom(a, b, f):
        raise RuntimeError("student bug")

    monkeypatch.setattr(postprocess, "clean", boom)
    run = StoredRun(run_id="r2", resp=resp, raw_answer=answer,
                    bundle=build_synthesis_bundle(
                        resp.query, resp.intent, resp.plan, resp.aggregated))
    out = render_run(run, default_pres)
    assert out.prose == answer.prose
    assert "student bug" in (out.config_warning or "")


# ---------------------------------------------------------------------
# Presentation config
# ---------------------------------------------------------------------

def test_presentation_defaults_when_missing(tmp_path):
    cfg = pres_mod.load(tmp_path / "absent.yaml")
    assert cfg.error is None
    assert cfg.shows("prose") is True


def test_presentation_reports_parse_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("sections: [unclosed\n")
    cfg = pres_mod.load(bad)
    assert cfg.error is not None
    assert cfg.shows("prose") is True          # still usable


def test_presentation_partial_file_keeps_other_defaults(tmp_path):
    """Deleting a block from the file must not wipe unrelated defaults."""
    p = tmp_path / "p.yaml"
    p.write_text("sections:\n  trace: false\n")
    cfg = pres_mod.load(p)
    assert cfg.shows("trace") is False
    assert cfg.shows("prose") is True
    assert cfg.bundle_options["send"]["peer_contexts"] is False


def test_presentation_write_rejects_invalid_yaml(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("sections:\n  trace: true\n")
    err = pres_mod.write_text("sections: [unclosed\n", p)
    assert err is not None
    assert "trace: true" in p.read_text()      # original preserved


# ---------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------

def test_run_store_evicts_oldest():
    store = RunStore(maxsize=3)
    for i in range(5):
        store.put(StoredRun(run_id=f"r{i}", resp=None))
    assert store.get("r0") is None
    assert store.get("r4") is not None
    assert store.latest().run_id == "r4"


def test_run_store_get_refreshes_lru():
    store = RunStore(maxsize=2)
    store.put(StoredRun(run_id="a", resp=None))
    store.put(StoredRun(run_id="b", resp=None))
    store.get("a")                              # 'a' now most recent
    store.put(StoredRun(run_id="c", resp=None))
    assert store.get("a") is not None
    assert store.get("b") is None
