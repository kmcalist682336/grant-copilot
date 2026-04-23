"""End-to-end orchestrator tests with mocked LLM and Census transport."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from scripts.chatbot.api_cache import APICache
from scripts.chatbot.census_caller import CensusCaller
from scripts.chatbot.concept_map import ConceptMap
from scripts.chatbot.decomposition_cache import (
    Decomposition, DecompositionCache,
)
from scripts.chatbot.orchestrator import (
    QueryResponse, _decompose_pending, answer_query,
)


# ---------------------------------------------------------------------------
# Helpers — scripted LLM that branches on system_prompt
# ---------------------------------------------------------------------------

class _RoutedLLM:
    """LLM mock that routes calls to scripted responses keyed by a prompt
    substring. Lets us cover the orchestrator's three distinct LLM
    callers (extractor, decomposer, synthesizer) with one client."""

    def __init__(self, routes: list[tuple[str, dict | list[dict]]]):
        # routes: [(prompt_substring, response_or_queue), ...]
        # Tried in order; first match wins. Queues pop on each match.
        self._routes = []
        for substr, resp in routes:
            queue = list(resp) if isinstance(resp, list) else [resp]
            self._routes.append((substr, queue))
        self.calls: list[dict] = []
        self._counters = {
            "calls": 0, "input_tokens": 0,
            "cached_tokens": 0, "output_tokens": 0,
        }

    def extract(self, *, system_prompt, user_text, schema, temperature=0.1):
        self.calls.append({"system_prompt": system_prompt[:80],
                           "user_text": user_text[:120]})
        self._counters["calls"] += 1
        self._counters["input_tokens"] += 100
        self._counters["output_tokens"] += 30
        for substr, queue in self._routes:
            if substr in system_prompt:
                if not queue:
                    raise AssertionError(
                        f"route {substr!r} ran out of responses"
                    )
                return queue.pop(0)
        raise AssertionError(
            f"no route matched system_prompt prefix: "
            f"{system_prompt[:200]!r}"
        )

    def reset_usage_counters(self) -> None:
        self._counters = dict.fromkeys(self._counters, 0)

    def get_usage_counters(self) -> dict:
        return dict(self._counters)


# ---------------------------------------------------------------------------
# Census API mock transport
# ---------------------------------------------------------------------------

def _census_handler(routes: dict[str, list]) -> httpx.MockTransport:
    """routes: {url_prefix: [header_row, *data_rows]}"""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("&key=")[0]   # strip API key
        for prefix, rows in routes.items():
            if url.startswith(prefix):
                return httpx.Response(200, json=rows)
        return httpx.Response(404, json=[])

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _patch_census_transport(monkeypatch):
    """Monkeypatch CensusCaller's __aenter__ to inject a default empty
    transport (overridden per-test where the test needs custom routes)."""
    # We don't override here — tests that need it set their own transport.
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cmap_for_orch():
    return ConceptMap.from_yaml()


@pytest.fixture
def runtime_caches(tmp_path):
    api = APICache(tmp_path / "api.db")
    decomp = DecompositionCache(tmp_path / "decomp.db")
    return api, decomp


@pytest.fixture
def empty_config():
    return {"vertex_ai": {"temperature": 0.0, "synth_temperature": 0.0}}


# ---------------------------------------------------------------------------
# Orchestrator: end-to-end happy path with monkeypatched CensusCaller
# ---------------------------------------------------------------------------

def _patched_caller(monkeypatch, transport: httpx.MockTransport):
    """Wrap CensusCaller so __aenter__ sees a MockTransport."""
    orig_init = CensusCaller.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("transport", transport)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(CensusCaller, "__init__", patched_init)


def test_orchestrator_happy_path_county_query(
    gazetteer_db, metadata_db, cmap_for_orch, runtime_caches,
    empty_config, monkeypatch,
):
    """Simple county query: extract → resolve → plan → fetch → aggregate
    → synthesize. Mocks out the LLM and the Census HTTP."""
    api_cache, decomp_cache = runtime_caches

    # LLM routes: extractor, synthesizer.
    llm = _RoutedLLM(routes=[
        ("intent extractor", {
            "geo_refs": [{
                "text": "DeKalb County",
                "ref_type": "administrative",
                "scope_hint": "Georgia",
            }],
            "concepts": [{
                "text": "median income",
                "canonical_hint": "median household income",
            }],
            "intent_type": "exact",
        }),
        ("Census data synthesizer", {
            "prose": "DeKalb County's median household income was $72,500.",
            "key_findings": ["$72,500"],
            "caveats": [],
        }),
    ])

    # Census mock — return a single row of B19013_001E.
    transport = _census_handler({
        "https://api.census.gov/data/": [
            ["NAME", "B19013_001E", "state", "county"],
            ["DeKalb County, GA", "72500", "13", "089"],
        ],
    })
    _patched_caller(monkeypatch, transport)

    response = asyncio.run(answer_query(
        "What's the median income in DeKalb County?",
        llm, gazetteer_db, metadata_db, cmap_for_orch,
        decomp_cache=decomp_cache, api_cache=api_cache,
        api_key=None, config=empty_config,
    ))
    assert isinstance(response, QueryResponse)
    assert response.error is None
    assert response.answer is not None
    assert "72,500" in response.answer.prose
    assert response.metrics.census_calls_total >= 1
    assert response.metrics.census_failures == 0
    # Aggregated value should carry the right raw number
    vals = [v.value for v in response.aggregated.values if v.value is not None]
    assert 72500.0 in vals


# ---------------------------------------------------------------------------
# Orchestrator: composite concept triggers decomposition + re-plan
# ---------------------------------------------------------------------------

def test_orchestrator_composite_decomposes_then_replans(
    gazetteer_db, metadata_db, cmap_for_orch, runtime_caches,
    empty_config, monkeypatch,
):
    """User asks about 'food deserts' (composite, not in cmap, no cache).
    Orchestrator should call decomposer → cache → re-plan → fetch
    sub-concepts → synthesize."""
    api_cache, decomp_cache = runtime_caches

    llm = _RoutedLLM(routes=[
        ("intent extractor", {
            "geo_refs": [{
                "text": "DeKalb County",
                "ref_type": "administrative",
                "scope_hint": "Georgia",
            }],
            "concepts": [{
                "text": "food deserts",
                "canonical_hint": "food deserts",
                "is_composite": True,
            }],
            "intent_type": "exact",
        }),
        ("decompose composite US Census concepts", {
            "sub_concepts": ["snap participation", "households without a vehicle"],
            "confidence": 0.6,
            "rationale": "ACS proxies for food access.",
        }),
        ("Census data synthesizer", {
            "prose": "Food access proxies in DeKalb County: SNAP and vehicle access summarized.",
            "key_findings": ["SNAP and vehicles"],
            "caveats": ["Composite proxy estimate."],
        }),
    ])

    # SNAP and vehicles both use B22003 / B25044. Return non-empty rows.
    transport = _census_handler({
        "https://api.census.gov/data/": [
            ["NAME", "X_001E", "X_002E", "Y_001E", "Y_003E", "Y_010E",
             "state", "county"],
            ["DeKalb", "100000", "12000", "100000", "5000", "8000",
             "13", "089"],
        ],
    })
    _patched_caller(monkeypatch, transport)

    response = asyncio.run(answer_query(
        "Tell me about food deserts in DeKalb County.",
        llm, gazetteer_db, metadata_db, cmap_for_orch,
        decomp_cache=decomp_cache, api_cache=api_cache,
        api_key=None, config=empty_config,
    ))
    assert response.error is None
    # Decomposer was called
    decomp_calls = [
        c for c in llm.calls
        if "decompose" in c["system_prompt"].lower()
    ]
    assert decomp_calls, "expected decomposer LLM call"
    # And the cache now has the entry
    cached = decomp_cache.get("food deserts")
    assert cached is not None
    assert cached.sub_concepts == [
        "snap participation", "households without a vehicle",
    ]
    # Synthesizer was called and produced prose
    assert response.answer is not None
    assert "DeKalb" in response.answer.prose
    # Plan should have re-planned with sub-concepts → multiple calls
    assert len(response.plan.calls) >= 1


# ---------------------------------------------------------------------------
# Orchestrator: extraction failure returns a graceful error
# ---------------------------------------------------------------------------

def test_orchestrator_short_input_returns_error_response(
    gazetteer_db, metadata_db, cmap_for_orch, runtime_caches,
    empty_config,
):
    api_cache, decomp_cache = runtime_caches
    # Empty LLM — extractor short-circuits on the empty input first
    llm = _RoutedLLM(routes=[("never matches", {})])
    response = asyncio.run(answer_query(
        "x",   # too short — InputTooShort short-circuit
        llm, gazetteer_db, metadata_db, cmap_for_orch,
        decomp_cache=decomp_cache, api_cache=api_cache,
        api_key=None, config=empty_config,
    ))
    assert response.error is not None
    assert "extraction failed" in response.error
    assert response.answer is None


# ---------------------------------------------------------------------------
# Orchestrator: census failure surfaces in caveats / metrics
# ---------------------------------------------------------------------------

def test_orchestrator_census_500_propagates_to_metrics(
    gazetteer_db, metadata_db, cmap_for_orch, runtime_caches,
    empty_config, monkeypatch,
):
    api_cache, decomp_cache = runtime_caches
    llm = _RoutedLLM(routes=[
        ("intent extractor", {
            "geo_refs": [{
                "text": "DeKalb County",
                "ref_type": "administrative",
                "scope_hint": "Georgia",
            }],
            "concepts": [{
                "text": "median income",
                "canonical_hint": "median household income",
            }],
            "intent_type": "exact",
        }),
        ("Census data synthesizer", {
            "prose": "Census did not return data for this query.",
            "key_findings": [],
            "caveats": ["Upstream Census API error."],
        }),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"server error")

    transport = httpx.MockTransport(handler)
    _patched_caller(monkeypatch, transport)

    response = asyncio.run(answer_query(
        "median income in DeKalb",
        llm, gazetteer_db, metadata_db, cmap_for_orch,
        decomp_cache=decomp_cache, api_cache=api_cache,
        api_key=None, config=empty_config,
    ))
    assert response.error is None  # We don't raise; we degrade
    assert response.metrics.census_failures >= 1
    # No data → no aggregated value with a real number
    real_values = [
        v for v in response.aggregated.values
        if v.value is not None or v.ratio is not None
    ]
    assert real_values == []


# ---------------------------------------------------------------------------
# _decompose_pending unit
# ---------------------------------------------------------------------------

def test_decompose_pending_calls_llm_for_each_unresolved(
    gazetteer_db, metadata_db, cmap_for_orch, tmp_path,
):
    """_decompose_pending must call decomposer once per unresolved
    composite and write each into the cache."""
    from scripts.chatbot.geo_resolver import resolve_intent
    from scripts.chatbot.models import (
        ExtractedConcept, ExtractedGeoRef, ExtractedIntent,
    )
    from scripts.chatbot.planner import plan_query

    cache = DecompositionCache(tmp_path / "decomp.db")
    intent = ExtractedIntent(
        geo_refs=[ExtractedGeoRef(text="Atlanta", ref_type="administrative",
                                  scope_hint="Georgia")],
        concepts=[
            ExtractedConcept(text="food deserts", is_composite=True),
            ExtractedConcept(text="urban heat", is_composite=True),
        ],
    )
    resolved = resolve_intent(intent, gazetteer_db)
    plan = plan_query(intent, resolved, cmap_for_orch, metadata_db,
                      decomp_cache=cache)

    llm = _RoutedLLM(routes=[
        ("decompose composite US Census concepts", [
            {"sub_concepts": ["snap participation"], "confidence": 0.5},
            {"sub_concepts": ["total population"], "confidence": 0.4},
        ]),
    ])
    n = _decompose_pending(plan, llm, cmap_for_orch, cache)
    assert n == 2
    assert cache.get("food deserts") is not None
    assert cache.get("urban heat") is not None
