"""LLM client abstraction for the chatbot extractor.

Two implementations:
  GeminiVertexClient — production client against Vertex AI (gemini-2.5-flash).
                       Uses explicit prompt caching to cut per-call cost
                       when the system prompt is stable across many turns.
  MockLLMClient      — scripted returns for unit tests; no network.

Both conform to the ``LLMClient`` Protocol so the extractor doesn't care
which one it's holding. google-genai is imported lazily inside
GeminiVertexClient so the test suite can run without the SDK installed.
"""
from __future__ import annotations

import inspect
import json
import logging
import threading
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Synchronous extraction contract."""

    def extract(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema: Any,
        temperature: float = 0.1,
    ) -> dict: ...


class LLMCallError(RuntimeError):
    """Raised when an LLM client fails to produce a parseable JSON response."""


# Gemini 2.5 Flash pricing on Vertex AI, USD per 1M tokens.
# https://cloud.google.com/vertex-ai/generative-ai/pricing
# Adjust if Google changes pricing.
FLASH_PRICE_INPUT_PER_M = 0.15
FLASH_PRICE_CACHED_INPUT_PER_M = 0.0375   # 75% discount on cache hits
FLASH_PRICE_OUTPUT_PER_M = 0.60


def _guess_caller_module() -> str:
    """Walk the stack to find the first frame outside this module —
    that's the node (extractor / grant_frame_matcher / synthesizer /
    plan_reviewer / …) that made the call. Used only to tag slow-
    call log lines; failures fall back to 'unknown'."""
    try:
        for frame_info in inspect.stack()[2:]:           # skip extract() + helper
            mod = inspect.getmodule(frame_info.frame)
            if mod is None:
                continue
            name = mod.__name__
            if name == __name__:
                continue
            if name.startswith("scripts.chatbot."):
                return name.split(".")[-1]
            return name
    except Exception:                                   # pragma: no cover
        return "unknown"
    return "unknown"


def _zero_usage() -> dict:
    return {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
    }


def estimate_cost_usd(usage: dict) -> float:
    """Compute an approximate USD cost for the given usage counters."""
    if not usage:
        return 0.0
    cached = usage.get("cached_tokens", 0) or 0
    fresh_input = max(0, (usage.get("input_tokens", 0) or 0) - cached)
    output = usage.get("output_tokens", 0) or 0
    return (
        fresh_input * FLASH_PRICE_INPUT_PER_M / 1_000_000
        + cached * FLASH_PRICE_CACHED_INPUT_PER_M / 1_000_000
        + output * FLASH_PRICE_OUTPUT_PER_M / 1_000_000
    )


class GeminiVertexClient:
    """Vertex AI Gemini client with explicit prompt caching.

    Args:
        project_id: GCP project (billed).
        location:   Vertex region (e.g., "us-central1").
        model:      Model ID (default "gemini-2.5-flash").
        temperature: Default sampling temperature; callers may override.
        timeout_seconds: End-to-end call timeout.
        enable_cache: If True, create an explicit cached-content entry
                       for the system prompt and reuse it across calls.
                       Disable for debugging or when prompts change every call.
        cache_ttl_seconds: Lifetime of cached content entries.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        timeout_seconds: float = 60.0,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        # Lazy import so MockLLMClient-only tests don't require google-genai.
        from google import genai  # type: ignore
        self._genai = genai
        self._client = genai.Client(
            vertexai=True, project=project_id, location=location,
        )
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cached-content state. Invalidated if the system_prompt changes.
        self._cached_name: Optional[str] = None
        self._cached_prompt: Optional[str] = None

        # Usage counters — accumulate across calls since the last reset.
        # Consumers (the REPL) reset before each user turn, then read
        # the counters after to report latency + token spend + cost.
        self._usage = _zero_usage()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _ensure_cache(self, system_prompt: str) -> Optional[str]:
        """Return a cached-content name for this prompt, creating one if
        necessary. Returns None if caching is disabled or creation failed.
        """
        if not self.enable_cache:
            return None
        if self._cached_prompt != system_prompt:
            # Prompt changed — invalidate. We don't explicitly delete the
            # old cache (it'll TTL-out) to keep this method fast.
            self._cached_name = None
            self._cached_prompt = system_prompt
        if self._cached_name is not None:
            return self._cached_name

        try:
            from google.genai import types  # type: ignore
            cache = self._client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl=f"{self.cache_ttl_seconds}s",
                ),
            )
            self._cached_name = cache.name
            logger.info(
                "Prompt cache created: %s (ttl=%ds, %d chars)",
                cache.name, self.cache_ttl_seconds, len(system_prompt),
            )
            return self._cached_name
        except Exception as e:
            # Vertex may reject caching for prompts under its minimum token
            # threshold. Log once and fall back to uncached calls.
            logger.warning(
                "Prompt caching disabled for this session — create failed: %s",
                e,
            )
            self.enable_cache = False
            return None

    def invalidate_cache(self) -> None:
        """Drop cache state so the next call rebuilds. Useful when the
        prompt file is edited mid-session."""
        self._cached_name = None
        self._cached_prompt = None

    # ------------------------------------------------------------------
    # Usage counters
    # ------------------------------------------------------------------

    def reset_usage_counters(self) -> None:
        """Zero the usage accumulator — call before each logical unit of
        work (e.g., one user turn) to capture just that turn's spend."""
        self._usage = _zero_usage()

    def get_usage_counters(self) -> dict:
        """Return a snapshot of accumulated usage since the last reset."""
        return dict(self._usage)

    def _record_usage(self, response) -> None:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return
        self._usage["calls"] += 1
        self._usage["input_tokens"] += (
            getattr(meta, "prompt_token_count", 0) or 0
        )
        self._usage["output_tokens"] += (
            getattr(meta, "candidates_token_count", 0) or 0
        )
        # Vertex exposes the cache-hit portion as cached_content_token_count.
        self._usage["cached_tokens"] += (
            getattr(meta, "cached_content_token_count", 0) or 0
        )

    # ------------------------------------------------------------------
    # Public: extract
    # ------------------------------------------------------------------

    def extract(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema: Any,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ) -> dict:
        """Make one Vertex call and parse the JSON body.

        A hard wall-clock timeout is enforced by running the underlying
        ``generate_content`` call on a worker thread and abandoning it
        if it exceeds ``timeout_seconds`` (defaults to
        ``self.timeout_seconds``). The google-genai SDK doesn't honor
        any per-call timeout on its own, so without this wrapper a
        stuck Vertex request can freeze the REPL indefinitely —
        observed historically on long synthesizer prose and frame-
        match calls under load.
        """
        from google.genai import types  # type: ignore

        cached_name = self._ensure_cache(system_prompt)

        cfg_kwargs: dict = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if cached_name is not None:
            cfg_kwargs["cached_content"] = cached_name
        else:
            cfg_kwargs["system_instruction"] = system_prompt

        effective_timeout = (
            timeout_seconds if timeout_seconds is not None
            else self.timeout_seconds
        )

        # We run the blocking google-genai call on a daemon thread
        # rather than via ThreadPoolExecutor. When a Vertex call
        # actually hangs, the `with ThreadPoolExecutor()` context
        # manager calls shutdown(wait=True) on exit which BLOCKS on
        # the still-running worker — so the caller-side timeout is
        # cosmetic, the whole pipeline stays frozen, and the only
        # thing that unsticks it is the outer signal watchdog. A
        # manually-started daemon thread avoids that: when the
        # timeout fires, we just stop waiting. The worker thread
        # keeps running in the background until google-genai
        # eventually gives up (or until the process exits, since
        # daemon=True means it doesn't block shutdown).
        holder: dict[str, Any] = {"response": None, "error": None}

        def _invoke():
            try:
                holder["response"] = self._client.models.generate_content(
                    model=self.model,
                    contents=user_text,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
            except BaseException as e:                 # noqa: BLE001
                holder["error"] = e

        # Tag the call with the closest identifiable caller so slow-
        # call logs let you see WHICH node stage was slow, without
        # plumbing a name argument through every caller.
        call_site = _guess_caller_module()
        t0 = time.monotonic()
        worker = threading.Thread(
            target=_invoke, name=f"vertex-{call_site}", daemon=True,
        )
        worker.start()
        worker.join(timeout=effective_timeout)
        elapsed = time.monotonic() - t0
        if worker.is_alive():
            raise LLMCallError(
                f"Vertex call timed out after {effective_timeout}s "
                f"(caller={call_site}, worker thread left running)"
            )
        # Log elapsed time for every call so slow-but-not-hung calls
        # are visible. Anything over 15s gets bumped to WARNING so
        # it's easy to grep for slow Vertex periods.
        if elapsed >= 15.0:
            logger.warning(
                "[SLOW Vertex] %s: %.1fs", call_site, elapsed,
            )
        else:
            logger.info(
                "Vertex call %s: %.2fs", call_site, elapsed,
            )

        try:
            if holder["error"] is not None:
                raise holder["error"]
            response = holder["response"]
        except LLMCallError:
            raise
        except Exception as e:
            # A CacheExpired or NotFound error can happen if our cache
            # TTL elapses; drop state and retry uncached once.
            msg = str(e).lower()
            if cached_name and ("cache" in msg or "expired" in msg or "not found" in msg):
                logger.info("Cache rejected by API; retrying without cache: %s", e)
                self.invalidate_cache()
                return self.extract(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    schema=schema,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
            raise LLMCallError(f"Vertex call failed: {e}") from e

        raw = response.text or ""
        if not raw.strip():
            raise LLMCallError("Vertex returned an empty response body")

        self._record_usage(response)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMCallError(
                f"Vertex response was not valid JSON: {raw[:200]!r}"
            ) from e


class MockLLMClient:
    """Scripted client for unit tests.

    Construct with a dict of ``{user_text_normalized: response_dict}``.
    On ``extract``, looks up the normalized user_text and returns the
    scripted response. Records every call for test assertions.
    """

    def __init__(self, responses: Optional[dict[str, dict]] = None):
        self.responses: dict[str, dict] = responses or {}
        self.calls: list[dict] = []

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def extract(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema: Any,
        temperature: float = 0.1,
    ) -> dict:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_text": user_text,
            "schema": schema,
            "temperature": temperature,
        })
        key = self._normalize(user_text)
        if key in self.responses:
            return self.responses[key]
        # Fuzzy fallback: first-line match so tests can include session
        # context in user_text without needing to re-script.
        first_line = user_text.split("\n", 1)[0]
        key2 = self._normalize(first_line)
        if key2 in self.responses:
            return self.responses[key2]
        raise KeyError(
            f"MockLLMClient has no response scripted for {user_text!r}"
        )


def build_default_client(config: dict) -> GeminiVertexClient:
    """Factory that reads config/chatbot.yaml and returns a ready client.

    Resolves project_id from config or env (GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT).
    Caller is responsible for having already run auth_check.check_auth().
    """
    import os
    vertex = config.get("vertex_ai", {})
    project_id = (
        vertex.get("project_id")
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    if not project_id:
        raise RuntimeError(
            "No GCP project resolved. Set GCP_PROJECT_ID or put "
            "vertex_ai.project_id in config/chatbot.yaml."
        )
    return GeminiVertexClient(
        project_id=project_id,
        location=vertex.get("location", "us-central1"),
        model=vertex.get("model", "gemini-2.5-flash"),
        temperature=vertex.get("temperature", 0.1),
        timeout_seconds=vertex.get("timeout_seconds", 60.0),
        enable_cache=vertex.get("enable_cache", True),
        cache_ttl_seconds=vertex.get("cache_ttl_seconds", 3600),
    )
