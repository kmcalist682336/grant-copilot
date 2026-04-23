"""vLLM inference client for the enrichment pipeline.

Wraps vLLM's OpenAI-compatible REST API with:
  - JSON schema guided decoding for reliable structured output
  - Async batch submission for throughput
  - Retry logic with exponential backoff
  - Provenance tracking on every response
  - Health check before committing to a full run

vLLM must be running as a server before Stage 4 starts:
    python -m vllm.entrypoints.openai.api_server \\
        --model <model_path> \\
        --port 8000 \\
        --max-model-len 32768 \\
        --gpu-memory-utilization 0.90

The model path can be set via CENSUS_MODEL_PATH env var or inference.yaml.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response and provenance types
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """A single LLM generation result with provenance metadata."""
    content: dict                   # Parsed JSON output
    raw_text: str                   # Raw model output before parsing
    prompt_name: str                # Which prompt template produced this
    prompt_hash: str                # sha256 of the rendered prompt
    model_id: str                   # Model identifier
    generation_settings: dict       # Temperature, max_tokens, etc.
    latency_seconds: float          # Wall-clock time for this call
    attempt: int                    # Which retry attempt (1 = first try)

    def provenance_block(self) -> dict:
        """Build the _provenance dict to embed in output files."""
        return {
            "model_id": self.model_id,
            "prompt_name": self.prompt_name,
            "prompt_hash": self.prompt_hash,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_settings": self.generation_settings,
            "latency_seconds": round(self.latency_seconds, 2),
            "attempt": self.attempt,
        }


class LLMCallError(Exception):
    """Raised when an LLM call fails after all retries."""


# ---------------------------------------------------------------------------
# JSON schema definitions for guided decoding
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict] = {
    "table_description": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
        },
        "required": ["description"],
        "additionalProperties": False,
    },
    "table_use_avoid": {
        "type": "object",
        "properties": {
            "use_when": {"type": "string"},
            "avoid_when": {"type": "string"},
        },
        "required": ["use_when", "avoid_when"],
        "additionalProperties": False,
    },
    "universe_rewrite": {
        "type": "object",
        "properties": {
            "universe_plain": {"type": "string"},
        },
        "required": ["universe_plain"],
        "additionalProperties": False,
    },
    "glossary_and_caveats": {
        "type": "object",
        "properties": {
            "glossary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                    },
                    "required": ["term", "definition"],
                    "additionalProperties": False,
                },
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["glossary", "caveats"],
        "additionalProperties": False,
    },
    "temporal_stability": {
        "type": "object",
        "properties": {
            "temporal_stability": {
                "type": "string",
                "enum": ["very_stable", "stable", "moderate", "volatile"],
            },
            "stability_rationale": {"type": "string"},
        },
        "required": ["temporal_stability", "stability_rationale"],
        "additionalProperties": False,
    },
    "flat_dataset_description": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "caveat": {"type": "string"},
        },
        "required": ["description", "caveat"],
        "additionalProperties": False,
    },
    # ── Stage 5 — Topic Tagging (discovery mode) ─────────────────────────
    # Discovery mode: model uses existing vocab but may propose new tags.
    "topic_tagging_discovery": {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["tag", "confidence", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 5,
            },
            "proposed_tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "proposed_l1": {"type": "string"},
                        "proposed_l2": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["tag", "proposed_l1", "proposed_l2", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tags", "proposed_tags"],
        "additionalProperties": False,
    },
    # ── Stage 5 — Topic Tagging (production mode) ────────────────────────
    # Production mode: vocabulary is locked, no proposals allowed.
    "topic_tagging_production": {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["tag", "confidence", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": ["tags"],
        "additionalProperties": False,
    },
    # ── Stage 6 — Question Generation ────────────────────────────────────
    "questions_canonical": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
    "questions_varied": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "measure_type_sought": {
                            "type": "string",
                            "enum": ["count", "rate", "median", "proportion",
                                     "change", "rank"],
                        },
                        "geography_specificity": {
                            "type": "string",
                            "enum": ["none", "state", "county", "place",
                                     "tract", "zip", "named_neighborhood"],
                        },
                        "temporal_framing": {
                            "type": "string",
                            "enum": ["point_in_time", "trend",
                                     "comparison_periods"],
                        },
                        "subpopulation_specified": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "aggregation_level": {
                            "type": "string",
                            "enum": ["individual", "household", "family",
                                     "housing_unit"],
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                        "register": {
                            "type": "string",
                            "enum": ["casual", "formal", "research",
                                     "comparison", "trend", "abbreviated"],
                        },
                    },
                    "required": [
                        "text", "measure_type_sought", "geography_specificity",
                        "temporal_framing", "subpopulation_specified",
                        "aggregation_level", "difficulty", "register",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
    "questions_negative": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "negative_type": {
                            "type": "string",
                            "enum": ["wrong_table", "wrong_geography",
                                     "requires_join"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["text", "negative_type", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },

    # ── Stage 7 — Variable descriptions (batched) ─────────────────────────
    "variable_description_batch": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "var_id": {"type": "string"},
                        "label_natural": {"type": "string"},
                        "description": {"type": "string"},
                        "column_name": {"type": "string"},
                    },
                    "required": ["var_id", "label_natural",
                                 "description", "column_name"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["variables"],
        "additionalProperties": False,
    },

    # ── Stage 7 — Variable questions (batched) ────────────────────────────
    "variable_questions_batch": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "var_id": {"type": "string"},
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 7,
                        },
                    },
                    "required": ["var_id", "questions"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["variables"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------

@dataclass
class VLLMClientConfig:
    host: str = "localhost"
    port: int = 8000
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    model_id: str = ""  # Auto-discovered from server if empty; set explicitly to override

    @classmethod
    def from_config(cls, config: dict) -> "VLLMClientConfig":
        v = config.get("vllm", {})
        b = config.get("batching", {})
        return cls(
            host=v.get("host", "localhost"),
            port=v.get("port", 8000),
            timeout_seconds=v.get("timeout_seconds", 120.0),
            max_retries=b.get("max_retries", 3),
            retry_delay_seconds=b.get("retry_delay_seconds", 2.0),
            # model_id deliberately not read from config — discovered from server
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# Async vLLM client
# ---------------------------------------------------------------------------

class VLLMClient:
    """Async client for vLLM's OpenAI-compatible API.

    Uses guided JSON decoding (via guided_json parameter) to guarantee
    structured output matching the schema for each prompt type.

    The model_id used in API calls is auto-discovered from the live server
    via GET /v1/models on first use, so it always matches whatever model
    vLLM is actually serving regardless of config file state.
    """

    def __init__(self, config: VLLMClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._model_id: Optional[str] = config.model_id or None
        self._working_structured_output_param: Optional[str] = None

    async def __aenter__(self) -> "VLLMClient":
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    async def _discover_model_id(self) -> str:
        """Query the server for the active model ID.

        vLLM's /v1/models endpoint returns the list of loaded models.
        We take the first one — in single-model deployments this is always
        the right answer.
        """
        url = f"{self.config.base_url}/v1/models"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"Could not discover model ID from {url}: HTTP {resp.status}"
                )
            data = await resp.json()
        models = data.get("data", [])
        if not models:
            raise RuntimeError(
                f"vLLM returned no models from {url}. "
                "Is the server fully started?"
            )
        model_id = models[0]["id"]
        logger.info("Discovered vLLM model ID: %s", model_id)
        return model_id

    async def get_model_id(self) -> str:
        """Return the active model ID, discovering it from the server if needed."""
        if not self._model_id:
            self._model_id = await self._discover_model_id()
        return self._model_id

    async def health_check(self) -> tuple[bool, str]:
        """Check that vLLM is running, responsive, and has a model loaded."""
        try:
            # Basic liveness check
            url = f"{self.config.base_url}/health"
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return False, f"vLLM health check returned HTTP {resp.status}"

            # Discover and cache model ID as part of health check
            model_id = await self.get_model_id()
            return True, f"vLLM server is healthy — serving model: {model_id}"

        except aiohttp.ClientConnectorError:
            return False, (
                f"Cannot connect to vLLM at {self.config.base_url}. "
                "Is the server running? Start it with:\n"
                "  python -m vllm.entrypoints.openai.api_server "
                f"--model <path> --port {self.config.port}"
            )
        except Exception as e:
            return False, f"Health check failed: {e}"

    async def generate(
        self,
        prompt: str,
        prompt_name: str,
        generation_settings: dict,
    ) -> LLMResponse:
        """Generate a JSON response for a prompt using guided decoding.

        Retries on transient failures with exponential backoff.
        Raises LLMCallError after max_retries exhausted.
        """
        schema = SCHEMAS.get(prompt_name)
        if schema is None:
            raise ValueError(f"No JSON schema defined for prompt '{prompt_name}'")

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            t0 = time.monotonic()
            try:
                raw_text, parsed = await self._call_api(
                    prompt=prompt,
                    schema=schema,
                    settings=generation_settings,
                )
                latency = time.monotonic() - t0
                return LLMResponse(
                    content=parsed,
                    raw_text=raw_text,
                    prompt_name=prompt_name,
                    prompt_hash=prompt_hash,
                    model_id=await self.get_model_id(),
                    generation_settings=generation_settings,
                    latency_seconds=latency,
                    attempt=attempt,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Parse failure — shouldn't happen with guided decoding
                # but handle defensively
                last_exc = e
                logger.warning(
                    "Parse error on attempt %d/%d for prompt '%s': %s",
                    attempt, self.config.max_retries, prompt_name, e,
                )
            except aiohttp.ClientError as e:
                last_exc = e
                logger.warning(
                    "Network error on attempt %d/%d for prompt '%s': %s",
                    attempt, self.config.max_retries, prompt_name, e,
                )

            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        raise LLMCallError(
            f"Failed to get valid response for prompt '{prompt_name}' "
            f"after {self.config.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    async def _call_api(
        self,
        prompt: str,
        schema: dict,
        settings: dict,
    ) -> tuple[str, dict]:
        """Make a single API call to vLLM with guided JSON decoding.

        Tries the guided_json parameter first (vLLM native), then falls back
        to response_format/json_schema (OpenAI-compatible). The working
        parameter is cached after the first successful call to avoid retrying.
        """
        model_id = await self.get_model_id()

        base_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.get("max_tokens", 1024),
            "temperature": settings.get("temperature", 0.3),
            "top_p": settings.get("top_p", 0.9),
        }

        # Build candidates in preference order.
        # Use cached working method if we've already found one.
        #
        # vLLM 0.19 with Gemma 4:
        #   guided_json  → returns content wrapped in markdown code fences
        #                  (```json ... ```) which breaks JSON parsing
        #   response_format/json_schema → returns clean JSON, no fences
        if self._working_structured_output_param:
            candidates = [self._working_structured_output_param]
        else:
            candidates = ["response_format", "guided_json"]

        url = f"{self.config.base_url}/v1/chat/completions"
        last_error: Optional[str] = None

        for param_style in candidates:
            if param_style == "guided_json":
                payload = {**base_payload, "guided_json": schema}
            else:  # response_format / json_schema (OpenAI-compatible)
                payload = {
                    **base_payload,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                }

            async with self._session.post(url, json=payload) as resp:
                body = await resp.text()

                if resp.status != 200:
                    last_error = f"HTTP {resp.status}: {body[:200]}"
                    logger.debug(
                        "Structured output param '%s' failed with %s — trying next",
                        param_style, last_error,
                    )
                    continue

                if not body.strip():
                    last_error = "Empty response body"
                    logger.debug(
                        "Structured output param '%s' returned empty body — trying next",
                        param_style,
                    )
                    continue

                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON response: {e} — body: {body[:200]}"
                    logger.debug(
                        "Structured output param '%s' returned unparseable JSON — trying next",
                        param_style,
                    )
                    continue

                raw_text = data["choices"][0]["message"]["content"]

                if not raw_text or not raw_text.strip():
                    last_error = "Model returned empty content"
                    logger.debug(
                        "Structured output param '%s' — model returned empty content",
                        param_style,
                    )
                    continue

                parsed = json.loads(raw_text)

                # Cache the working parameter for future calls
                if not self._working_structured_output_param:
                    logger.info(
                        "Structured output: using '%s' parameter (cached for session)",
                        param_style,
                    )
                    self._working_structured_output_param = param_style

                return raw_text, parsed

        # All candidates exhausted
        raise aiohttp.ClientError(
            f"No structured output parameter worked. Last error: {last_error}. "
            "Check that your vLLM version supports guided JSON decoding."
        )

    async def generate_batch(
        self,
        prompts: list[tuple[str, str, dict]],
    ) -> list[LLMResponse | Exception]:
        """Generate responses for multiple (prompt, prompt_name, settings) tuples.

        Returns a list of the same length as prompts. Each element is either
        an LLMResponse or an exception if that specific prompt failed.
        Processes all prompts concurrently — vLLM handles GPU scheduling.
        """
        tasks = [
            self.generate(prompt, name, settings)
            for prompt, name, settings in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
