"""An LLM client that records every call, for the trace pane.

Wraps any object satisfying the ``LLMClient`` protocol and forwards to
it, keeping the exact system prompt, user text and response for each
call. Because it sits at the client boundary rather than inside one
function, it captures *every* LLM node in the pipeline — scope gate,
clarifier, frame matcher, router critic, plan reviewer, followup finder,
synthesizer — with no pipeline changes at all.

Node attribution comes from the response schema's title, which is
distinct per node (``_SynthesisLLMOutput``, ``ScopeGateOutput``, …).
That's a heuristic, not a contract: an unrecognized schema is recorded
as "unknown", which is honest and still shows the prompt and response.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional


def _schema_label(schema: Any) -> str:
    """Best-effort node name from the response schema."""
    if isinstance(schema, dict):
        for key in ("title", "$id", "name"):
            val = schema.get(key)
            if isinstance(val, str) and val:
                return val.lstrip("_")
    return "unknown"


class RecordingLLMClient:
    """Transparent recording proxy over an LLMClient.

    Forwards unknown attributes to the wrapped client, so usage counters
    and any provider-specific methods keep working.
    """

    def __init__(self, inner: Any, max_calls: int = 100):
        self._inner = inner
        self._calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._max_calls = max_calls

    # --- recording ---------------------------------------------------

    def extract(
        self,
        *,
        system_prompt: str,
        user_text: str,
        schema: Any = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        started = time.time()
        error: Optional[str] = None
        response: Any = None
        try:
            response = self._inner.extract(
                system_prompt=system_prompt,
                user_text=user_text,
                schema=schema,
                temperature=temperature,
                **kwargs,
            )
            return response
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._record({
                "node": _schema_label(schema),
                "seconds": round(time.time() - started, 3),
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_text": user_text,
                "response": response,
                "error": error,
            })

    def _record(self, call: dict[str, Any]) -> None:
        with self._lock:
            # Bounded so a runaway loop can't exhaust memory. Keep the
            # most recent calls — the synthesizer runs last and is the
            # one being iterated on.
            self._calls.append(call)
            if len(self._calls) > self._max_calls:
                del self._calls[0]

    # --- access ------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._calls)

    def last_for(self, node: str) -> Optional[dict[str, Any]]:
        """Most recent call whose node label contains ``node``."""
        needle = node.lower()
        with self._lock:
            for call in reversed(self._calls):
                if needle in call["node"].lower():
                    return call
        return None

    def synthesis_call(self) -> Optional[dict[str, Any]]:
        return self.last_for("synthesis")

    # --- transparency ------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this class doesn't define.
        return getattr(self._inner, name)
