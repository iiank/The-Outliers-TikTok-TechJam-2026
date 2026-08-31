"""Schema-constrained LLM client for the TechJam shopping agent.

Used by ``state.llm_extractor`` for joint intent and slot extraction. Calls
OpenAI-compatible ``POST {base_url}/chat/completions`` endpoints so providers
can be changed through environment variables without modifying code.

===========================  ================================================
``LLM_BASE_URL``             API root, no trailing ``/chat/completions``.
``LLM_API_KEY``              Bearer token. **Required**, never committed.
``LLM_MODEL``                Model id as that provider spells it.
``LLM_TIMEOUT``              Per-attempt socket timeout, seconds. Default 20.
``LLM_MAX_ATTEMPTS``         Total tries including the first. Default 2.
``LLM_MIN_INTERVAL``         Seconds enforced between call starts, process-wide.
                              Default 0 (off). Set this against a tight TPM quota
                              (e.g. Groq free tier) so requests never outrun the
                              window in the first place.
``LLM_MAX_TOKENS``           Completion cap. Default 2048.
``LLM_USER_AGENT``           Overrides the default agent string.
``LLM_REASONING_EFFORT``     Sent only if set, e.g. ``low``. See caveat below.
===========================  ================================================

Low reasoning effort can reduce completion-token usage but may degrade
structured values. Verify extraction quality before enabling it.

Example providers::

    # Groq
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=openai/gpt-oss-120b

    # Google Gemini OpenAI-compatible endpoint
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    LLM_MODEL=gemini-2.0-flash

Caller guarantees:

* Responses must match the requested JSON schema. Invalid or unconstrained
  output fails rather than being repaired or inferred.
* Transport and parsing failures return ``LLMResult(data=None, error=...)``
  instead of raising, because evaluator exceptions count as misses.
* Undeclared response keys are removed before data reaches dialogue state.

Token usage is accumulated process-wide and drained by the agent once per turn::

    from state.llm_client import drain_usage
    ...
    return {..., "usage": drain_usage()}

Uses only the Python standard library.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "LLMResult",
    "call_json",
    "drain_usage",
    "reset_usage",
    "string_array",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_ATTEMPTS = 2

# Reasoning models may consume completion tokens before producing the JSON
# payload, so the limit must leave enough room for both reasoning and output.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_USER_AGENT = "outliers-techjam-agent/1.0 (+python-urllib)"

# Retry only transient failures
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Fallback retry delay when the provider does not specify one.
_RETRY_BACKOFF = 0.8

# Cap provider-suggested delays to avoid stalling the evaluator.
_MAX_SUGGESTED_WAIT = 30.0

# Parse cooldowns such as "Please try again in 19.035s" from 429 responses.
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)?", re.IGNORECASE)


def _retry_after_seconds(detail: str, headers: Any) -> Optional[float]:
    """Return the provider's suggested retry delay, if available."""
    
    header_value = None
    try:
        header_value = headers.get("Retry-After") if headers is not None else None
    except AttributeError:
        header_value = None
    if header_value:
        try:
            return min(_MAX_SUGGESTED_WAIT, max(0.0, float(header_value)))
        except (TypeError, ValueError):
            pass
    match = _RETRY_AFTER_RE.search(detail or "")
    if not match:
        return None
    value = float(match.group(1))
    if (match.group(2) or "").lower() == "ms":
        value /= 1000.0
    return min(_MAX_SUGGESTED_WAIT, max(0.0, value))


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    """Result of one LLM request.

    Attributes:
        data: Schema-filtered response object, or ``None`` on failure.
        prompt_tokens: Reported prompt tokens, or 0 when unavailable.
        completion_tokens: Reported completion tokens, or 0 when unavailable.
        error: Machine-readable failure code, or ``None`` on success.
    """

    data: Optional[Dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.data is not None

    def usage(self) -> Dict[str, int]:
        """Return this request's token usage in the agent contract format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --------------------------------------------------------------------------
# Usage meter
# --------------------------------------------------------------------------


class _UsageMeter:
    """Thread-safe process-wide token counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt = 0
        self._completion = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._prompt += max(0, int(prompt_tokens))
            self._completion += max(0, int(completion_tokens))

    def drain(self) -> Dict[str, int]:
        with self._lock:
            totals = {"prompt_tokens": self._prompt, "completion_tokens": self._completion}
            self._prompt = 0
            self._completion = 0
        return totals


_METER = _UsageMeter()


class _Throttle:
    """Enforce a process-wide minimum interval between request starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self, min_interval: float) -> None:
        if min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + min_interval
        if delay > 0:
            time.sleep(delay)


_THROTTLE = _Throttle()


def drain_usage() -> Dict[str, int]:
    """Return accumulated token usage and reset the counter."""
    return _METER.drain()


def reset_usage() -> None:
    """Reset accumulated token usage."""
    _METER.drain()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

_WARNED: set = set()
_WARN_LOCK = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    """Log a warning only on its first occurrence."""
    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    LOGGER.warning(message)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Schema helper
# --------------------------------------------------------------------------


def string_array(enum: Optional[Iterable[str]] = None, description: str = "") -> Dict[str, Any]:
    """Build a JSON Schema fragment for an array of strings.

    Args:
        enum: Optional allowed values for each array item.
        description: Optional field-level schema description.
    """
    items: Dict[str, Any] = {"type": "string"}
    if enum is not None:
        items["enum"] = list(enum)
    fragment: Dict[str, Any] = {"type": "array", "items": items}
    if description:
        fragment["description"] = description
    return fragment


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


def call_json(
    system_prompt: str,
    user_payload: str,
    schema: Dict[str, Any],
    schema_name: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
) -> LLMResult:
    """Request one schema-constrained JSON object without raising.

    Args:
        system_prompt: Stable task instructions sent as the system message.
        user_payload: Per-turn input sent as the user message.
        schema: Strict JSON Schema describing the expected object.
        schema_name: Provider-facing identifier for the schema.
        max_tokens: Optional completion-token limit.
        temperature: Sampling temperature; defaults to deterministic extraction.

    Returns:
        ``LLMResult`` containing schema-filtered data or a failure code. Token
        usage is recorded even when response parsing fails.
    """
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        _warn_once(
            "no_credentials",
            "LLM_API_KEY is not set: state.llm_client returns no data, so slot "
            "extraction and intent detection both degrade to 'no information' "
            "every turn. Export LLM_API_KEY (and optionally LLM_BASE_URL / "
            "LLM_MODEL) to enable them.",
        )
        return LLMResult(error="no_credentials")

    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = _env_float("LLM_TIMEOUT", DEFAULT_TIMEOUT)
    attempts = max(1, _env_int("LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    limit = max_tokens if max_tokens is not None else _env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": limit,
    }
    
    effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
    if effort:
        payload["reasoning_effort"] = effort

    body = json.dumps(
        {
            **payload,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": os.environ.get("LLM_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        },
    )

    _THROTTLE.wait(_env_float("LLM_MIN_INTERVAL", 0.0))

    last_error = "unexpected"
    for attempt in range(1, attempts + 1):
        wait_seconds = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            return _parse_envelope(raw, schema)
        except urllib.error.HTTPError as error:
            last_error = f"http_{error.code}"
            detail = ""
            try:
                detail = error.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                pass
            LOGGER.warning("LLM HTTP %s on attempt %s/%s: %s", error.code, attempt, attempts, detail)
            if error.code not in _RETRY_STATUS:
                break
            wait_seconds = _retry_after_seconds(detail, getattr(error, "headers", None))
        except urllib.error.URLError as error:
            last_error = "timeout" if isinstance(error.reason, TimeoutError) else "network"
            LOGGER.warning("LLM %s on attempt %s/%s: %s", last_error, attempt, attempts, error.reason)
        except TimeoutError:
            last_error = "timeout"
            LOGGER.warning("LLM timeout on attempt %s/%s", attempt, attempts)
        except Exception as error:
            last_error = "unexpected"
            LOGGER.warning("LLM unexpected failure on attempt %s/%s: %r", attempt, attempts, error)
            break
        if attempt < attempts:
            time.sleep(wait_seconds if wait_seconds is not None else _RETRY_BACKOFF * attempt)

    return LLMResult(error=last_error)


def _parse_envelope(raw: str, schema: Dict[str, Any]) -> LLMResult:
    """Parse and validate an OpenAI-compatible response envelope."""
    
    try:
        envelope = json.loads(raw)
    except (ValueError, TypeError):
        return LLMResult(error="bad_json")
    if not isinstance(envelope, dict):
        return LLMResult(error="bad_json")

    prompt_tokens, completion_tokens = _read_usage(envelope.get("usage"))
    _METER.add(prompt_tokens, completion_tokens)

    content = _read_content(envelope)
    if content is None:
        return LLMResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error="empty_response",
        )

    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        LOGGER.warning("LLM returned non-JSON content; is response_format json_schema supported?")
        return LLMResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error="bad_json",
        )

    if not isinstance(payload, dict):
        return LLMResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error="schema_mismatch",
        )

    allowed = set(schema.get("properties") or {})
    filtered = {key: value for key, value in payload.items() if key in allowed}
    dropped = set(payload) - allowed
    if dropped:
        # Belt and braces: `strict` should make this impossible, but a key the
        # caller never declared must not reach the dialogue state.
        LOGGER.debug("dropped undeclared keys from LLM output: %s", sorted(dropped))

    return LLMResult(
        data=filtered,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _read_usage(usage: Any) -> tuple:
    """Read token counts from common OpenAI-compatible usage formats."""
    if not isinstance(usage, dict):
        return 0, 0

    def pick(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)):
                return max(0, int(value))
        return 0

    return pick("prompt_tokens", "input_tokens"), pick("completion_tokens", "output_tokens")


def _read_content(envelope: Dict[str, Any]) -> Optional[str]:
    """Return the first choice's message text, if present."""
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts: List[str] = [
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        joined = "".join(parts).strip()
        return joined or None
    return None
