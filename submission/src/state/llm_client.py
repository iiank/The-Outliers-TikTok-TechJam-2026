"""Schema-constrained LLM transport for the TechJam shopping agent.

The transport behind :mod:`state.llm_extractor`'s joint intent-and-slot call.
It speaks the OpenAI-compatible
``POST {base_url}/chat/completions`` shape, which every free-tier endpoint the
team is likely to use already exposes, so the provider is an environment
variable rather than a code change:

===========================  ================================================
``LLM_BASE_URL``             API root, no trailing ``/chat/completions``.
``LLM_API_KEY``              Bearer token. **Required**, never committed.
``LLM_MODEL``                Model id as that provider spells it.
``LLM_TIMEOUT``              Per-attempt socket timeout, seconds. Default 20.
``LLM_MAX_ATTEMPTS``         Total tries including the first. Default 2.
``LLM_MAX_TOKENS``           Completion cap. Default 2048.
``LLM_USER_AGENT``           Overrides the default agent string.
``LLM_REASONING_EFFORT``     Sent only if set, e.g. ``low``. See caveat below.
===========================  ================================================

``LLM_REASONING_EFFORT=low`` cuts completion tokens ~60% on gpt-oss but was
measured emitting ``"< =120"`` for a budget instead of ``"<=120"``. ``budget_bounds``
tolerates that now, but treat low effort as a throughput lever to verify, not a
free win.

Verified shapes for the defaults, and the drop-in alternatives::

    # Groq (the built-in default). Free tier is 8000 tokens/min, 1000 req/min.
    LLM_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=openai/gpt-oss-120b

    # Google Gemini, OpenAI-compatibility endpoint
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
    LLM_MODEL=gemini-2.0-flash

Two invariants the callers depend on:

* **Constrained output only.** Every request sends
  ``response_format={"type": "json_schema", ..., "strict": True}``, and the
  reply is read with :func:`json.loads` and nothing else. There is no prose
  salvage path, no fence stripping, no key repair. A provider that ignores
  ``response_format`` therefore fails cleanly instead of feeding half-parsed
  guesses into the dialogue state. Keys outside the schema are dropped even if
  the provider lets them through, so a caller can trust the key set absolutely.
* **Never raises.** Timeouts, HTTP errors, malformed JSON, and missing
  credentials all come back as ``LLMResult(data=None, error=...)``. The
  competition harness counts an exception as a miss
  (docs/competition_specification.md), so failure has to be a value.

Token accounting for the response ``usage`` field lives here too, because both
callers return schema-constrained payloads with no room for it. Every call adds
to a process-wide meter; the agent drains it once per turn:

    from state.llm_client import drain_usage
    ...
    return {..., "usage": drain_usage()}

Standard library only, matching the rest of the submission.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "LLMResult",
    "call_json",
    "credentials_present",
    "drain_usage",
    "usage_totals",
    "reset_usage",
    "string_array",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_ATTEMPTS = 2

#: Completion ceiling. Generous because the default model is a *reasoning*
#: model: gpt-oss emits its chain of thought into the completion before the JSON
#: object, so a ceiling sized for the JSON alone truncates mid-reasoning and the
#: provider rejects the turn with ``json_validate_failed`` ("max completion
#: tokens reached before generating a valid document"). 512 was enough for slot
#: extraction and failed ~75% of intent classifications.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_USER_AGENT = "outliers-techjam-agent/1.0 (+python-urllib)"

#: Transient statuses worth one more try. 400/401/404 are configuration bugs and
#: retrying them only burns the clock.
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Backoff before a retry, seconds. Short: the harness is waiting on us.
_RETRY_BACKOFF = 0.8


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    """One completed (or failed) call.

    Attributes:
        data: The parsed object, already filtered to the schema's declared keys.
            ``None`` on any failure, which is the signal callers branch on.
        prompt_tokens: Reported prompt tokens, 0 when the provider omits them.
        completion_tokens: Reported completion tokens, 0 when omitted.
        error: Short machine-readable failure tag, ``None`` on success. One of
            ``no_credentials``, ``http_<status>``, ``network``, ``timeout``,
            ``bad_json``, ``schema_mismatch``, ``empty_response``, or
            ``unexpected``.
    """

    data: Optional[Dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.data is not None

    def usage(self) -> Dict[str, int]:
        """This call's tokens in the contract's ``usage`` shape."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --------------------------------------------------------------------------
# Usage meter
# --------------------------------------------------------------------------


class _UsageMeter:
    """Process-wide token counter, safe to touch from more than one thread."""

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

    def totals(self) -> Dict[str, int]:
        with self._lock:
            return {"prompt_tokens": self._prompt, "completion_tokens": self._completion}


_METER = _UsageMeter()


def drain_usage() -> Dict[str, int]:
    """Return tokens spent since the last drain, and reset the counter.

    Call once per turn, right before building the response, so ``usage`` reports
    that turn's cost. Both non-negative integers, so it satisfies the
    ``usage`` sub-schema in docs/agent_api_contract.json as-is.
    """
    return _METER.drain()


def usage_totals() -> Dict[str, int]:
    """Peek at the undrained counter without resetting it."""
    return _METER.totals()


def reset_usage() -> None:
    """Zero the counter, e.g. in ``Agent.reset`` or between test cases."""
    _METER.drain()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

_WARNED: set = set()
_WARN_LOCK = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    """Log ``message`` the first time ``key`` is seen.

    A missing key would otherwise be invisible: every turn just returns ``{}``
    and the run looks like the old no-op stub.
    """
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


def credentials_present() -> bool:
    """True when ``LLM_API_KEY`` is set to something non-empty.

    Cheap enough to call per turn, and lets a caller skip building a prompt it
    cannot send.
    """
    return bool(os.environ.get("LLM_API_KEY", "").strip())


# --------------------------------------------------------------------------
# Schema helper
# --------------------------------------------------------------------------


def string_array(enum: Optional[Iterable[str]] = None, description: str = "") -> Dict[str, Any]:
    """A JSON-schema fragment for an array of strings.

    Both callers want the same fragment repeatedly, so build it in one place.

    Args:
        enum: Restrict items to this set, for a field that names attributes
            rather than free values.
        description: Field-level instruction. Models follow these more reliably
            than the same sentence buried in the system prompt.
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
    """Ask the model for one object matching ``schema``. Never raises.

    Args:
        system_prompt: The task rules. Kept identical across turns so a
            provider with prompt caching can reuse it.
        user_payload: The per-turn content. JSON is a good choice: it keeps the
            utterance visually separate from the state context, which makes
            prompt injection through a customer message much less likely to
            read as an instruction.
        schema: A JSON Schema **object** with ``properties``. Sent as-is under
            ``strict: true``, so it needs ``additionalProperties: false`` and
            every property listed in ``required``.
        schema_name: Identifier for the schema; some providers require it.
        max_tokens: Completion cap. Defaults to ``LLM_MAX_TOKENS``.
        temperature: 0.0 by default. Extraction and classification should be
            reproducible across runs; the evaluator is deterministic and a
            wandering extractor makes local scores impossible to compare.

    Returns:
        An :class:`LLMResult`. On success ``data`` holds only keys declared in
        ``schema["properties"]``. Tokens are added to the shared meter whether
        or not parsing succeeded, because a failed call still costs money.
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
    # Reasoning models bill their chain of thought as completion tokens, which
    # dominates cost and eats a per-minute token quota. Where a provider exposes
    # a depth control, turning it down is the cheapest lever available. Sent only
    # when set, since providers that do not know the field reject it.
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

    last_error = "unexpected"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            return _parse_envelope(raw, schema)
        except urllib.error.HTTPError as error:
            last_error = f"http_{error.code}"
            # The body usually names the real problem (unsupported
            # response_format, unknown model). Worth one log line.
            detail = ""
            try:
                detail = error.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # pragma: no cover - best effort only
                pass
            LOGGER.warning("LLM HTTP %s on attempt %s/%s: %s", error.code, attempt, attempts, detail)
            if error.code not in _RETRY_STATUS:
                break
        except urllib.error.URLError as error:
            # URLError wraps socket.timeout, DNS failure, refused connection.
            last_error = "timeout" if isinstance(error.reason, TimeoutError) else "network"
            LOGGER.warning("LLM %s on attempt %s/%s: %s", last_error, attempt, attempts, error.reason)
        except TimeoutError:
            last_error = "timeout"
            LOGGER.warning("LLM timeout on attempt %s/%s", attempt, attempts)
        except Exception as error:  # pragma: no cover - defensive
            last_error = "unexpected"
            LOGGER.warning("LLM unexpected failure on attempt %s/%s: %r", attempt, attempts, error)
            break
        if attempt < attempts:
            time.sleep(_RETRY_BACKOFF * attempt)

    return LLMResult(error=last_error)


def _parse_envelope(raw: str, schema: Dict[str, Any]) -> LLMResult:
    """Turn a chat-completions response body into an :class:`LLMResult`.

    Records tokens before validating content, so a reply that parses badly is
    still billed in the report.
    """
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
        # A provider that ignored response_format lands here. Deliberately not
        # salvaged: guessing at prose would put invented constraints into state.
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
    """Pull token counts out of a provider's ``usage`` block.

    Most OpenAI-compatible endpoints use ``prompt_tokens``/``completion_tokens``;
    a few report ``input_tokens``/``output_tokens``. Missing counts are 0 rather
    than an error, since the contract only needs non-negative integers.
    """
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
    """Extract the first choice's message text, or ``None``.

    Tolerates the list-of-parts content shape some gateways emit, since that is
    a transport detail rather than a model guess.
    """
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
