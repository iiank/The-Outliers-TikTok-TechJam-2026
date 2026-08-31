from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_stale_agent = sys.modules.get("agent")
if _stale_agent is not None and not hasattr(_stale_agent, "__path__"):
    del sys.modules["agent"]

from agent.intent_classifier import (
    IntentClassifier,
    LLMIntentClassifier,
    RegexIntentClassifier,
)
from agent.message_builder import (
    LLMMessageBuilder,
    MessageBuilder,
    TemplateMessageBuilder,
)
from agent.shopping_agent import Agent as _ShoppingAgent
from state.dialogue_state import DialogueStateTracker
from state.regex_extractor import extract_slots

__all__ = ["Agent"]


def _has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _default_intent_classifier() -> IntentClassifier:
    return LLMIntentClassifier() if _has_anthropic_key() else RegexIntentClassifier()


def _default_message_builder() -> MessageBuilder:
    return LLMMessageBuilder() if _has_anthropic_key() else TemplateMessageBuilder()


class Agent(_ShoppingAgent):
    """The graded entrypoint. See module docstring for the default-wiring
    rationale; every piece remains swappable via the parent class's
    constructor if you want to override a default explicitly."""

    def __init__(self) -> None:
        super().__init__(
            state_tracker=DialogueStateTracker(extractor=extract_slots),
            intent_classifier=_default_intent_classifier(),
            message_builder=_default_message_builder(),
        )
