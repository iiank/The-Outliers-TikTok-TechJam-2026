"""``Agent`` for the TechJam evaluator (``docs/agent_api_contract.json``).

Coordinates state tracking, search, and response phrasing through narrow
interfaces so each component can be replaced independently.

* State tracking is handled by ``state.dialogue_state.DialogueStateTracker``.
  Intent is populated by the joint intent-and-slot extractor, so this module
  does not perform a separate intent-classification call.
* Search handles retrieval and selection of the next attribute to ask about.
* Message phrasing is handled by ``agent.message_builder``.

Integration boundary:

* ``state.dialogue_state`` provides ``DialogueState`` and
  ``DialogueStateTracker`` and is accessed through ``StateTracker`` and
  ``self.state_tracker``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from agent.message_builder import LLMMessageBuilder, MessageBuilder
from state.dialogue_state import DialogueState, DialogueStateTracker
from state.llm_client import drain_usage, reset_usage
from search import search

__all__ = ["Agent", "StateTracker"]


class StateTracker(Protocol):
    """Mirror the public interface of the state module's tracker."""

    def reset(self, session_id: str, user_profile: Optional[Dict[str, Any]] = None) -> DialogueState: ...
    def get_state(self, session_id: str) -> DialogueState: ...
    def update(self, user_message: str, current_state: DialogueState, turn: Optional[int] = None) -> DialogueState: ...
    def record_recommendations(self, state: DialogueState, parent_asins: List[str]) -> DialogueState: ...
    def record_ask(self, state: DialogueState, ask_attribute: Optional[str]) -> DialogueState: ...


def _combined_usage(*parts: Dict[str, int]) -> Dict[str, int]:
    prompt_tokens = sum(part.get("prompt_tokens", 0) for part in parts if part)
    completion_tokens = sum(part.get("completion_tokens", 0) for part in parts if part)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class Agent:
    """Evaluator-facing agent with swappable state and message components."""

    def __init__(
        self,
        state_tracker: Optional[StateTracker] = None,
        message_builder: Optional[MessageBuilder] = None,
    ) -> None:
        self.state_tracker: StateTracker = state_tracker or DialogueStateTracker()
        self.message_builder: MessageBuilder = message_builder or LLMMessageBuilder()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.state_tracker.reset(session_id, user_profile)
        reset_usage()

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        prior_state = self.state_tracker.get_state(session_id)
        state = self.state_tracker.update(user_message, prior_state, turn)
        mode = state.intent or "browsing"

        candidates, ask_attribute, _diagnostics = search(state, user_message)
        candidates = candidates[:top_k]
        self.state_tracker.record_recommendations(state, candidates)
        self.state_tracker.record_ask(state, ask_attribute)

        message, message_usage = self.message_builder.build(ask_attribute, mode, state, candidates)
        extractor_usage = drain_usage()

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": parent_asin} for parent_asin in candidates],
            "usage": _combined_usage(extractor_usage, message_usage),
        }

    def respond_chat(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        '''Return the response contract used by the frontend's interactive page.'''
        prior_state = self.state_tracker.get_state(session_id)
        state = self.state_tracker.update(user_message, prior_state, turn)
        mode = state.intent or "browsing"

        candidates, ask_attribute, diagnostics = search(state, user_message)
        candidates = candidates[:top_k]
        self.state_tracker.record_recommendations(state, candidates)

        message, _message_usage = self.message_builder.build(ask_attribute, mode, state, candidates)

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": parent_asin} for parent_asin in candidates],
            "diagnostics": diagnostics,
        }