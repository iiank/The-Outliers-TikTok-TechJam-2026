"""Conformant ``Agent`` for the TechJam evaluator (``docs/agent_api_contract.json``).

Ties together three pieces behind narrow interfaces, so each can be
developed and swapped independently:

* state tracking (``state.dialogue_state.DialogueStateTracker`` -- owned
  elsewhere, treated here as a black box);
* intent classification, buy vs. browse (``agent.intent_classifier``);
* search: retrieval plus the attribute worth asking about next -- owned
  elsewhere (the ``Searcher`` protocol below), treated here as a black box;
* message phrasing (``agent.message_builder``).

CROSS-BOUNDARY POINTS IN THIS FILE (everywhere this module reads from or
calls into code owned by someone else -- marked inline as ``# BOUNDARY(...)``):

* ``BOUNDARY(state)``  -- ``state.dialogue_state`` (the state teammate's
  module): the ``DialogueState``/``DialogueStateTracker`` import, the
  ``StateTracker`` protocol below (mirrors their public methods), the
  default construction of ``DialogueStateTracker()``, and every
  ``self.state_tracker.*`` call in ``respond()``.
* ``BOUNDARY(search)`` -- the retrieval teammate's black box: the
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from agent.intent_classifier import IntentClassifier, LLMIntentClassifier
from agent.message_builder import LLMMessageBuilder, MessageBuilder
from state.dialogue_state import DialogueState, DialogueStateTracker  # BOUNDARY(state): state teammate's module

__all__ = ["Agent", "StateTracker", "Searcher"]


class StateTracker(Protocol):
    """BOUNDARY(state): structural contract satisfied by
    ``state.dialogue_state.DialogueStateTracker`` -- owned by the state
    teammate. Mirrors their public methods; not implemented here."""

    def reset(self, session_id: str, user_profile: Optional[Dict[str, Any]] = None) -> DialogueState: ...
    def get_state(self, session_id: str) -> DialogueState: ...
    def update(self, user_message: str, current_state: DialogueState, turn: Optional[int] = None) -> DialogueState: ...
    def record_recommendations(self, state: DialogueState, parent_asins: List[str]) -> DialogueState: ...


class Searcher(Protocol):
    """BOUNDARY(search): structural contract for the retrieval black box --
    owned by the retrieval teammate, not implemented in this module. See the
    plan doc for the proposed shape and the open questions to align on with
    them."""

    def search(self, state: DialogueState, mode: str, top_k: int) -> Tuple[List[str], Optional[str]]:
        """Return up to ``top_k`` ``parent_asin`` values (best first) and the
        single ``ask_attribute`` worth asking about next, or ``None`` if
        nothing unresolved is left to ask."""
        ...


def _combined_usage(*parts: Dict[str, int]) -> Dict[str, int]:
    prompt_tokens = sum(part.get("prompt_tokens", 0) for part in parts if part)
    completion_tokens = sum(part.get("completion_tokens", 0) for part in parts if part)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class Agent:
    """Conformant ``reset``/``respond`` agent -- entry point for the evaluator.

    ``searcher`` has no default: it's the retrieval teammate's black box, not
    something this module builds a stand-in for. ``state_tracker``,
    ``intent_classifier``, and ``message_builder`` default to the real
    dialogue-state tracker and this module's own LLM-backed implementations,
    but are all independently swappable (fakes for tests, the offline
    regex/template alternatives).
    """

    def __init__(
        self,
        searcher: Searcher,
        state_tracker: Optional[StateTracker] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        message_builder: Optional[MessageBuilder] = None,
    ) -> None:
        self.state_tracker: StateTracker = state_tracker or DialogueStateTracker()  # BOUNDARY(state): default-constructs the state teammate's tracker
        self.searcher: Searcher = searcher  # BOUNDARY(search): required, no default -- caller must supply the retrieval teammate's implementation
        self.intent_classifier: IntentClassifier = intent_classifier or LLMIntentClassifier()
        self.message_builder: MessageBuilder = message_builder or LLMMessageBuilder()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.state_tracker.reset(session_id, user_profile)  # BOUNDARY(state)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        prior_state = self.state_tracker.get_state(session_id)  # BOUNDARY(state): reads the state teammate's DialogueState
        mode = self.intent_classifier.classify(user_message, prior_state)
        state = self.state_tracker.update(user_message, prior_state, turn)  # BOUNDARY(state): folds this turn's message into their state model

        candidates, ask_attribute = self.searcher.search(state, mode, top_k)  # BOUNDARY(search): the retrieval teammate's black box
        candidates = candidates[:top_k]
        self.state_tracker.record_recommendations(state, candidates)  # BOUNDARY(state)

        message, message_usage = self.message_builder.build(ask_attribute, mode, state, candidates)
        classify_usage = getattr(self.intent_classifier, "last_usage", {}) or {}

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": parent_asin} for parent_asin in candidates],
            "usage": _combined_usage(classify_usage, message_usage),
        }
