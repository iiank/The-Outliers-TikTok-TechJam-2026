"""Conversation-state management for the TechJam shopping agent.

Public surface (see README_dialogue_state.md):

    from state.dialogue_state import DialogueStateTracker, DialogueState
    from state.llm_extractor import extract_slots
    from state.context_distiller import distill
    from state.intent_classifier import classify_intent
    from state.llm_client import drain_usage

``dialogue_state`` is import-safe and never touches the network. The other four
modules read ``LLM_*`` environment variables at call time, not import time, so
importing this package still costs nothing.
"""

from __future__ import annotations

from .dialogue_state import (
    ASK_ATTRIBUTES,
    SLOT_KEYS,
    DialogueState,
    DialogueStateTracker,
    budget_bounds,
    empty_session_profile,
    extract_slots,
    no_preference_attributes,
)

__all__ = [
    "ASK_ATTRIBUTES",
    "SLOT_KEYS",
    "DialogueState",
    "DialogueStateTracker",
    "budget_bounds",
    "empty_session_profile",
    "extract_slots",
    "no_preference_attributes",
]
