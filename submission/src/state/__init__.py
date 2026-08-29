"""Conversation-state management for the TechJam shopping agent.

Public surface (see README_dialogue_state.md):

    from state.dialogue_state import DialogueStateTracker, DialogueState
    from state.llm_extractor import extract_slots
    from state.context_distiller import distill
    from state.llm_client import drain_usage

``dialogue_state`` is import-safe and never touches the network. The other
three modules read ``LLM_*`` environment variables at call time, not import
time, so importing this package still costs nothing.
"""

from __future__ import annotations

from .context_distiller import distill
from .dialogue_state import (
    ASK_ATTRIBUTES,
    INTENT_LABELS,
    SLOT_KEYS,
    DialogueState,
    DialogueStateTracker,
    budget_bounds,
    empty_session_profile,
    extract_slots,
    no_preference_attributes,
)
from .llm_client import drain_usage

__all__ = [
    "ASK_ATTRIBUTES",
    "INTENT_LABELS",
    "SLOT_KEYS",
    "DialogueState",
    "DialogueStateTracker",
    "budget_bounds",
    "distill",
    "drain_usage",
    "empty_session_profile",
    "extract_slots",
    "no_preference_attributes",
]
