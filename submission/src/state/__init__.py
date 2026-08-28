"""Conversation-state management for the TechJam shopping agent.

Public surface (see README_dialogue_state.md):

    from state.dialogue_state import DialogueStateTracker, DialogueState
"""

from __future__ import annotations

from .dialogue_state import (
    ASK_ATTRIBUTES,
    SLOT_KEYS,
    DialogueState,
    DialogueStateTracker,
    empty_session_profile,
    extract_slots,
    no_preference_attributes,
)

__all__ = [
    "ASK_ATTRIBUTES",
    "SLOT_KEYS",
    "DialogueState",
    "DialogueStateTracker",
    "empty_session_profile",
    "extract_slots",
    "no_preference_attributes",
]
