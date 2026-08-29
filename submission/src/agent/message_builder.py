"""Natural-language response generation.

The evaluator scores ``ask_attribute`` and ``recommendations``; ``message``
content is never inspected by the scripted customer-simulator (see the plan
doc). It still matters for the demo and any live/interactive use, so the
default path calls an LLM to phrase it naturally from ``ask_attribute``,
``mode``, and the session state.

``TemplateMessageBuilder`` is a complete, tested, offline alternative --
swap it in with ``Agent(message_builder=TemplateMessageBuilder())``. As with
the intent classifier, there's no automatic fallback: an ``LLMMessageBuilder``
API failure raises, same as any other ``respond()`` exception.

CROSS-BOUNDARY POINTS IN THIS FILE:

* ``BOUNDARY(state)`` -- the ``DialogueState`` import (type only) and, in
  ``LLMMessageBuilder.build()``, the ``state.session_profile`` read: reads
  data owned by the state teammate's ``DialogueState`` class.
* ``BOUNDARY(external: Anthropic API)`` -- ``LLMMessageBuilder`` calls out to
  the Anthropic Messages API over the network. Same network/cost visibility
  note as ``intent_classifier.py``.
* Deliberately NOT a boundary: ``candidates`` (the ``parent_asin`` list from
  ``Searcher.search()``) is treated as opaque IDs here -- this module never
  looks up catalog/product data for them. See the note on ``LLMMessageBuilder``
  below for why.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple

from state.dialogue_state import DialogueState  # BOUNDARY(state): type import only

__all__ = ["MessageBuilder", "TemplateMessageBuilder", "LLMMessageBuilder"]


class MessageBuilder(Protocol):
    def build(
        self,
        ask_attribute: Optional[str],
        mode: str,
        state: DialogueState,
        candidates: List[str],
    ) -> Tuple[str, Dict[str, int]]:
        """Return ``(message, usage)``. ``usage`` is ``{}`` when no LLM call was made."""
        ...


_TEMPLATES: Dict[str, str] = {
    "category": "What kind of item are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Any color preference?",
    "size": "What size do you need?",
    "style": "Any particular style in mind?",
    "brand": "Any brand you prefer?",
    "budget": "What's your budget range?",
    "feature": "Any specific feature that matters to you?",
    "use_case": "What will you mainly use this for?",
    "other": "Is there anything else I should know to narrow this down?",
}


class TemplateMessageBuilder:
    """Deterministic, zero-dependency phrasing keyed by ``ask_attribute``."""

    def build(
        self,
        ask_attribute: Optional[str],
        mode: str,
        state: DialogueState,
        candidates: List[str],
    ) -> Tuple[str, Dict[str, int]]:
        if ask_attribute and ask_attribute in _TEMPLATES:
            return _TEMPLATES[ask_attribute], {}
        return "Here are the closest matches I found.", {}


class LLMMessageBuilder:
    # LLM phrasing from ``ask_attribute``, ``mode``, and the session state.

    def __init__(self, client: Optional[object] = None, model: str = "claude-haiku-4-5") -> None:
        self._client = client
        self.model = model

    def _get_client(self):
        if self._client is None:
            import anthropic  # BOUNDARY(external: Anthropic API)

            self._client = anthropic.Anthropic()
        return self._client

    def build(
        self,
        ask_attribute: Optional[str],
        mode: str,
        state: DialogueState,
        candidates: List[str],
    ) -> Tuple[str, Dict[str, int]]:
        client = self._get_client()
        # "rejected" is excluded: those are values the customer declined, not
        # known preferences -- including it here previously presented a
        # rejected value (and the raw "no_preference:<attr>" marker string)
        # to the model as if it were a confirmed fact.
        known = ", ".join(
            f"{key}={values}" for key, values in state.session_profile.items()
            if key != "rejected" and values
        )  # BOUNDARY(state): reads DialogueState.session_profile

        if ask_attribute:
            instruction = f"Ask the customer about their {ask_attribute} preference in one short, natural sentence."
        else:
            instruction = (
                "You have nothing further worth asking -- present the recommendations confidently "
                "in one short, natural sentence. Don't ask a question."
            )

        context = f"Known so far this session: {known}." if known else "Nothing disclosed yet this session."
        response = client.messages.create(  # BOUNDARY(external: Anthropic API) -- network call
            model=self.model,
            max_tokens=120,
            system=(
                "You are a concise, friendly shopping assistant. Write exactly ONE short sentence "
                "for the customer, in plain conversational language. Never mention 'attributes', "
                "internal reasoning, or that you're an AI."
            ),
            messages=[{"role": "user", "content": f"{context}\n{instruction}"}],
        )
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        text = next((block.text for block in response.content if block.type == "text"), "").strip()
        if not text:
            text = _TEMPLATES.get(ask_attribute, "Here are the closest matches I found.") if ask_attribute else "Here are the closest matches I found."
        return text, usage
