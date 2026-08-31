"""Generate customer-facing responses.

The default builder uses an LLM to phrase responses from the requested
attribute and session state.

Use ``TemplateMessageBuilder`` for deterministic offline output:

    Agent(message_builder=TemplateMessageBuilder())

LLM failures propagate through ``respond()`` without an automatic fallback.

Integration boundaries:

* ``DialogueState`` is imported for typing, and ``LLMMessageBuilder.build()``
  reads ``state.session_profile``.
* ``LLMMessageBuilder`` calls the Anthropic Messages API.
* ``candidates`` contains opaque ``parent_asin`` values from
  ``Searcher.search()``; this module does not inspect product data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple

from state.dialogue_state import DialogueState

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
    """Build deterministic responses without external dependencies.
    Currently being used in CRIS.
    """

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
    """Generate natural phrasing from the requested attribute and session state.
    Currently not being used in CRIS.
    """

    def __init__(self, client: Optional[object] = None, model: str = "claude-haiku-4-5") -> None:
        self._client = client
        self.model = model

    def _get_client(self):
        if self._client is None:
            import anthropic

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
        known = ", ".join(
            f"{key}={values}" for key, values in state.session_profile.items()
            if key != "rejected" and values
        )

        if ask_attribute:
            instruction = f"Ask the customer about their {ask_attribute} preference in one short, natural sentence."
        else:
            instruction = (
                "You have nothing further worth asking -- present the recommendations confidently "
                "in one short, natural sentence. Don't ask a question."
            )

        context = f"Known so far this session: {known}." if known else "Nothing disclosed yet this session."
        response = client.messages.create(
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
