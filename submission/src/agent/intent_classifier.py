"""Buy vs. browse intent classification.

Two implementations of the same interface: a zero-dependency regex
heuristic, and an LLM classifier that reads more nuance. ``Agent``'s default
wiring calls the LLM classifier live; the regex classifier is a complete,
tested, documented offline alternative -- swap it in explicitly with
``Agent(intent_classifier=RegexIntentClassifier())``. There is no automatic
fallback between the two: if the LLM call fails, ``classify`` raises, and
that turn degrades to a miss the same way any other ``respond()`` exception
does (see the plan doc's flagged risk).

CROSS-BOUNDARY POINTS IN THIS FILE:

* ``BOUNDARY(state)`` -- the ``DialogueState`` import (type only) and, in
  ``RegexIntentClassifier``, the ``state.filled_attributes()`` call: reads a
  method owned by the state teammate's ``DialogueState`` class.
* ``BOUNDARY(external: Anthropic API)`` -- ``LLMIntentClassifier`` calls out
  to the Anthropic Messages API over the network (``import anthropic`` /
  ``client.messages.create(...)``). Not a repo-internal boundary, but a
  network/cost dependency worth the same visibility -- see
  ``docs/submission_rules.md`` on disclosing model/network usage.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Protocol

from state.dialogue_state import DialogueState  # BOUNDARY(state): type import only

__all__ = ["IntentClassifier", "RegexIntentClassifier", "LLMIntentClassifier"]

MODES = ("buy", "browse")


class IntentClassifier(Protocol):
    def classify(self, user_message: str, state: DialogueState) -> str:
        """Return ``"buy"`` or ``"browse"``."""
        ...


_BUY_RE = re.compile(
    r"\b(i need|i want|looking for a|must have|has to (?:be|have)|require[sd]?|"
    r"specifically|size \d+|under \$|budget of)\b"
    # Not wrapped in the leading \b above: "$" is a non-word character, so a
    # boundary almost never holds immediately before it in real text ("$80"
    # is preceded by a space or string-start, not a word char) -- \b\$\d+\b
    # only ever matched contrived input glued to a word ("cost$8").
    r"|\$\d+",
    re.IGNORECASE,
)
_BROWSE_RE = re.compile(
    # "maybe" removed: it's too broad, and fires alongside a real buy signal
    # ("Maybe I need a size 10 boot" matched both), turning a clear "buy"
    # into a tie instead of reading it correctly.
    r"\b(just (?:looking|browsing)|not sure|still exploring|no rush|open to|"
    r"any (?:ideas|suggestions)|thinking about|just curious)\b",
    re.IGNORECASE,
)


class RegexIntentClassifier:
    """Keyword heuristic: concrete requirement language reads as "buy",
    vague/exploratory phrasing reads as "browse". No network dependency."""
    
    # Not called in agent as of now but built jic

    def classify(self, user_message: str, state: DialogueState) -> str:
        message = user_message or ""
        buy_hit = bool(_BUY_RE.search(message))
        browse_hit = bool(_BROWSE_RE.search(message))
        if buy_hit and not browse_hit:
            return "buy"
        if browse_hit and not buy_hit:
            return "browse"
        # Tie or no signal: lean on whether a hard constraint is already known.
        return "buy" if state.filled_attributes() else "browse"  # BOUNDARY(state): reads DialogueState.filled_attributes()


_SYSTEM_PROMPT = """You are a message classifier. Classify one customer message from a shopping conversation as either "buy" or "browse".

"buy": the customer states (now or earlier this session) a concrete requirement -- a specific product, size, material, color, brand, or a firm budget. They know roughly what they want.
"browse": the customer is exploring -- vague, open-ended, no firm requirement yet, open to suggestions.

Examples:
Message: "I need a waterproof size 10 hiking boot under $80." -> buy
Message: "Just looking around, not sure what I want yet." -> browse
Message: "Something for the gym I guess, nothing specific." -> browse
Message: "It has to be black leather, that's non-negotiable." -> buy
Message: "Any ideas? I'm open to whatever looks good." -> browse
Message: "I'm looking for a wool sweater, medium, under $60." -> buy

Reply with exactly one word: buy or browse. Nothing else."""


class LLMIntentClassifier:
    """Few-shot LLM classifier. Re-run every turn so mid-session shifts
    (Intent Override) are picked up, not just the opening message."""

    def __init__(self, client: Optional[object] = None, model: str = "claude-haiku-4-5") -> None:
        self._client = client
        self.model = model
        self.last_usage: Dict[str, int] = {}

    def _get_client(self):
        if self._client is None:
            import anthropic  # BOUNDARY(external: Anthropic API)

            self._client = anthropic.Anthropic()
        return self._client

    def classify(self, user_message: str, state: DialogueState) -> str:
        client = self._get_client()
        # "rejected" is excluded: those are values the customer declined, not
        # known preferences -- including it here previously presented a
        # rejected value (and the raw "no_preference:<attr>" marker string)
        # to the model as if it were a confirmed fact.
        known = ", ".join(
            f"{key}={values}" for key, values in state.session_profile.items()
            if key != "rejected" and values
        )  # BOUNDARY(state): reads DialogueState.session_profile
        context = f"Known so far this session: {known}." if known else "Nothing disclosed yet this session."
        response = client.messages.create(  # BOUNDARY(external: Anthropic API) -- network call
            model=self.model,
            max_tokens=8,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{context}\nMessage: {user_message!r}"}],
        )
        self.last_usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        text = next((block.text for block in response.content if block.type == "text"), "")
        return "buy" if "buy" in text.lower() else "browse"
