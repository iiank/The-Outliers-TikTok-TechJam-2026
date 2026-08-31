"""Regex slot extractor: cheap first pass before the LLM.

Closed-vocabulary slots (material, color, size, budget, use_case, style) are
resolved with local keyword/regex matching, with no network call or token use.

Open-vocabulary slots (category, brand, feature, other, no_preference,
rejected) are not resolved here. Turns that require them fall through to
``state.llm_extractor.extract_slots``.

Escalation rule: use the LLM only when regex finds nothing or when a
single-value slot (``category``, ``budget``, ``size``) has multiple distinct
matches that require contextual disambiguation. Everything else is resolved
locally.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .dialogue_state import ASK_ATTRIBUTES, DialogueState, SINGLE_VALUE_SLOTS
from .llm_extractor import extract_slots as _llm_extract_slots

__all__ = ["extract_slots", "regex_extract_slots"]

_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|denim|linen|suede|fleece)\b",
    re.I,
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|navy|beige|tan)\b",
    re.I,
)
_USE_CASE_RE = re.compile(
    r"\b(hiking|running|gym|winter|summer|outdoor|work|casual|formal|travel)\b", re.I
)
_STYLE_RE = re.compile(
    r"\b(slim fit|regular fit|relaxed fit|crew neck|v-neck|long sleeve|short sleeve|sleeveless)\b",
    re.I,
)
_SIZE_WORD_RE = re.compile(
    r"\b(extra small|extra large|xs|small|medium|large|xxl|xl)\b", re.I
)
_SIZE_NUM_RE = re.compile(r"\bsize\s*(\d{1,2})\b", re.I)

_BUDGET_RE = re.compile(
    r"\bunder\s+\$?(?P<under>\d+(?:\.\d{2})?)\b"
    r"|\bover\s+\$?(?P<over>\d+(?:\.\d{2})?)\b"
    r"|\bbudget\s+of\s+\$?(?P<around1>\d+(?:\.\d{2})?)\b"
    r"|\$\s?(?P<around2>\d+(?:\.\d{2})?)",
    re.I,
)
_BUY_RE = re.compile(
    r"\b(i need|i want|looking for a|must have|has to (?:be|have)|require[sd]?|specifically)\b",
    re.I,
)
_BROWSE_RE = re.compile(
    r"\b(just (?:looking|browsing)|not sure|still exploring|no rush|open to|"
    r"any (?:ideas|suggestions)|thinking about|just curious)\b",
    re.I,
)

_DECLINE_RE = re.compile(
    r"\b(no preference|don'?t have (?:an? )?(?:additional )?preference|"
    r"not picky|use your judgment|whatever'?s? fine|don'?t care)\b",
    re.I,
)


def _hits(pattern: "re.Pattern[str]", message: str) -> List[str]:
    return [match.group(0).strip().lower() for match in pattern.finditer(message)]


def _size_hits(message: str) -> List[str]:
    return _hits(_SIZE_WORD_RE, message) + [
        f"size {match.group(1)}" for match in _SIZE_NUM_RE.finditer(message)
    ]


def _budget_hits(message: str) -> List[str]:
    """Return budget matches in the format expected by ``budget_bounds()``."""
    hits = []
    for match in _BUDGET_RE.finditer(message):
        groups = match.groupdict()
        if groups["under"]:
            hits.append(f"<={groups['under']}")
        elif groups["over"]:
            hits.append(f">={groups['over']}")
        else:
            hits.append(f"~{groups['around1'] or groups['around2']}")
    return hits


def _intent(message: str, current_state: DialogueState, has_budget: bool) -> str:
    buy = bool(_BUY_RE.search(message)) or has_budget
    browse = bool(_BROWSE_RE.search(message))
    if buy != browse:
        return "buying" if buy else "browsing"
    has_constraint = any(current_state.session_profile.values())
    return "buying" if has_constraint else "browsing"


def _decline_hit(message: str, current_state: DialogueState) -> List[str]:
    asked = (current_state.previous_ask_attribute or "").strip()
    if asked in ASK_ATTRIBUTES and _DECLINE_RE.search(message):
        return [asked]
    return []


def regex_extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    """Extract closed-vocabulary slots locally without an LLM call."""
    message = (user_message or "").strip()
    if not message:
        return {}

    budget = _budget_hits(message)
    declined = _decline_hit(message, current_state)
    result: Dict[str, Any] = {
        key: values
        for key, values in {
            "material": _hits(_MATERIAL_RE, message),
            "color": _hits(_COLOR_RE, message),
            "size": _size_hits(message),
            "budget": budget,
            "use_case": _hits(_USE_CASE_RE, message),
            "style": _hits(_STYLE_RE, message),
            "no_preference": declined,
        }.items()
        if values
    }
    result["intent"] = _intent(message, current_state, has_budget=bool(budget))
    return result


def extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    """Use regex extraction first and fall back to the LLM when needed.

    Matches the ``DialogueStateTracker`` extractor interface and return shape.
    """
    regex_result = regex_extract_slots(user_message, current_state)
    conflicting = any(
        key in SINGLE_VALUE_SLOTS and len(values) > 1
        for key, values in regex_result.items()
        if key != "intent"
    )
    resolved_anything = len(regex_result) > 1
    if resolved_anything and not conflicting:
        return regex_result
    return _llm_extract_slots(user_message, current_state) or regex_result
