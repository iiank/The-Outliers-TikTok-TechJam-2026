from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

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

_CATEGORY_CATALOG_PATH = "data/catalog.jsonl"
_CATEGORY_MIN_COUNT = 5
_CATEGORY_JUNK_RE = re.compile(r"[\d$]")

_category_pattern_cache: List[Optional["re.Pattern[str]"]] = [None]


def _load_category_terms(catalog_path: str = _CATEGORY_CATALOG_PATH) -> List[str]:
    counts: Dict[str, int] = {}
    try:
        with open(catalog_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    product = json.loads(line)
                except ValueError:
                    continue
                categories = product.get("categories") or []
                if not categories:
                    continue
                leaf = str(categories[-1]).strip().lower()
                if not leaf or _CATEGORY_JUNK_RE.search(leaf):
                    continue
                counts[leaf] = counts.get(leaf, 0) + 1
    except OSError:
        return []
    terms = [term for term, n in counts.items() if n >= _CATEGORY_MIN_COUNT]
    terms.sort(key=lambda t: (-len(t.split()), -len(t)))
    return terms


def _category_pattern() -> Optional["re.Pattern[str]"]:
    cached = _category_pattern_cache[0]
    if cached is not None:
        return cached
    terms = _load_category_terms()
    if not terms:
        return None
    parts = []
    for term in terms:
        if " " in term:
            escaped = re.escape(term)
        elif term.endswith("s"):
            escaped = re.escape(term[:-1]) + "s?"
        else:
            escaped = re.escape(term) + "s?"
        parts.append(escaped)
    pattern = re.compile(r"\b(" + "|".join(parts) + r")\b", re.I)
    _category_pattern_cache[0] = pattern
    return pattern


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _category_hits(message: str) -> List[str]:
    pattern = _category_pattern()
    if not pattern:
        return []
    first_clause = _SENTENCE_SPLIT_RE.split(message, maxsplit=1)[0]
    hits = _hits(pattern, first_clause)
    return hits if hits else _hits(pattern, message)


def _hits(pattern: "re.Pattern[str]", message: str) -> List[str]:
    return [match.group(0).strip().lower() for match in pattern.finditer(message)]


def _size_hits(message: str) -> List[str]:
    return _hits(_SIZE_WORD_RE, message) + [
        f"size {match.group(1)}" for match in _SIZE_NUM_RE.finditer(message)
    ]


def _budget_hits(message: str) -> List[str]:
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


def _category_in_context(current_state: DialogueState) -> bool:
    asked = (current_state.previous_ask_attribute or "").strip()
    return asked in ("", "category")


def regex_extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    message = (user_message or "").strip()
    if not message:
        return {}

    category_in_context = _category_in_context(current_state)

    budget = _budget_hits(message)
    declined = _decline_hit(message, current_state)
    result: Dict[str, Any] = {
        key: values
        for key, values in {
            "category": _category_hits(message) if category_in_context else [],
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


def _drop_offtopic_category(result: Dict[str, Any], current_state: DialogueState) -> Dict[str, Any]:
    known_category = current_state.session_profile.get("category") or []
    if _category_in_context(current_state) or not result.get("category") or not known_category:
        return result
    result = dict(result)
    del result["category"]
    known_lower = {value.lower() for value in known_category}
    rejected = result.get("rejected")
    if rejected:
        rejected_list = [rejected] if isinstance(rejected, str) else list(rejected)
        kept = [value for value in rejected_list if str(value).strip().lower() not in known_lower]
        if kept:
            result["rejected"] = kept
        else:
            del result["rejected"]
    return result


def extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    regex_result = regex_extract_slots(user_message, current_state)
    conflicting = any(
        key in SINGLE_VALUE_SLOTS and key != "category" and len(values) > 1
        for key, values in regex_result.items()
        if key != "intent"
    )
    resolved_anything = len(regex_result) > 1
    category_known = bool(current_state.session_profile.get("category")) or bool(
        regex_result.get("category")
    )
    pure_decline = set(regex_result) <= {"intent", "no_preference"} and "no_preference" in regex_result
    if resolved_anything and not conflicting and (category_known or pure_decline):
        return regex_result
    llm_result = _llm_extract_slots(user_message, current_state)
    if llm_result:
        return _drop_offtopic_category(llm_result, current_state)
    return regex_result
