"""Dialogue State Tracker (DST) for the TechJam conversational shopping agent.

Turns customer utterances into one small, stable, JSON-serializable state
object that the router, retrieval, and re-ranking modules read.

Contract for downstream modules:

* ``to_dict()`` returns pure dicts/lists/strings/ints/bools.
* Slot keys are exactly the ``ask_attribute`` enum from
  ``docs/agent_api_contract.json`` plus ``rejected``.
* Every slot value is a ``list[str]``, always present, possibly empty —
  callers never need ``.get()`` guards.

This module never reads natural language: ``DialogueStateTracker.update``
derives every state change from the injected extractor's return value. The
in-module ``extract_slots`` is a no-op stub; the LLM-backed implementation is
``state.llm_extractor.extract_slots``, injected via
``DialogueStateTracker(extractor=...)``.

Standard library only. No network calls from this file.
"""

from __future__ import annotations
import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "ASK_ATTRIBUTES",
    "INTENT_LABELS",
    "SLOT_KEYS",
    "DialogueState",
    "DialogueStateTracker",
    "extract_slots",
    "empty_session_profile",
    "no_preference_attributes",
    "budget_bounds",
]

ASK_ATTRIBUTES: Tuple[str, ...] = ("category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other",)

SLOT_KEYS: Tuple[str, ...] = ASK_ATTRIBUTES + ("rejected",)
INTENT_LABELS: Tuple[str, ...] = ("buying", "browsing")
SINGLE_VALUE_SLOTS = frozenset({"category", "budget", "size"})

MAX_VALUE_LEN = 180
TOP_K = 10


def empty_session_profile() -> Dict[str, List[str]]:
    """Return a fresh session_profile with every slot present and empty."""
    return {key: [] for key in SLOT_KEYS}


def _as_str_list(value: Any) -> List[str]:
    """Coerce hand-written or reloaded JSON into a list of strings."""
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


@dataclass
class DialogueState:
    session_id: str = ""
    turn: int = 0
    session_profile: Dict[str, List[str]] = field(default_factory=empty_session_profile)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    previous_top_10: List[str] = field(default_factory=list)
    previous_ask_attribute: str = ""
    conflicts_with_previous: bool = False
    intent: Optional[str] = None
    attribute_refusals: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copied, JSON-serializable view of this state."""
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DialogueState":
        """Rebuild a state from :meth:`to_dict` output, or a partial dict."""
        profile = empty_session_profile()
        for key, values in (payload.get("session_profile") or {}).items():
            if key in profile:
                profile[key] = _as_str_list(values)
        user_profile = payload.get("user_profile") or {}
        asked = str(payload.get("previous_ask_attribute") or "")
        raw_refusals = payload.get("attribute_refusals")
        attribute_refusals = (
            {
                str(key): int(value)
                for key, value in raw_refusals.items()
                if str(key) in ASK_ATTRIBUTES and isinstance(value, (int, float))
            }
            if isinstance(raw_refusals, Mapping)
            else {}
        )
        return cls(
            session_id=str(payload.get("session_id", "")),
            turn=int(payload.get("turn", 0)),
            session_profile=profile,
            user_profile=dict(user_profile) if isinstance(user_profile, Mapping) else {},
            previous_top_10=_as_str_list(payload.get("previous_top_10")),
            previous_ask_attribute=asked if asked in ASK_ATTRIBUTES else "",
            conflicts_with_previous=bool(payload.get("conflicts_with_previous", False)),
            intent=payload.get("intent") if payload.get("intent") in INTENT_LABELS else None,
            attribute_refusals=attribute_refusals,
        )

    def copy(self) -> "DialogueState":
        return copy.deepcopy(self)

    def filled_attributes(self) -> List[str]:
        """Attribute slots holding at least one value. Excludes ``rejected``."""
        return [key for key in ASK_ATTRIBUTES if self.session_profile.get(key)]

    def missing_attributes(self) -> List[str]:
        """Empty slots, minus the ones declared no-preference."""
        declined = set(no_preference_attributes(self.session_profile))
        return [
            key
            for key in ASK_ATTRIBUTES
            if not self.session_profile.get(key) and key not in declined
        ]

    def query_terms(self) -> List[str]:
        """All positive slot values, flattened."""
        terms: List[str] = []
        for key in ASK_ATTRIBUTES:
            terms.extend(self.session_profile.get(key, []))
        return terms


def budget_bounds(session_profile: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Turn the ``budget`` slot's strings into numbers a price filter can use.

    The slot holds ``"<=120"``, ``">=25"``, or ``"~60"``. ``None`` means
    unconstrained.
    """
    bounds: Dict[str, Optional[float]] = {
        "min_price": None,
        "max_price": None,
        "target_price": None,
    }
    for raw in session_profile.get("budget") or []:
        text = "".join(str(raw).split())
        for prefix, key, tighter in (
            ("<=", "max_price", min),
            (">=", "min_price", max),
            ("~", "target_price", None),
        ):
            if not text.startswith(prefix):
                continue
            try:
                value = float(text[len(prefix):].strip())
            except ValueError:
                break
            current = bounds[key]
            bounds[key] = value if current is None or tighter is None else tighter(current, value)
            break
    return bounds


def no_preference_attributes(session_profile: Mapping[str, Any]) -> List[str]:
    """Attributes the customer said they have no preference for."""
    prefix = "no_preference:"
    return [
        str(value)[len(prefix):]
        for value in session_profile.get("rejected", [])
        if str(value).startswith(prefix)
    ]

# iian
_NO_PREFERENCE_RE = re.compile(
    r"^\s*i don.t have (?:a|an additional) preference for ([a-z_]+)\b", re.I
)
_NO_SIGNAL_RE = re.compile(r"^\s*those options are not quite right yet\b", re.I)

def simulator_shortcut(message: str) -> Optional[Dict[str, Any]]:
    text = message or ""
    found = _NO_PREFERENCE_RE.match(text)
    if found:
        attribute = found.group(1).lower()
        return {"no_preference": [attribute]} if attribute in ASK_ATTRIBUTES else {}
    if _NO_SIGNAL_RE.match(text):
        return {}
    return None

SlotExtractor = Callable[[str, DialogueState], Dict[str, Any]]


def _clean(value: str) -> str:
    """Collapse whitespace, drop edge punctuation, clip to MAX_VALUE_LEN."""
    cleaned = re.sub(r"\s+", " ", value).strip(" -;,.:!?\t\n")
    return cleaned[:MAX_VALUE_LEN].rstrip()


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Word-boundary containment, so "red" does not match "predator"."""
    return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack, re.IGNORECASE))


def _add(slots: Dict[str, List[str]], attribute: str, value: str) -> None:
    """Append a cleaned value to a slot, deduped by case and by substring."""
    cleaned = _clean(value)
    if not cleaned:
        return
    bucket = slots.setdefault(attribute, [])
    for index, existing in enumerate(bucket):
        if existing.lower() == cleaned.lower() or _contains_phrase(existing, cleaned):
            return
        if _contains_phrase(cleaned, existing):
            bucket[index] = cleaned
            return
    bucket.append(cleaned)


def extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    """Does nothing. Returns ``{}``, meaning "no new info this turn."
    Safe default so just importing this file never calls the network.
    """
    return {}


class _SessionHistory:
    """Incrementally-updated bookkeeping for one session."""

    def __init__(self) -> None:
        self.value_first_seen: Dict[Tuple[str, str], int] = {}
        self.attribute_revisions: Dict[str, int] = {}
        self.rejection_origin: Dict[str, Tuple[str, int]] = {}
        self.override_turns: List[int] = []
        self.mentioned_values: set = set()
        self.last_turn_attributes: List[str] = []
        self.turns_observed: int = 0


class DialogueStateTracker:
    """Owns state transitions for one or more sessions. Keeps a per-session cache (:meth:`get_state`)."""

    def __init__(self, extractor: Optional[SlotExtractor] = None) -> None:
        self.extractor: SlotExtractor = extractor or extract_slots
        self._states: Dict[str, DialogueState] = {}
        self._histories: Dict[str, _SessionHistory] = {}

    def reset(self, session_id: str, user_profile: Optional[Mapping[str, Any]] = None) -> DialogueState:
        """Start a fresh session, mirroring ``Agent.reset``."""
        state = DialogueState(
            session_id=str(session_id),
            turn=0,
            session_profile=empty_session_profile(),
            user_profile=dict(user_profile or {}),
            previous_top_10=[],
            previous_ask_attribute="",
            conflicts_with_previous=False,
            intent=None,
            attribute_refusals={},
        )
        self._states[state.session_id] = state
        self._histories[state.session_id] = _SessionHistory()
        return state

    def get_state(self, session_id: str) -> DialogueState:
        """Return the cached state for a session.

        Raises:
            KeyError: if ``reset`` was never called for this session.
        """
        try:
            return self._states[str(session_id)]
        except KeyError:
            raise KeyError(f"reset() must be called before update() for session {session_id!r}") from None

    def record_recommendations(self, state: DialogueState, parent_asins: List[str]) -> DialogueState:
        """Store what was shown this turn so the next turn can see it.

        Dedupes and clips to :data:`TOP_K`, matching what the evaluator scores.
        """
        unique: List[str] = []
        for asin in parent_asins:
            value = str(asin).strip()
            if value and value not in unique:
                unique.append(value)
            if len(unique) >= TOP_K:
                break
        state.previous_top_10 = unique
        self._states[state.session_id] = state
        return state

    def record_ask(self, state: DialogueState, ask_attribute: Optional[str]) -> DialogueState:
        """Store the ``ask_attribute`` being sent this turn, for the next turn.

        ``None`` clears the record, so a stale attribute cannot mislead the
        next turn's extractor.

        Args:
            state: The state for this turn, mutated in place and returned.
            ask_attribute: A member of :data:`ASK_ATTRIBUTES`, or ``None``.
                Anything else is stored as ``""``.
        """
        attribute = str(ask_attribute or "")
        state.previous_ask_attribute = attribute if attribute in ASK_ATTRIBUTES else ""
        self._states[state.session_id] = state
        return state

    def update(
        self,
        user_message: str,
        current_state: DialogueState,
        turn: Optional[int] = None,
    ) -> DialogueState:
        """Fold one utterance into the state and return a new state object. ``current_state`` is never changed.
        """
        state = current_state.copy()
        state.turn = current_state.turn + 1 if turn is None else int(turn)
        state.conflicts_with_previous = False
        revised_attributes: set = set()
        message = user_message or ""
        shortcut = simulator_shortcut(message)

        if shortcut is None:
            extracted = dict(self.extractor(message, current_state) or {})
        else:
            extracted = dict(shortcut)

        # iian
        # extracted = dict(self.extractor(message, current_state) or {})

        raw_intent = extracted.pop("intent", None)
        state.intent = raw_intent if raw_intent in INTENT_LABELS else None

        # (a) Retractions named by the extractor.
        displaced: set = set()
        raw_retracted = extracted.pop("rejected", None) or []
        if isinstance(raw_retracted, str):
            raw_retracted = [raw_retracted]
        retracted = [text for text in (_clean(str(value)) for value in raw_retracted) if text]
        for value in retracted:
            for attribute in ASK_ATTRIBUTES:
                kept = [
                    held
                    for held in state.session_profile[attribute]
                    if held.lower() != value.lower() and not _contains_phrase(held, value)
                ]
                if len(kept) != len(state.session_profile[attribute]):
                    for dropped in state.session_profile[attribute]:
                        if dropped not in kept:
                            displaced.add(dropped.lower())
                            if dropped not in state.session_profile["rejected"]:
                                state.session_profile["rejected"].append(dropped)
                    state.session_profile[attribute] = kept
                    revised_attributes.add(attribute)
            displaced.add(value.lower())
            if value not in state.session_profile["rejected"]:
                state.session_profile["rejected"].append(value)
        if retracted:
            state.conflicts_with_previous = True

        # (b) No-preference declarations (Boundary scenario).
        raw_declined = extracted.pop("no_preference", None) or []
        if isinstance(raw_declined, str):
            raw_declined = [raw_declined]
        for value in raw_declined:
            attribute = _clean(str(value)).lower()
            if not attribute:
                continue
            if attribute not in ASK_ATTRIBUTES:
                attribute = "other"
            marker = f"no_preference:{attribute}"
            if marker not in state.session_profile["rejected"]:
                state.session_profile["rejected"].append(marker)

        # (c) Apply new values.
        for attribute, raw_values in extracted.items():
            if attribute not in ASK_ATTRIBUTES:
                continue
            values = _as_str_list(raw_values)
            existing = state.session_profile[attribute]
            incompatible = (
                bool(existing)
                and attribute in SINGLE_VALUE_SLOTS
                and not _same_values(existing, values)
            )
            if incompatible:
                state.conflicts_with_previous = True
                revised_attributes.add(attribute)
                for stale in existing:
                    displaced.add(stale.lower())
                    if stale not in state.session_profile["rejected"]:
                        state.session_profile["rejected"].append(stale)
                state.session_profile[attribute] = []
            for value in values:
                # A value dropped earlier this turn is not taken back.
                if value.lower() in displaced:
                    continue
                _add(state.session_profile, attribute, value)

        # if the attribute asked last turn is still unanswered/undeclined, its
        # count keeps growing; otherwise it clears.
        asked_last_turn = current_state.previous_ask_attribute
        if asked_last_turn:
            answered = bool(state.session_profile.get(asked_last_turn))
            declined = asked_last_turn in no_preference_attributes(state.session_profile)
            if answered or declined:
                state.attribute_refusals.pop(asked_last_turn, None)
            else:
                state.attribute_refusals[asked_last_turn] = (
                    state.attribute_refusals.get(asked_last_turn, 0) + 1
                )

        self._states[state.session_id] = state
        history = self._histories.setdefault(state.session_id, _SessionHistory())
        self._update_history(history, current_state, state, revised_attributes)
        return state

    def _update_history(
        self,
        history: "_SessionHistory",
        current_state: DialogueState,
        state: DialogueState,
        revised_attributes: "set",
    ) -> None:
        """Fold one turn into ``history`` in place.

        One before/after diff of ``session_profile``, done once per turn,
        instead of replaying a growing snapshot log later.
        """
        history.turns_observed += 1
        touched: set = set()
        for attribute in ASK_ATTRIBUTES:
            before = {value.lower() for value in current_state.session_profile.get(attribute, [])}
            after = {value.lower() for value in state.session_profile.get(attribute, [])}
            for folded in after - before:
                history.value_first_seen.setdefault((attribute, folded), state.turn)
                history.mentioned_values.add(folded)
                touched.add(attribute)
            for folded in before - after:
                history.rejection_origin[folded] = (attribute, state.turn)
                touched.add(attribute)
        for attribute in revised_attributes:
            history.attribute_revisions[attribute] = history.attribute_revisions.get(attribute, 0) + 1
        history.last_turn_attributes = sorted(touched)
        if state.conflicts_with_previous:
            history.override_turns.append(state.turn)

    def get_history_summary(self, session_id: str) -> Dict[str, Any]:
        """A compact, JSON-serializable view of one session's history so far."""
        history = self._histories.get(str(session_id)) or _SessionHistory()
        value_first_seen: Dict[str, Dict[str, int]] = {}
        for (attribute, folded), seen_turn in history.value_first_seen.items():
            value_first_seen.setdefault(attribute, {})[folded] = seen_turn
        return {
            "value_first_seen": value_first_seen,
            "attribute_revisions": dict(history.attribute_revisions),
            "rejection_origin": {
                folded: [attribute, dropped_turn]
                for folded, (attribute, dropped_turn) in history.rejection_origin.items()
            },
            "override_turns": list(history.override_turns),
            "mentioned_values": sorted(history.mentioned_values),
            "last_turn_attributes": list(history.last_turn_attributes),
            "turns_observed": history.turns_observed,
        }


def _same_values(existing: List[str], incoming: List[str]) -> bool:
    """True when the incoming values add nothing new."""
    known = {value.lower() for value in existing}
    return all(value.lower() in known for value in incoming)
