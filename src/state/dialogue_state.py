"""Dialogue State Tracker (DST) for the TechJam conversational shopping agent.

Turns customer utterances into one small, stable, JSON-serializable state object
that the router, retrieval, and re-ranking modules read.

Contract for downstream modules:

* ``to_dict()`` returns pure dicts/lists/strings/ints/bools. Reading a value
  never requires importing this module.
* Slot keys are exactly the ``ask_attribute`` enum from
  ``docs/agent_api_contract.json`` (category, material, color, size, style,
  brand, budget, feature, use_case, other) plus ``rejected``.
* Every slot value is a ``list[str]``, always present, possibly empty. No
  ``None``, no missing keys, so callers never need ``.get()`` guards.

Slot extraction is NOT implemented yet. ``extract_slots`` is a stub that returns
nothing, so slots never fill. Write it, then inject it with
``DialogueStateTracker(extractor=...)``; see the TODO list in that function.
Everything else here is finished and does not change when you do.

Standard library only, like the starter agent.
"""

from __future__ import annotations
import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "ASK_ATTRIBUTES",
    "SLOT_KEYS",
    "DialogueState",
    "DialogueStateTracker",
    "extract_slots",
    "empty_session_profile",
    "no_preference_attributes",
]


#: The ``ask_attribute`` enum from docs/agent_api_contract.json, minus ``null``.
#: Slot names match it exactly, so an unfilled slot name can be used directly as
#: the ``ask_attribute`` field of a response.
ASK_ATTRIBUTES: Tuple[str, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)

SLOT_KEYS: Tuple[str, ...] = ASK_ATTRIBUTES + ("rejected",)

#: Slots where a second, different value contradicts the first instead of
#: refining it. Triggers override detection even without a negation word.
SINGLE_VALUE_SLOTS = frozenset({"category", "budget", "size"})

#: Longest slot value kept. Matches the evaluator's own constraint clipping.
MAX_VALUE_LEN = 180

#: Recommendations scored per turn (``top_k`` is const 10 in the contract).
TOP_K = 10


def empty_session_profile() -> Dict[str, List[str]]:
    """Return a fresh session_profile with every slot present and empty."""
    return {key: [] for key in SLOT_KEYS}


def _as_str_list(value: Any) -> List[str]:
    """Coerce hand-written or reloaded JSON into a list of strings.

    A bare string becomes a one-item list rather than a list of characters,
    which is the trap when a slot is written as ``"color": "black"``.
    """
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


@dataclass
class DialogueState:
    """Everything the agent knows about one session at one point in time.

    Attributes:
        session_id: The session id from the harness, so a state object is
            self-describing in logs.
        turn: Mirrors the ``turn`` argument the harness passes to ``respond()``
            (1-based). ``0`` means reset happened and no turn ran yet.
        session_profile: Turn-by-turn slots. Keys are exactly :data:`SLOT_KEYS`;
            values are lists of strings in first-seen order.
        user_profile: The anonymized profile given at ``reset()``. Read-only
            long-term context. This module never changes it.
        previous_top_10: ``parent_asin`` values shown on the previous turn. Empty
            at session start. Set by
            :meth:`DialogueStateTracker.record_recommendations`.
        conflicts_with_previous: True when this turn's utterance contradicted
            existing state. Recomputed every turn, so it is not sticky.
    """

    session_id: str = ""
    turn: int = 0
    session_profile: Dict[str, List[str]] = field(default_factory=empty_session_profile)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    previous_top_10: List[str] = field(default_factory=list)
    conflicts_with_previous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep-copied, JSON-serializable view of this state.

        Cross this boundary once, then work in plain dicts.
        """
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DialogueState":
        """Rebuild a state from :meth:`to_dict` output, or from a partial dict."""
        profile = empty_session_profile()
        for key, values in (payload.get("session_profile") or {}).items():
            if key in profile:
                profile[key] = _as_str_list(values)
        user_profile = payload.get("user_profile") or {}
        return cls(
            session_id=str(payload.get("session_id", "")),
            turn=int(payload.get("turn", 0)),
            session_profile=profile,
            user_profile=dict(user_profile) if isinstance(user_profile, Mapping) else {},
            previous_top_10=_as_str_list(payload.get("previous_top_10")),
            conflicts_with_previous=bool(payload.get("conflicts_with_previous", False)),
        )

    def copy(self) -> "DialogueState":
        """Deep copy, so callers cannot change a state they were handed."""
        return copy.deepcopy(self)

    def filled_attributes(self) -> List[str]:
        """Attribute slots holding at least one value. Excludes ``rejected``."""
        return [key for key in ASK_ATTRIBUTES if self.session_profile.get(key)]

    def missing_attributes(self) -> List[str]:
        """Empty slots, minus the ones declared no-preference.

        Every element is a legal ``ask_attribute`` value, ready for a
        question-selection policy.
        """
        declined = set(no_preference_attributes(self.session_profile))
        return [
            key
            for key in ASK_ATTRIBUTES
            if not self.session_profile.get(key) and key not in declined
        ]

    def query_terms(self) -> List[str]:
        """All positive slot values, flattened. A cheap retrieval query seed."""
        terms: List[str] = []
        for key in ASK_ATTRIBUTES:
            terms.extend(self.session_profile.get(key, []))
        return terms


def no_preference_attributes(session_profile: Mapping[str, Any]) -> List[str]:
    """Attributes the customer said they have no preference for.

    Takes a ``session_profile`` dict, so downstream modules can call it without
    holding a :class:`DialogueState`.
    """
    prefix = "no_preference:"
    return [
        str(value)[len(prefix):]
        for value in session_profile.get("rejected", [])
        if str(value).startswith(prefix)
    ]


SlotExtractor = Callable[[str, DialogueState], Dict[str, List[str]]]


def _clean(value: str) -> str:
    """Collapse whitespace, drop edge punctuation, clip to MAX_VALUE_LEN."""
    cleaned = re.sub(r"\s+", " ", value).strip(" -;,.:!?\t\n")
    return cleaned[:MAX_VALUE_LEN].rstrip()


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Word-boundary containment, so "red" does not match "predator"."""
    return bool(re.search(r"\b" + re.escape(needle) + r"\b", haystack, re.IGNORECASE))


def _add(slots: Dict[str, List[str]], attribute: str, value: str) -> None:
    """Append a cleaned value to a slot, deduped by case and by substring.

    A short hit already covered by a longer phrase in the same slot is dropped
    ("leather" next to "full-grain leather"). The longer phrase wins if it
    arrives second.
    """
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


def extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, List[str]]:
    """Extract slot values from one utterance. NOT IMPLEMENTED YET.

    This is the only place in the module that reads natural language, and right
    now it is an empty stub: it returns ``{}``, so no slot ever fills. Everything
    else already works and does not change when this is written, including the
    state schema, override detection, ``rejected``, and the transition log.

    Args:
        user_message: The raw customer utterance for this turn.
        current_state: State before this turn, available as context.

    Returns:
        ``{attribute: [values]}`` for the attributes stated in this message. Keys
        must come from :data:`ASK_ATTRIBUTES`, with one optional extra key:

        ``"rejected"``: values the customer just dropped, copied as they appear
        in ``current_state.session_profile``. The tracker removes each one from
        whichever slot holds it, records it as a negative term, and sets
        ``conflicts_with_previous``. To clear a whole slot, list all of its
        current values. Leave the key out when nothing was retracted, which lets
        the tracker fall back to its own keyword check.

        Do not emit ``no_preference:<attribute>`` markers. Those stay the
        tracker's job.

    TODO: implement with an LLM structured-output call.

      1. Define a JSON schema whose properties are exactly ASK_ATTRIBUTES plus
         "rejected", each one an array of strings, and tell the model to fill
         only what the message actually says. No guessing.
      2. Pass ``current_state.session_profile`` in as context. The model needs it
         to refine an existing value instead of repeating it, and it cannot fill
         "rejected" correctly without seeing what is currently held.
      3. Normalize budget to "<=45", ">=45", or "~45" before returning. Nothing
         downstream parses prose prices any more.
      4. Return ``{}`` on an API error, timeout, or schema mismatch, so a bad
         turn degrades into "no new information". The harness counts an
         exception as a miss, per docs/competition_specification.md. On that
         path the tracker's keyword check is the only override signal left,
         which is why it is still here.
      5. Report prompt and completion token counts back to the agent for the
         response ``usage`` field.
      6. Inject it with ``DialogueStateTracker(extractor=my_llm_fn)``. Nothing in
         this module needs to change.
    """
    return {}


# Fallback override detection, used when the extractor names no retraction (for
# example when an API call failed). "Actually, ignore my earlier preference. What
# I need is: ..." is the evaluator's literal override line; the rest of the list
# covers paraphrase.
_OVERRIDE_RE = re.compile(
    r"\b(actually|instead|forget|ignore|nevermind|never mind|scratch that|"
    r"no longer|rather than|changed my mind|on second thought|"
    r"not looking for|don't want|do not want|no thanks)\b",
    re.IGNORECASE,
)
# Bare "no" and "not" are checked separately. They are common in the simulator's
# non-override replies, so they only count once the fast paths below have run.
_WEAK_NEGATION_RE = re.compile(r"\bnot\b|\bno\b", re.IGNORECASE)
#: Comparatives where "no"/"not" is part of a constraint, not a retraction.
_BENIGN_NEGATION_RE = re.compile(
    r"\bno(?:t)? (?:more|less|larger|smaller|bigger) than\b", re.IGNORECASE
)

_NO_PREFERENCE_RE = re.compile(
    r"\b(?:no|any|don'?t have|do not have|without)\s*(?:an?y?\s+)?(?:additional\s+)?"
    r"preference\b\s*(?:for|on|about)?\s*([a-z_]+)?",
    re.IGNORECASE,
)
_NO_INFO_RE = re.compile(
    r"not quite right|nothing (?:here|there) (?:is|works)|none of (?:these|those)",
    re.IGNORECASE,
)


class DialogueStateTracker:
    """Owns state transitions for one or more sessions.

    The harness only passes ``session_id`` to ``Agent.respond()``, so this class
    also keeps a per-session cache (:meth:`get_state`). The required
    ``update(user_message, current_state)`` signature is unchanged; the cache is
    a convenience, not the source of truth.

    Args:
        extractor: Slot-extraction callable. Defaults to :func:`extract_slots`.
            Pass an LLM-backed function with the same signature to upgrade the
            tracker without touching this class.
    """

    def __init__(self, extractor: Optional[SlotExtractor] = None) -> None:
        self.extractor: SlotExtractor = extractor or extract_slots
        self._states: Dict[str, DialogueState] = {}
        self._transition_log: List[Dict[str, Any]] = []

    def reset(self, session_id: str, user_profile: Optional[Mapping[str, Any]] = None) -> DialogueState:
        """Start a fresh session, mirroring ``Agent.reset``.

        Empties ``session_profile``, stores ``user_profile`` as read-only
        context, zeroes ``turn`` and ``previous_top_10``. Returns the new state,
        which is also available from :meth:`get_state`.
        """
        state = DialogueState(
            session_id=str(session_id),
            turn=0,
            session_profile=empty_session_profile(),
            user_profile=dict(user_profile or {}),
            previous_top_10=[],
            conflicts_with_previous=False,
        )
        self._states[state.session_id] = state
        return state

    def get_state(self, session_id: str) -> DialogueState:
        """Return the cached state for a session.

        Raises:
            KeyError: if ``reset`` was never called for this session. Same
                precondition the starter agent enforces in ``respond``.
        """
        try:
            return self._states[str(session_id)]
        except KeyError:
            raise KeyError(f"reset() must be called before update() for session {session_id!r}") from None

    def record_recommendations(self, state: DialogueState, parent_asins: List[str]) -> DialogueState:
        """Store what was shown this turn so the next turn can see it.

        Call after building the response. Dedupes and clips to :data:`TOP_K`,
        matching what the evaluator scores.
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

    def update(
        self,
        user_message: str,
        current_state: DialogueState,
        turn: Optional[int] = None,
    ) -> DialogueState:
        """Fold one utterance into the state and return a new state object.

        ``current_state`` is never changed.

        Args:
            user_message: Raw customer utterance.
            current_state: State from ``reset()`` or the previous ``update()``.
            turn: The harness's ``turn`` argument. Defaults to
                ``current_state.turn + 1``.

        Returns:
            A new :class:`DialogueState`. ``conflicts_with_previous`` covers this
            turn only.
        """
        old_snapshot = current_state.to_dict()
        state = current_state.copy()
        state.turn = current_state.turn + 1 if turn is None else int(turn)
        state.conflicts_with_previous = False
        reasons: List[str] = []
        message = user_message or ""

        # Fast path A, Boundary scenario: "I don't have a preference for color."
        # Runs before override detection, because "don't" would otherwise read
        # as a negation marker.
        # LIMITATION: returns before the extractor runs, so a message that both
        # declines an attribute and states a new constraint loses the constraint.
        # The simulator sends these as standalone lines, so it does not bite
        # today. Revisit when the LLM extractor lands and paraphrase is possible.
        no_preference = _NO_PREFERENCE_RE.search(message)
        if no_preference:
            attribute = (no_preference.group(1) or "").lower()
            if attribute not in ASK_ATTRIBUTES:
                attribute = "other"
            marker = f"no_preference:{attribute}"
            if marker not in state.session_profile["rejected"]:
                state.session_profile["rejected"].append(marker)
            reasons.append(f"no_preference_declared:{attribute}")
            return self._commit(state, old_snapshot, reasons)

        # Fast path B: "those options are not quite right yet" carries no new
        # constraint, so extracting from it only adds noise.
        if _NO_INFO_RE.search(message):
            reasons.append("no_new_information")
            return self._commit(state, old_snapshot, reasons)

        extracted = dict(self.extractor(message, current_state) or {})

        # (a) Retractions named by the extractor take priority. An LLM that sees
        #     the current slots can say exactly what the customer dropped, which
        #     no pattern can work out from wording alone.
        displaced: set = set()
        # Model output is untrusted: accept a bare string, and drop empty values,
        # which would otherwise match every slot and wipe the whole profile.
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
                    reasons.append(f"retracted:{attribute}")
            displaced.add(value.lower())
            if value not in state.session_profile["rejected"]:
                state.session_profile["rejected"].append(value)
        if retracted:
            state.conflicts_with_previous = True

        # (b) Fallback override check, for when the extractor named nothing. Bare
        #     "no"/"not" only counts after the harmless comparatives ("no more
        #     than $40") are taken out.
        probe = _BENIGN_NEGATION_RE.sub(" ", message)
        override_marker = bool(_OVERRIDE_RE.search(message)) or bool(_WEAK_NEGATION_RE.search(probe))
        # Nothing held yet means nothing to contradict, so an opening line like
        # "I'm not sure what I want" must not raise the flag. Still logged, so
        # the marker is visible when debugging.
        had_prior_constraints = any(
            current_state.session_profile.get(key) for key in ASK_ATTRIBUTES
        )
        if override_marker and had_prior_constraints:
            state.conflicts_with_previous = True
            reasons.append("negation_marker")
        elif override_marker:
            reasons.append("negation_marker_without_prior_state")

        # Trust a precise retraction list over a blunt keyword hit: if the
        # extractor said what to drop, do not also clear slots it chose to keep.
        clear_on_marker = override_marker and not retracted

        # (c) Apply new values, moving whatever they contradict to `rejected`.
        for attribute, values in extracted.items():
            if attribute not in ASK_ATTRIBUTES:
                continue
            existing = state.session_profile[attribute]
            incompatible = bool(existing) and (
                clear_on_marker or attribute in SINGLE_VALUE_SLOTS
            ) and not _same_values(existing, values)
            if incompatible:
                state.conflicts_with_previous = True
                reasons.append(f"slot_replaced:{attribute}")
                for stale in existing:
                    displaced.add(stale.lower())
                    if stale not in state.session_profile["rejected"]:
                        state.session_profile["rejected"].append(stale)
                state.session_profile[attribute] = []
            before = len(state.session_profile[attribute])
            for value in values:
                # A value dropped earlier this turn is not taken back: "forget
                # the running shoes, I want hiking boots" only mentions
                # "running" to retract it. Keeps slots and `rejected` disjoint.
                if value.lower() in displaced:
                    continue
                _add(state.session_profile, attribute, value)
            if not incompatible and len(state.session_profile[attribute]) > before:
                reasons.append(f"slot_filled:{attribute}")

        if not extracted and not retracted:
            reasons.append("no_slots_extracted")

        return self._commit(state, old_snapshot, reasons)

    def _commit(
        self,
        state: DialogueState,
        old_snapshot: Dict[str, Any],
        reasons: List[str],
    ) -> DialogueState:
        """Cache the new state and add a transition-log entry."""
        self._states[state.session_id] = state
        self._transition_log.append(
            {
                "turn": state.turn,
                "old_state": old_snapshot,
                "new_state": state.to_dict(),
                "trigger_reason": ", ".join(reasons) if reasons else "no_change",
            }
        )
        return state

    def get_transition_log(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the transition log, JSON-serializable, oldest first.

        Each entry is ``{turn, old_state, new_state, trigger_reason}``. Pass
        ``session_id`` to filter to one session.
        """
        if session_id is None:
            return copy.deepcopy(self._transition_log)
        wanted = str(session_id)
        return [
            copy.deepcopy(entry)
            for entry in self._transition_log
            if entry["new_state"].get("session_id") == wanted
        ]


def _same_values(existing: List[str], incoming: List[str]) -> bool:
    """True when the incoming values add nothing new, so there is no conflict."""
    known = {value.lower() for value in existing}
    return all(value.lower() in known for value in incoming)
