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

This module never reads natural language. ``DialogueStateTracker.update`` derives
every state change from what the injected extractor returns, so the only
language understanding in the pipeline lives behind that one callable. The
in-module ``extract_slots`` is a no-op stub that returns ``{}``; the LLM-backed
implementation is ``state.llm_extractor.extract_slots``, injected with
``DialogueStateTracker(extractor=...)``.

Standard library only, like the starter agent. No network calls from this file.
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

#: Buying vs. Browsing (Pillar I, dual-track routing). Set from the same joint
#: extraction call that fills the slots above — see ``state.llm_extractor``.
INTENT_LABELS: Tuple[str, ...] = ("Buying", "Browsing")

#: Slots where a second, different value contradicts the first instead of
#: refining it. A structural check on already-extracted values, so it catches an
#: override the extractor did not name in ``rejected``.
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
        previous_ask_attribute: The ``ask_attribute`` sent on the previous turn,
            or ``""`` if none was. Set by
            :meth:`DialogueStateTracker.record_ask`. This is what makes a bare
            reply readable: "black" means nothing on its own, but "black" right
            after we asked ``color`` is a colour. The extractor is given it for
            exactly that reason, and a question-selection policy can read it to
            avoid asking the same thing twice running.
        conflicts_with_previous: True when this turn's utterance contradicted
            existing state. Recomputed every turn, so it is not sticky.
        intent: ``"Buying"``, ``"Browsing"``, or ``None`` when this turn's
            joint extraction call did not resolve one (empty message, API
            failure, or an out-of-enum reply). Recomputed every turn from the
            same call that fills ``session_profile``, so it is not sticky
            either — do not read ``None`` as a default label.
    """

    session_id: str = ""
    turn: int = 0
    session_profile: Dict[str, List[str]] = field(default_factory=empty_session_profile)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    previous_top_10: List[str] = field(default_factory=list)
    previous_ask_attribute: str = ""
    conflicts_with_previous: bool = False
    intent: Optional[str] = None

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
        asked = str(payload.get("previous_ask_attribute") or "")
        return cls(
            session_id=str(payload.get("session_id", "")),
            turn=int(payload.get("turn", 0)),
            session_profile=profile,
            user_profile=dict(user_profile) if isinstance(user_profile, Mapping) else {},
            previous_top_10=_as_str_list(payload.get("previous_top_10")),
            previous_ask_attribute=asked if asked in ASK_ATTRIBUTES else "",
            conflicts_with_previous=bool(payload.get("conflicts_with_previous", False)),
            intent=payload.get("intent") if payload.get("intent") in INTENT_LABELS else None,
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


def budget_bounds(session_profile: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Turn the ``budget`` slot's strings into numbers a price filter can use.

    The slot holds ``"<=120"``, ``">=25"``, or ``"~60"`` (see the extractor's
    contract). This is the only place that spelling is decoded, so a numeric
    filter never has to know about it — and if the format ever changes, one
    function changes with it.

    Takes a ``session_profile`` dict rather than a :class:`DialogueState`, so it
    can be called on ``state.to_dict()["session_profile"]`` from a module that
    imports nothing from here.

    Args:
        session_profile: Any dict with a ``budget`` key. A missing or empty
            ``budget`` is fine and yields all-``None``.

    Returns:
        ``{"min_price": float|None, "max_price": float|None,
        "target_price": float|None}``. ``None`` means "unconstrained", so
        ``min_price is None and max_price is None and target_price is None``
        is the "no budget stated, skip filtering" test.

        A range gives both bounds: ``[">=25", "<=60"]`` -> ``min 25, max 60``.
        Repeated bounds take the *tighter* one, so a stale ``"<=120"`` sitting
        beside a newer ``"<=80"`` cannot widen the filter. ``"~60"`` sets
        ``target_price`` only; how much slack to allow around it is the filter's
        decision, not this module's.

        Unparseable entries are skipped rather than raised on. The extractor is
        told to normalize, but it is an LLM, so a stray ``"under $50"`` must
        degrade to "no numeric constraint" instead of taking down the turn.
    """
    bounds: Dict[str, Optional[float]] = {
        "min_price": None,
        "max_price": None,
        "target_price": None,
    }
    for raw in session_profile.get("budget") or []:
        # Strip *all* whitespace, not just the ends. Observed failure: a model
        # emitting "< =120" instead of "<=120", which then silently parses as no
        # constraint at all and drops the customer's price cap.
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
                break  # normalization failed upstream; ignore this entry
            current = bounds[key]
            bounds[key] = value if current is None or tighter is None else tighter(current, value)
            break
    return bounds


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


#: Return type is ``Dict[str, Any]``, not ``Dict[str, List[str]]``, because the
#: one optional ``"intent"`` key holds a bare string (``"Buying"``/``"Browsing"``),
#: not a list — every other key is still a list of strings.
SlotExtractor = Callable[[str, DialogueState], Dict[str, Any]]


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


def extract_slots(user_message: str, current_state: DialogueState) -> Dict[str, Any]:
    """No-op extractor. Returns ``{}``, so no slot ever fills and intent stays unknown.

    This is the module's default so that importing ``dialogue_state`` never
    touches the network. It is also the shape every real extractor must match:
    the tracker derives *all* of its state changes from this return value, so
    anything not reported here does not happen.

    The LLM-backed implementation is :func:`state.llm_extractor.extract_slots`,
    which does joint intent detection and slot filling in one call — the
    standard NLU pattern, rather than two separate calls for the two tasks.
    Inject it with ``DialogueStateTracker(extractor=...)``; nothing else in this
    module changes.

    Args:
        user_message: The raw customer utterance for this turn.
        current_state: State before this turn, available as context. Pass
            ``current_state.session_profile`` to the model: it cannot tell a
            refinement from a retraction without seeing what is already held.

    Returns:
        ``{attribute: [values]}`` for the attributes stated in this message, plus
        two optional control keys and one optional label:

        ``"rejected"``: values the customer just dropped, copied as they appear
        in ``current_state.session_profile``. The tracker removes each one from
        whichever slot holds it, records it as a negative term, and sets
        ``conflicts_with_previous``. To clear a whole slot, list all of its
        current values. Leave the key out when nothing was retracted.

        ``"no_preference"``: attribute *names* (not values) the customer said
        they have no preference for, which is the Boundary scenario. The tracker
        turns each one into the ``no_preference:<attribute>`` marker in
        ``rejected`` that :func:`no_preference_attributes` reads, so
        ``missing_attributes()`` stops offering it as a question. A name outside
        :data:`ASK_ATTRIBUTES` is recorded as ``other``. Declaring
        no-preference does not clear a value the slot already holds; name that
        value in ``rejected`` as well if the customer withdrew it.

        ``"intent"``: a bare string, one of :data:`INTENT_LABELS`, or left out
        when unresolved. Unlike the keys above it is not a list and it never
        touches ``session_profile`` — the tracker reads it straight onto
        :attr:`DialogueState.intent`.

        Attribute values are always arrays of strings. ``budget`` values must
        already be normalized to ``"<=45"``, ``">=45"``, or ``"~45"``; a range
        is two values. Nothing downstream parses prose prices.

        Return ``{}`` on an API error, timeout, or schema mismatch. There is no
        pattern-matching layer behind this call, so a failed turn degrades to
        "no new information for this turn": the state carries forward
        unchanged. That is deliberate — the harness counts a raised exception
        as a miss, per docs/competition_specification.md, so never raise.

        Report the call's prompt and completion token counts for the response
        ``usage`` field. Since the return type has no room for them, record them
        through :mod:`state.llm_client`'s usage meter and let the agent drain it
        once per turn.
    """
    return {}


class _SessionHistory:
    """Compact, incrementally-updated bookkeeping for one session.

    Everything :mod:`state.context_distiller` needs beyond the current state —
    when each currently-held value first appeared, how many times an attribute
    was revised, where a rejected value came from, which turns had an
    override — kept as a handful of small dicts sized to the number of
    distinct values and attributes ever touched, not to the number of turns.
    Updated once per turn in :meth:`DialogueStateTracker.update`; nothing ever
    replays a log to rebuild it, because nothing keeps one.
    """

    def __init__(self) -> None:
        self.value_first_seen: Dict[Tuple[str, str], int] = {}
        self.attribute_last_touched: Dict[str, int] = {}
        self.attribute_revisions: Dict[str, int] = {}
        self.rejection_origin: Dict[str, Tuple[str, int]] = {}
        self.override_turns: List[int] = []
        self.mentioned_values: set = set()
        self.last_turn_attributes: List[str] = []
        self.turns_observed: int = 0


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
        self._histories: Dict[str, _SessionHistory] = {}

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
            previous_ask_attribute="",
            conflicts_with_previous=False,
            intent=None,
        )
        self._states[state.session_id] = state
        self._histories[state.session_id] = _SessionHistory()
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

    def record_ask(self, state: DialogueState, ask_attribute: Optional[str]) -> DialogueState:
        """Store the ``ask_attribute`` being sent this turn, for the next turn.

        Call it with whatever goes into the response's ``ask_attribute`` field,
        including ``None`` when asking nothing — passing ``None`` clears the
        record, so a stale attribute cannot mislead the next turn's extractor.

        Pairs with :meth:`record_recommendations`: both save what the agent is
        about to send so the following turn can interpret the reply against it.

        Args:
            state: The state for this turn, mutated in place and returned.
            ask_attribute: A member of :data:`ASK_ATTRIBUTES`, or ``None``.
                Anything else is stored as ``""``, since a value outside the
                contract's enum could not have been asked.
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
        """Fold one utterance into the state and return a new state object.

        ``current_state`` is never changed.

        Args:
            user_message: Raw customer utterance.
            current_state: State from ``reset()`` or the previous ``update()``.
            turn: The harness's ``turn`` argument. Defaults to
                ``current_state.turn + 1``.

        Returns:
            A new :class:`DialogueState`. ``conflicts_with_previous`` and
            ``intent`` both cover this turn only.

        Every change below comes from the extractor's return value. This method
        never inspects ``user_message`` itself, so an extractor that returns
        ``{}`` — including on an API failure — means "no new information this
        turn" and the state carries forward untouched, except ``intent``, which
        resets to ``None`` rather than carrying the previous turn's label.
        """
        state = current_state.copy()
        state.turn = current_state.turn + 1 if turn is None else int(turn)
        state.conflicts_with_previous = False
        # Attributes this turn actually replaced or retracted a held value on,
        # for the revision count in _update_history. Not a general debug trace —
        # nothing else reads this, so it only tracks what that one method needs.
        revised_attributes: set = set()
        message = user_message or ""

        extracted = dict(self.extractor(message, current_state) or {})

        # Intent is read from the same joint call as the slots, but it is not a
        # slot: it never enters session_profile and it is not sticky like a
        # constraint would be — a failed or empty turn means "unknown this
        # turn", not "still whatever it was last turn".
        raw_intent = extracted.pop("intent", None)
        state.intent = raw_intent if raw_intent in INTENT_LABELS else None

        # (a) Retractions named by the extractor. An LLM that sees the current
        #     slots can say exactly what the customer dropped, which no pattern
        #     could work out from wording alone.
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
                    revised_attributes.add(attribute)
            displaced.add(value.lower())
            if value not in state.session_profile["rejected"]:
                state.session_profile["rejected"].append(value)
        if retracted:
            state.conflicts_with_previous = True

        # (b) No-preference declarations, the Boundary scenario. The extractor
        #     names the *attribute*; the marker convention and its spelling stay
        #     here so `no_preference_attributes` has a single writer.
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

        # (c) Apply new values, moving whatever they contradict to `rejected`.
        #     The only conflict test left is structural: a second, different
        #     value in a slot that holds exactly one. It reads extracted values,
        #     never the utterance.
        for attribute, raw_values in extracted.items():
            if attribute not in ASK_ATTRIBUTES:
                continue
            # Model output is untrusted: a bare string where a list belongs must
            # not be iterated character by character.
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
                # A value dropped earlier this turn is not taken back: "forget
                # the running shoes, I want hiking boots" only mentions
                # "running" to retract it. Keeps slots and `rejected` disjoint.
                if value.lower() in displaced:
                    continue
                _add(state.session_profile, attribute, value)

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

        One before/after diff of ``session_profile``, done once as the turn
        happens, in place of replaying a growing log of full snapshots later.
        Same information context_distiller needs, computed at O(attributes)
        per turn instead of O(turns) every time distill() is called.
        """
        history.turns_observed += 1
        touched: set = set()
        for attribute in ASK_ATTRIBUTES:
            before = {value.lower() for value in current_state.session_profile.get(attribute, [])}
            after = {value.lower() for value in state.session_profile.get(attribute, [])}
            for folded in after - before:
                history.value_first_seen.setdefault((attribute, folded), state.turn)
                history.mentioned_values.add(folded)
                history.attribute_last_touched[attribute] = state.turn
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
        """A compact, JSON-serializable view of one session's history so far.

        This is what :mod:`state.context_distiller` reads instead of a
        transition log: a handful of small dicts sized to the number of
        distinct values and attributes ever touched, not to turns times a full
        state snapshot each.
        """
        history = self._histories.get(str(session_id)) or _SessionHistory()
        value_first_seen: Dict[str, Dict[str, int]] = {}
        for (attribute, folded), seen_turn in history.value_first_seen.items():
            value_first_seen.setdefault(attribute, {})[folded] = seen_turn
        return {
            "value_first_seen": value_first_seen,
            "attribute_last_touched": dict(history.attribute_last_touched),
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
    """True when the incoming values add nothing new, so there is no conflict."""
    known = {value.lower() for value in existing}
    return all(value.lower() in known for value in incoming)
