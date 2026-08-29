"""Personalized Context Distillation (Pillar III, Dynamic Context Programming).

Reads the accumulated dialogue history and derives what the raw state cannot
express: how *durable* each constraint is, what the customer has moved away
from, and how they behave as a shopper. Nothing here calls a model. Every field
is computed from two things this repo already produces —
:meth:`state.dialogue_state.DialogueStateTracker.get_transition_log` and the
current :class:`state.dialogue_state.DialogueState`.

    from state.context_distiller import distill

    context = distill(tracker.get_transition_log(session_id), state)
    ranker_context = context["short_term"]     # -> Pillar I semantic ranking
    persist(context["long_term"]["carry_forward"])

Output is plain dicts, lists, strings, numbers, and booleans — the same contract
:meth:`DialogueState.to_dict` holds itself to, so it is
:func:`json.dumps`-safe and needs no import to read.

Why derive rather than copy
===========================

``session_profile`` says *what* the customer asked for. It cannot say which
constraint to trust when two of them fight over the same product, because it
holds no history: a value stated once on turn 6 and a value that has survived
five turns of questioning look identical in it. The transition log holds that
history but is far too verbose to put in a ranking prompt — it carries a full
before-and-after state snapshot per turn. This module is the compression step
between them, and everything it emits is a *derived* quantity, not a copy.

Schema
======

Top level::

    {
      "schema_version": 1,
      "session_id": str,
      "turn": int,             # turn the distillation describes
      "turns_observed": int,   # log entries seen; < turn if a session was resumed
      "short_term": {...},
      "long_term": {...}
    }

``schema_version`` is here so a consumer can refuse a shape it does not know;
bump it on any breaking change. ``turns_observed`` is separate from ``turn``
because the confidence arithmetic below is only as good as the history it saw,
and a caller passing a filtered or truncated log should be able to tell.

``short_term`` — compressed current session, for the semantic ranker
--------------------------------------------------------------------

Sized to be pasted into an LLM ranking prompt, so it stays flat and short.

``constraints``: list of one entry per filled attribute, **sorted by
``confidence`` descending**, each::

    {"attribute": str,            # a legal ask_attribute value
     "values": [str],             # as currently held
     "first_seen_turn": int,      # oldest value: when the slot was first filled
     "last_touched_turn": int,    # newest value: when it last gained content
     "turns_held": int,
     "revisions": int,            # times this attribute was replaced or retracted
     "confidence": float}        # 0.15-0.95

The sort order *is* the payload: it hands the ranker a priority sequence, which
is the thing the raw state is structurally incapable of expressing (a dict of
slots has no ordering, and every slot looks equally certain). ``confidence``
combines the three signals the log makes available:

* **Survival.** A constraint the customer kept through later turns of
  questioning has been implicitly re-confirmed. ``+0.10`` per turn survived,
  capped at three turns — past that, extra age says little.
* **Instability.** An attribute the customer has already rewritten is a poor
  thing to rank on. ``-0.15`` per revision, capped at three.
* **Freshness.** ``+0.10`` when the attribute gained a value on the current
  turn. Keyed on ``last_touched_turn``, not ``first_seen_turn``, so adding a
  second colour to a slot filled on turn 2 still counts as current activity.

Base 0.55, clamped to ``[0.15, 0.95]``: never certain, never dismissible.

The weights are deliberately coarse. They encode an *ordering*, not a calibrated
probability, and with only three inputs and two caps they produce a handful of
bands rather than distinct scores — several constraints tying is normal, and
ties break alphabetically by attribute so the same state always distils to the
same bytes. Treat ``confidence`` as "which of these should win a conflict",
never as "how likely is this to be true".

Both turn fields are kept alongside it so a consumer that wants a different
trade-off can compute its own weight instead of reverse-engineering this one.

``avoid``: negative terms with provenance, each::

    {"value": str, "attribute": str|None, "dropped_turn": int|None}

Raw ``rejected`` is a flat list that mixes withdrawn values with
``no_preference:`` markers and says nothing about where a value came from. Here
the markers are gone, and each term carries the slot it vacated, so a ranker can
penalize "black" as a *colour* without also penalizing a product whose brand
name contains it.

``declined_attributes``: attribute names from the ``no_preference:`` markers.
Not a duplicate of the raw markers: the marker prefix is a storage detail, and
this list is already parsed and validated against ``ASK_ATTRIBUTES``.

``open_attributes``: still worth asking about — ``state.missing_attributes()``,
carried through so a consumer has the question surface and the ranking context
in one object instead of two.

``unstable_attributes``: attributes revised at least once. The ranker should
weight these below their ``confidence`` alone would suggest, because the
customer has already demonstrated they will move.

``focus_attributes``: attributes that changed on the most recent turn. This is
the "what just happened" signal; on an override turn it is exactly what should
dominate the re-rank.

``digest``: a single line of natural language assembled from the fields above,
so a prompt can interpolate one string instead of walking the structure. Purely
a convenience view — it adds no information.

``long_term`` — cross-turn patterns, meant to outlive the session
-----------------------------------------------------------------

Everything here is a *pattern over* the log rather than a fact from one turn,
which is what makes it worth persisting against a returning customer.

``stable_preferences``: values still held at the end, with ``first_seen_turn``
and ``turns_held``. These survived the whole session, so they are the strongest
candidates for a durable taste profile.

``abandoned_preferences``: values the customer withdrew, with ``dropped_turn``.
Detected by presence in the final ``rejected`` list rather than by disappearing
from a slot, which distinguishes a reversal from a mere refinement — the tracker
also removes a short value when a longer one supersedes it ("leather" giving way
to "full-grain leather"), and that is not an abandonment.

``refinement_count``: how many times a value was superseded by a more specific
one. High refinement with low ``revision_count`` is a customer who knows what
they want and is getting more precise; the inverse is a customer changing their
mind. The two are indistinguishable in the raw state.

``revision_profile``: per attribute, ``{fills, replacements, retractions}``,
counted from the log's ``trigger_reason`` strings. Which attributes a customer
churns on is a stable personal trait and a direct input to question selection —
do not spend a turn asking about the slot they keep rewriting.

``volatility``: revisions over total slot events, ``0.0``-``1.0``. One number
for "how much does this shopper change their mind", cheap to threshold on.

``decisiveness``: distinct values established per turn observed. Pairs with
``volatility``: high decisiveness plus low volatility means front-load
recommendations; the inverse means keep clarifying.

``override_turns``: turns where ``conflicts_with_previous`` was set. Matches the
15% Intent Override scenario in docs/competition_specification.md, and gives a
failure analysis the turn index without re-reading the log.

``profile_corroboration``: the ``preference_tags`` from the anonymized
``user_profile``, split into ``confirmed`` (echoed by something the customer
said this session) and ``unobserved``. The profile is a prior; this says which
parts of it this session actually supports. Confirmed tags are safe to weight
up, unobserved ones are not — which is the safe-personalization line the spec
draws around the aggregate profile.

``carry_forward``: the subset a caller should persist beyond the session, split
into ``prefer`` / ``avoid`` / ``indifferent_attributes`` / ``confirmed_tags``.
Collected in one place so persistence is one write and does not have to re-apply
this module's judgement about what is durable.

Standard library only.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .dialogue_state import (
    ASK_ATTRIBUTES,
    DialogueState,
    no_preference_attributes,
)

__all__ = ["SCHEMA_VERSION", "distill", "distill_session"]

#: Bump on any breaking change to the emitted shape.
SCHEMA_VERSION = 1

#: Confidence arithmetic. Named so the docstring's explanation and the code
#: cannot drift apart.
_CONFIDENCE_BASE = 0.55
_SURVIVAL_BONUS = 0.10
_SURVIVAL_CAP = 3
_REVISION_PENALTY = 0.15
_REVISION_CAP = 3
_FRESHNESS_BONUS = 0.10
_CONFIDENCE_FLOOR = 0.15
_CONFIDENCE_CEILING = 0.95

_MARKER_PREFIX = "no_preference:"

#: Words too generic to corroborate a preference tag.
_TAG_STOPWORDS = frozenset({"and", "for", "of", "the", "with", "a", "an", "to", "in", "on"})


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def distill_session(tracker: Any, state: DialogueState) -> Dict[str, Any]:
    """Convenience wrapper: pull the session's log off a tracker, then distill.

    Args:
        tracker: A :class:`state.dialogue_state.DialogueStateTracker`. Only
            ``get_transition_log`` is used, so any object with that method works.
        state: The current state, normally the one ``update()`` just returned.
    """
    return distill(tracker.get_transition_log(state.session_id), state)


def distill(
    transition_log: Sequence[Mapping[str, Any]],
    state: DialogueState,
) -> Dict[str, Any]:
    """Derive the short-term and long-term context for one session.

    Args:
        transition_log: Entries from ``get_transition_log()``, oldest first,
            each ``{turn, old_state, new_state, trigger_reason}``. Filter it to
            one session before passing it in; a mixed log produces meaningless
            history. An empty log is fine and yields an empty-but-shaped result.
        state: The current dialogue state.

    Returns:
        A JSON-serializable dict in the schema documented on this module. Safe
        to call every turn: it is a linear pass over the log with no I/O.
    """
    entries = [entry for entry in (transition_log or []) if isinstance(entry, Mapping)]
    history = _History(entries)

    profile = state.session_profile
    current_turn = state.turn
    declined = no_preference_attributes(profile)
    negatives = [
        value
        for value in profile.get("rejected", [])
        if not str(value).startswith(_MARKER_PREFIX)
    ]

    constraints = _constraints(profile, history, current_turn)
    unstable = sorted(
        attribute
        for attribute, counts in history.events.items()
        if counts["replacements"] or counts["retractions"]
    )
    focus = history.last_turn_attributes(current_turn)
    avoid = _avoid(negatives, history)

    short_term: Dict[str, Any] = {
        "constraints": constraints,
        "avoid": avoid,
        "declined_attributes": declined,
        "open_attributes": state.missing_attributes(),
        "unstable_attributes": unstable,
        "focus_attributes": focus,
        "digest": "",
    }
    short_term["digest"] = _digest(short_term)

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.session_id,
        "turn": current_turn,
        "turns_observed": history.turns_observed,
        "short_term": short_term,
        "long_term": _long_term(state, history, negatives, declined, constraints),
    }


# --------------------------------------------------------------------------
# History walk
# --------------------------------------------------------------------------


class _History:
    """One linear pass over the transition log.

    Everything downstream reads from this, so the log is walked exactly once no
    matter how many derived fields are asked for.

    Attributes:
        first_seen: ``(attribute, folded_value) -> turn`` a value first appeared.
        removed: ``(attribute, folded_value) -> turn`` it last left a slot.
            Entries are deleted again if the value comes back.
        display: ``(attribute, folded_value) -> value`` as last written, so
            output keeps the customer's casing.
        events: ``attribute -> {fills, replacements, retractions}``.
        declared_turn: ``attribute -> turn`` a no-preference was declared.
        override_turns: Turns where ``conflicts_with_previous`` was set.
        turns_observed: Number of usable log entries.
    """

    def __init__(self, entries: Iterable[Mapping[str, Any]]) -> None:
        self.first_seen: Dict[Tuple[str, str], int] = {}
        self.removed: Dict[Tuple[str, str], int] = {}
        self.display: Dict[Tuple[str, str], str] = {}
        self.events: Dict[str, Dict[str, int]] = {
            attribute: {"fills": 0, "replacements": 0, "retractions": 0}
            for attribute in ASK_ATTRIBUTES
        }
        self.declared_turn: Dict[str, int] = {}
        self.override_turns: List[int] = []
        self.changed_at: Dict[int, Set[str]] = {}
        self.turns_observed = 0

        for entry in entries:
            self.turns_observed += 1
            turn = _as_int(entry.get("turn"))
            old = _slots(entry.get("old_state"))
            new = _slots(entry.get("new_state"))

            self._count_reasons(entry.get("trigger_reason"), turn)
            if _truthy(entry.get("new_state"), "conflicts_with_previous"):
                self.override_turns.append(turn)

            touched = self.changed_at.setdefault(turn, set())
            for attribute in ASK_ATTRIBUTES:
                before = {value.lower(): value for value in old.get(attribute, [])}
                after = {value.lower(): value for value in new.get(attribute, [])}
                for folded, value in after.items():
                    if folded in before:
                        continue
                    pair = (attribute, folded)
                    self.display[pair] = value
                    self.first_seen.setdefault(pair, turn)
                    # Back in a slot, so no longer abandoned.
                    self.removed.pop(pair, None)
                    touched.add(attribute)
                for folded, value in before.items():
                    if folded in after:
                        continue
                    pair = (attribute, folded)
                    self.display.setdefault(pair, value)
                    self.removed[pair] = turn
                    touched.add(attribute)
            if not touched:
                self.changed_at.pop(turn, None)

    def _count_reasons(self, trigger_reason: Any, turn: int) -> None:
        """Tally ``slot_filled:``/``slot_replaced:``/``retracted:`` markers.

        The tracker already names every transition it made, so counting its
        reasons is cheaper and less error-prone than re-deriving intent from the
        state diff.
        """
        for reason in str(trigger_reason or "").split(","):
            name, _, attribute = reason.strip().partition(":")
            if name == "no_preference_declared":
                self.declared_turn.setdefault(attribute or "other", turn)
                continue
            if attribute not in self.events:
                continue
            if name == "slot_filled":
                self.events[attribute]["fills"] += 1
            elif name == "slot_replaced":
                self.events[attribute]["replacements"] += 1
            elif name == "retracted":
                self.events[attribute]["retractions"] += 1

    def revisions(self, attribute: str) -> int:
        counts = self.events.get(attribute) or {}
        return counts.get("replacements", 0) + counts.get("retractions", 0)

    def last_turn_attributes(self, current_turn: int) -> List[str]:
        """Attributes that changed on the newest turn present in the log."""
        if not self.changed_at:
            return []
        turn = current_turn if current_turn in self.changed_at else max(self.changed_at)
        return sorted(self.changed_at.get(turn, set()))


# --------------------------------------------------------------------------
# short_term
# --------------------------------------------------------------------------


def _constraints(
    profile: Mapping[str, Any],
    history: _History,
    current_turn: int,
) -> List[Dict[str, Any]]:
    """Build the confidence-ranked constraint list.

    An attribute filled before this module ever saw a log entry has no
    ``first_seen_turn`` to work from; it is treated as first seen on the current
    turn, which costs it the survival bonus rather than inventing history.
    """
    constraints: List[Dict[str, Any]] = []
    for attribute in ASK_ATTRIBUTES:
        values = list(profile.get(attribute) or [])
        if not values:
            continue
        turns = [
            history.first_seen[(attribute, value.lower())]
            for value in values
            if (attribute, value.lower()) in history.first_seen
        ]
        # Oldest value dates the constraint; newest dates its last change. An
        # attribute can be both long-held and freshly extended, and collapsing
        # the two would hide a value added this turn behind an old sibling.
        first_seen_turn = min(turns) if turns else current_turn
        last_touched_turn = max(turns) if turns else current_turn
        revisions = history.revisions(attribute)
        constraints.append(
            {
                "attribute": attribute,
                "values": values,
                "first_seen_turn": first_seen_turn,
                "last_touched_turn": last_touched_turn,
                "turns_held": max(1, current_turn - first_seen_turn + 1),
                "revisions": revisions,
                "confidence": _confidence(
                    first_seen_turn, last_touched_turn, revisions, current_turn
                ),
            }
        )
    # Stable tie-break on attribute name, so the same state always distils to
    # the same bytes — the local evaluator is deterministic and a prompt that
    # reorders between runs makes scores incomparable.
    constraints.sort(key=lambda item: (-item["confidence"], item["attribute"]))
    return constraints


def _confidence(
    first_seen_turn: int,
    last_touched_turn: int,
    revisions: int,
    current_turn: int,
) -> float:
    """Survival, instability, and freshness folded into one 0.15-0.95 weight.

    See the module docstring for why each term is shaped this way. Freshness is
    keyed on ``last_touched_turn``, so extending a long-held attribute this turn
    still registers as current activity. Rounded to two places because the
    precision is not real and a long float invites a consumer to over-read it.
    """
    survived = max(0, current_turn - first_seen_turn)
    score = _CONFIDENCE_BASE
    score += _SURVIVAL_BONUS * min(survived, _SURVIVAL_CAP)
    score -= _REVISION_PENALTY * min(revisions, _REVISION_CAP)
    if last_touched_turn >= current_turn:
        score += _FRESHNESS_BONUS
    return round(max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, score)), 2)


def _avoid(negatives: Sequence[str], history: _History) -> List[Dict[str, Any]]:
    """Attach the vacated slot and the turn to each negative term.

    A term the extractor named but that never sat in a slot gets ``None`` for
    both, which is honest: it is still a negative signal, just an unattributed
    one.
    """
    origins: Dict[str, Tuple[str, int]] = {}
    for (attribute, folded), turn in history.removed.items():
        # First removal wins, so a value that bounced reports where it started.
        origins.setdefault(folded, (attribute, turn))

    avoid: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for value in negatives:
        text = str(value).strip()
        folded = text.lower()
        if not text or folded in seen:
            continue
        seen.add(folded)
        attribute, turn = origins.get(folded, (None, None))
        avoid.append({"value": text, "attribute": attribute, "dropped_turn": turn})
    return avoid


def _digest(short_term: Mapping[str, Any]) -> str:
    """One prompt-ready line. Adds no information the structure lacks."""
    parts: List[str] = []
    for item in short_term["constraints"]:
        parts.append(
            "{attribute}={values} ({confidence:.2f})".format(
                attribute=item["attribute"],
                values="/".join(item["values"]),
                confidence=item["confidence"],
            )
        )
    line = "Wants: " + ("; ".join(parts) if parts else "nothing stated yet")
    avoid = [str(item["value"]) for item in short_term["avoid"]]
    if avoid:
        line += ". Avoid: " + ", ".join(avoid)
    if short_term["declined_attributes"]:
        line += ". No preference on: " + ", ".join(short_term["declined_attributes"])
    if short_term["focus_attributes"]:
        line += ". Just changed: " + ", ".join(short_term["focus_attributes"])
    return line + "."


# --------------------------------------------------------------------------
# long_term
# --------------------------------------------------------------------------


def _long_term(
    state: DialogueState,
    history: _History,
    negatives: Sequence[str],
    declined: Sequence[str],
    constraints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Cross-turn patterns, plus the explicit carry-forward subset."""
    profile = state.session_profile
    current_turn = state.turn
    negative_set = {str(value).strip().lower() for value in negatives}

    held: Set[Tuple[str, str]] = {
        (attribute, value.lower())
        for attribute in ASK_ATTRIBUTES
        for value in profile.get(attribute) or []
    }

    stable: List[Dict[str, Any]] = []
    for attribute in ASK_ATTRIBUTES:
        for value in profile.get(attribute) or []:
            first_seen_turn = history.first_seen.get((attribute, value.lower()), current_turn)
            stable.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "first_seen_turn": first_seen_turn,
                    "turns_held": max(1, current_turn - first_seen_turn + 1),
                }
            )

    abandoned: List[Dict[str, Any]] = []
    refinements = 0
    for (attribute, folded), turn in sorted(history.removed.items(), key=lambda kv: kv[1]):
        if (attribute, folded) in held:
            continue
        value = history.display.get((attribute, folded), folded)
        if folded in negative_set:
            abandoned.append(
                {
                    "attribute": attribute,
                    "value": value,
                    "dropped_turn": turn,
                    "turns_held": max(
                        0, turn - history.first_seen.get((attribute, folded), turn)
                    ),
                }
            )
        else:
            # Left a slot without being rejected: superseded by a more specific
            # phrase, not withdrawn.
            refinements += 1

    total_events = sum(
        counts["fills"] + counts["replacements"] + counts["retractions"]
        for counts in history.events.values()
    )
    revisions = sum(
        counts["replacements"] + counts["retractions"] for counts in history.events.values()
    )
    corroboration = _corroborate(state.user_profile, profile, history)

    return {
        "stable_preferences": stable,
        "abandoned_preferences": abandoned,
        "refinement_count": refinements,
        "revision_profile": {
            attribute: dict(counts)
            for attribute, counts in sorted(history.events.items())
            if any(counts.values())
        },
        "volatility": round(revisions / total_events, 2) if total_events else 0.0,
        "decisiveness": (
            round(len(stable) / history.turns_observed, 2) if history.turns_observed else 0.0
        ),
        "override_turns": sorted(set(history.override_turns)),
        "profile_corroboration": corroboration,
        "carry_forward": {
            "prefer": [item["value"] for item in stable],
            "avoid": [item["value"] for item in abandoned],
            "indifferent_attributes": list(declined),
            "confirmed_tags": corroboration["confirmed"],
            # Repeating the top constraint's attribute is cheap and tells a
            # future session which slot mattered most last time.
            "led_with": constraints[0]["attribute"] if constraints else None,
        },
    }


def _corroborate(
    user_profile: Mapping[str, Any],
    session_profile: Mapping[str, Any],
    history: _History,
) -> Dict[str, List[str]]:
    """Split ``preference_tags`` by whether this session's words support them.

    Token overlap rather than substring matching, so the tag ``"fit"`` is not
    confirmed by the word ``"outfit"``. Every value the session ever held is
    considered, including ones later dropped: the customer did say them, and a
    tag is about taste rather than about the current constraint set.
    """
    tags = user_profile.get("preference_tags") if isinstance(user_profile, Mapping) else None
    if not isinstance(tags, (list, tuple)):
        return {"confirmed": [], "unobserved": []}

    vocabulary: Set[str] = set()
    for attribute in ASK_ATTRIBUTES:
        for value in session_profile.get(attribute) or []:
            vocabulary |= _tokens(str(value))
    for (_, folded) in history.first_seen:
        vocabulary |= _tokens(folded)

    confirmed: List[str] = []
    unobserved: List[str] = []
    for tag in tags:
        text = str(tag).strip()
        if not text:
            continue
        tokens = _tokens(text)
        target = confirmed if tokens and tokens <= vocabulary else unobserved
        target.append(text)
    return {"confirmed": confirmed, "unobserved": unobserved}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _tokens(text: str) -> Set[str]:
    """Lowercase alphanumeric words, minus filler. No regex needed."""
    folded = "".join(char if char.isalnum() else " " for char in text.lower())
    return {word for word in folded.split() if len(word) > 1 and word not in _TAG_STOPWORDS}


def _slots(snapshot: Any) -> Dict[str, List[str]]:
    """Pull a ``session_profile`` out of a log snapshot, tolerating junk."""
    if not isinstance(snapshot, Mapping):
        return {}
    profile = snapshot.get("session_profile")
    if not isinstance(profile, Mapping):
        return {}
    return {
        key: [str(value) for value in values]
        for key, values in profile.items()
        if isinstance(values, (list, tuple))
    }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _truthy(snapshot: Any, key: str) -> bool:
    return bool(snapshot.get(key)) if isinstance(snapshot, Mapping) else False


def _self_check() -> None:  # pragma: no cover - exercised by __main__ only
    """Round-trip the output through json, so the contract cannot silently rot."""
    state = DialogueState(session_id="check", turn=1)
    json.dumps(distill([], state))
