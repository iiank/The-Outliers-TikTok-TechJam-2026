"""Personalized Context Distillation 

Reads the accumulated dialogue history and derives what the raw state cannot
express: how *durable* each constraint is, what the customer has moved away
from, and how they behave as a shopper. Nothing here calls a model. Every field
is computed from two things this repo already produces —
:meth:`state.dialogue_state.DialogueStateTracker.get_history_summary` and the
current :class:`state.dialogue_state.DialogueState`.

    from state.context_distiller import distill

    context = distill(tracker.get_history_summary(session_id), state)
    ranker_context = context["short_term"]     # -> Pillar I semantic ranking
    persist(context["session_summary"]["carry_forward"])

Output is plain dicts, lists, strings, numbers, and booleans — the same contract
:meth:`DialogueState.to_dict` holds itself to, so it is
:func:`json.dumps`-safe and needs no import to read.

Why derive rather than copy
===========================

``session_profile`` says *what* the customer asked for. It cannot say which
constraint to trust when two of them fight over the same product, because it
holds no history: a value stated once on turn 6 and a value that has survived
five turns of questioning look identical in it. The history summary holds that
history — but as a handful of small dicts the tracker updates incrementally,
one before/after diff per turn as it happens, not a full state snapshot kept
per turn and replayed later. This module is the compression step between
``session_profile`` and a ranking prompt, and everything it emits is a
*derived* quantity, not a copy.

Schema
======

Top level::

    {
      "schema_version": 1,
      "session_id": str,
      "turn": int,             # turn the distillation describes
      "turns_observed": int,   # log entries seen; < turn if a session was resumed
      "short_term": {...},
      "session_summary": {...}
    }

``schema_version`` is here so a consumer can refuse a shape it does not know;
bump it on any breaking change. ``turns_observed`` is separate from ``turn``
so a caller passing a summary from a resumed or filtered session can tell it
is incomplete.

``short_term`` — compressed current session, for the semantic ranker
--------------------------------------------------------------------

Sized to be pasted into an LLM ranking prompt, so it stays flat and short.

``constraints``: list of one entry per filled attribute, sorted alphabetically
by attribute for a deterministic order, each::

    {"attribute": str,            # a legal ask_attribute value
     "values": [str],             # as currently held
     "first_seen_turn": int,      # oldest value: when the slot was first filled
     "last_touched_turn": int,    # newest value: when it last gained content
     "turns_held": int,
     "revisions": int}            # times this attribute was replaced or retracted

Every field here is a plain fact read off the history summary, not a score:
no field is weighted, combined, or ranked against another. A consumer that
wants to prioritize one constraint over another — e.g. "trust whichever has
survived more turns and been revised less" — computes that itself from
``first_seen_turn``, ``turns_held``, and ``revisions``, using whatever
trade-off its own ranking logic needs. This module does not guess at that
trade-off on the ranker's behalf.

Both turn fields are kept alongside it so a consumer that wants a different
trade-off can compute its own weight instead of reverse-engineering this one.

``avoid``: negative terms with provenance, each::

    {"value": str, "attribute": str|None, "dropped_turn": int|None}

Raw ``rejected`` is a flat list that mixes withdrawn values with
``no_preference:`` markers and says nothing about where a value came from. Here
the markers are gone, and each term carries the slot it vacated, so a ranker can
penalize "black" as a *colour* without also penalizing a product whose brand
name contains it.

``unstable_attributes``: attributes revised at least once — the raw
``revisions`` count from ``constraints``, filtered to non-zero, so a consumer
that only wants "has this customer changed their mind on X" does not have to
scan the full constraint list for it.

``focus_attributes``: attributes that changed on the most recent turn. This is
the "what just happened" signal; on an override turn it is exactly what should
dominate the re-rank.

``digest``: a single line of natural language assembled from the fields above,
so a prompt can interpolate one string instead of walking the structure. Purely
a convenience view — it adds no information.

``session_summary`` — this session's read on the customer, not cross-session memory
------------------------------------------------------------------------------------

Named for what it actually is, not what a "long-term memory" section usually
implies: ``docs/agent_api_contract.json`` gives ``reset_request`` a
``session_id`` and an already-computed ``user_profile``, with no identifier
that links two sessions for the same customer. Nothing here can be written out
and read back on a future session, so none of it persists past this session —
it is a judgement about the session's arc so far, for the *rest of this
session's* turns, not a store of anything durable beyond that.

``override_turns``: turns where ``conflicts_with_previous`` was set. Matches the
15% Intent Override scenario in docs/competition_specification.md, and gives a
failure analysis the turn index without re-reading the log.

``profile_corroboration``: the ``preference_tags`` from the anonymized
``user_profile``, split into ``confirmed`` (echoed by something the customer
said this session) and ``unobserved``. The profile is a prior; this says which
parts of it this session actually supports. Confirmed tags are safe to weight
up, unobserved ones are not — which is the safe-personalization line the spec
draws around the aggregate profile.

``carry_forward``: the currently-held state, shaped for a ranker to bias on for
the rest of this session — ``prefer`` (every value currently in a slot),
``avoid`` (current negative terms), ``indifferent_attributes``, and
``confirmed_tags``. Collected in one place so a caller reads one object instead
of re-deriving "what does this customer currently want" from ``session_profile``
and ``rejected`` separately.

Standard library only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set

from .dialogue_state import (
    ASK_ATTRIBUTES,
    DialogueState,
    no_preference_attributes,
)

__all__ = ["SCHEMA_VERSION", "distill"]

#: Bump on any breaking change to the emitted shape.
SCHEMA_VERSION = 4

_MARKER_PREFIX = "no_preference:"

#: Words too generic to corroborate a preference tag.
_TAG_STOPWORDS = frozenset({"and", "for", "of", "the", "with", "a", "an", "to", "in", "on"})


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def distill(
    history_summary: Mapping[str, Any],
    state: DialogueState,
) -> Dict[str, Any]:
    """Derive the short-term and long-term context for one session.

    Args:
        history_summary: From ``DialogueStateTracker.get_history_summary()``:
            ``value_first_seen``, ``attribute_last_touched``,
            ``attribute_revisions``, ``rejection_origin``, ``override_turns``,
            ``mentioned_values``, ``last_turn_attributes``, ``turns_observed``.
            An empty mapping is fine and yields an empty-but-shaped result.
        state: The current dialogue state.

    Returns:
        A JSON-serializable dict in the schema documented on this module. Safe
        to call every turn: every field here is a plain lookup or a pass over
        the ten attributes, not a scan over turn history.
    """
    summary = history_summary if isinstance(history_summary, Mapping) else {}

    profile = state.session_profile
    current_turn = state.turn
    declined = no_preference_attributes(profile)
    negatives = [
        value
        for value in profile.get("rejected", [])
        if not str(value).startswith(_MARKER_PREFIX)
    ]

    value_first_seen = summary.get("value_first_seen") or {}
    attribute_revisions = summary.get("attribute_revisions") or {}
    rejection_origin = summary.get("rejection_origin") or {}

    constraints = _constraints(profile, value_first_seen, attribute_revisions, current_turn)
    unstable = sorted(
        attribute for attribute, count in attribute_revisions.items() if count
    )
    focus = sorted(summary.get("last_turn_attributes") or [])
    avoid = _avoid(negatives, rejection_origin)

    short_term: Dict[str, Any] = {
        "constraints": constraints,
        "avoid": avoid,
        "unstable_attributes": unstable,
        "focus_attributes": focus,
        "digest": "",
    }
    short_term["digest"] = _digest(short_term, declined)

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.session_id,
        "turn": current_turn,
        "turns_observed": int(summary.get("turns_observed") or 0),
        "short_term": short_term,
        "session_summary": _session_summary(state, summary, negatives, declined, constraints),
    }


# --------------------------------------------------------------------------
# short_term
# --------------------------------------------------------------------------


def _constraints(
    profile: Mapping[str, Any],
    value_first_seen: Mapping[str, Mapping[str, int]],
    attribute_revisions: Mapping[str, int],
    current_turn: int,
) -> List[Dict[str, Any]]:
    """Build the constraint list, sorted deterministically by attribute name.

    An attribute filled before the tracker ever recorded a turn has no
    first-seen entry to work from; it is treated as first seen on the current
    turn, which costs it the survival bonus rather than inventing history.
    """
    constraints: List[Dict[str, Any]] = []
    for attribute in ASK_ATTRIBUTES:
        values = list(profile.get(attribute) or [])
        if not values:
            continue
        seen_for_attribute = value_first_seen.get(attribute) or {}
        turns = [
            seen_for_attribute[value.lower()]
            for value in values
            if value.lower() in seen_for_attribute
        ]
        # Oldest value dates the constraint; newest dates its last change. An
        # attribute can be both long-held and freshly extended, and collapsing
        # the two would hide a value added this turn behind an old sibling.
        first_seen_turn = min(turns) if turns else current_turn
        last_touched_turn = max(turns) if turns else current_turn
        revisions = int(attribute_revisions.get(attribute) or 0)
        constraints.append(
            {
                "attribute": attribute,
                "values": values,
                "first_seen_turn": first_seen_turn,
                "last_touched_turn": last_touched_turn,
                "turns_held": max(1, current_turn - first_seen_turn + 1),
                "revisions": revisions,
            }
        )
    # Alphabetical by attribute: deterministic, and it makes no claim about
    # which constraint the ranker should trust more. That judgment is the
    # ranker's to make from first_seen_turn/turns_held/revisions, not this
    # module's to pre-decide.
    constraints.sort(key=lambda item: item["attribute"])
    return constraints


def _avoid(
    negatives: Sequence[str],
    rejection_origin: Mapping[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    """Attach the vacated slot and the turn to each negative term.

    A term the extractor named but that never sat in a slot gets ``None`` for
    both, which is honest: it is still a negative signal, just an unattributed
    one.
    """
    avoid: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for value in negatives:
        text = str(value).strip()
        folded = text.lower()
        if not text or folded in seen:
            continue
        seen.add(folded)
        origin = rejection_origin.get(folded)
        attribute, turn = tuple(origin) if origin else (None, None)
        avoid.append({"value": text, "attribute": attribute, "dropped_turn": turn})
    return avoid


def _digest(short_term: Mapping[str, Any], declined: Sequence[str]) -> str:
    """One prompt-ready line. Adds no information the structure lacks."""
    parts: List[str] = [
        "{attribute}={values}".format(attribute=item["attribute"], values="/".join(item["values"]))
        for item in short_term["constraints"]
    ]
    line = "Wants: " + ("; ".join(parts) if parts else "nothing stated yet")
    avoid = [str(item["value"]) for item in short_term["avoid"]]
    if avoid:
        line += ". Avoid: " + ", ".join(avoid)
    if declined:
        line += ". No preference on: " + ", ".join(declined)
    if short_term["focus_attributes"]:
        line += ". Just changed: " + ", ".join(short_term["focus_attributes"])
    return line + "."


# --------------------------------------------------------------------------
# session_summary
# --------------------------------------------------------------------------


def _session_summary(
    state: DialogueState,
    summary: Mapping[str, Any],
    negatives: Sequence[str],
    declined: Sequence[str],
    constraints: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """This session's override history, profile corroboration, and carry-forward bias."""
    profile = state.session_profile
    mentioned_values = summary.get("mentioned_values") or []
    corroboration = _corroborate(state.user_profile, profile, mentioned_values)
    prefer = [value for attribute in ASK_ATTRIBUTES for value in profile.get(attribute) or []]

    return {
        "override_turns": sorted(set(summary.get("override_turns") or [])),
        "profile_corroboration": corroboration,
        "carry_forward": {
            "prefer": prefer,
            "avoid": list(negatives),
            "indifferent_attributes": list(declined),
            "confirmed_tags": corroboration["confirmed"],
            # Repeating the top constraint's attribute is cheap and tells the
            # rest of this session which slot mattered most so far.
            "led_with": constraints[0]["attribute"] if constraints else None,
        },
    }


def _corroborate(
    user_profile: Mapping[str, Any],
    session_profile: Mapping[str, Any],
    mentioned_values: Sequence[str],
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
    for folded in mentioned_values:
        vocabulary |= _tokens(str(folded))

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
