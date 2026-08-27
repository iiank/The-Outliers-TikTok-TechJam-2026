# Dialogue State Tracker (`state.dialogue_state`)

Turns the customer's messages into one small state object that the intent router,
retrieval, and re-ranking modules can read. The object is plain data, so it is
easy to pass around and easy to print.

- Module: [src/state/dialogue_state.py](dialogue_state.py)
- Standard library only, like the starter agent. No LLM calls yet.

**The interface below is stable, so you can build against it.**

> **Slot extraction is not written yet.** `extract_slots` is an empty stub that
> returns nothing, so `session_profile` stays empty on every turn no matter what
> the customer says. The schema, the override flag, `rejected`, and the
> transition log all work now. Treat the examples below as the shape you will get
> once the extractor is implemented, and see the numbered TODO list in
> `extract_slots` for what that involves. There are no tests for this module at
> the moment.

---

## 1. State schema

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | `str` | The session id from the harness, so a state object makes sense on its own in logs. |
| `turn` | `int` | Mirrors the `turn` argument the harness passes to `respond()` (1 to 10). `0` means reset happened and no turn has run yet. |
| `session_profile` | `dict[str, list[str]]` | The slots, filled up turn by turn. Keys are **exactly** the `ask_attribute` enum from `docs/agent_api_contract.json`: `category, material, color, size, style, brand, budget, feature, use_case, other`, plus `rejected`. |
| `user_profile` | `dict` | The anonymized profile given at `reset()`. This is read-only long-term context, and this module never changes it. |
| `previous_top_10` | `list[str]` | The `parent_asin` values shown on the *previous* turn. Empty at session start. |
| `conflicts_with_previous` | `bool` | True when this turn's message contradicted what we already had. It is recalculated every turn, so it does not stay True once the conflict turn is over. |

What is always true about `session_profile`:

- Every key is always there, and every value is always a list. You never get
  `None` and never a bare string, so you do not need `.get()` guards.
- Values keep the order they were first seen. Duplicates are removed, ignoring
  case and ignoring shorter phrases already covered by a longer one, so
  `"leather"` is dropped when `"full-grain leather"` is present.
- `budget` values must be written as `"<=80"`, `">=25"`, or `"~60"`, so retrieval
  does not have to read prose. A range gives two values, so
  `"at least $25 but no more than $60"` becomes `[">=25", "<=60"]` in no
  particular order. This is now the extractor's job, so the stub does not do it
  yet. See item 3 of its TODO list.
- `rejected` holds two kinds of strings:
  - `"no_preference:<attribute>"` means the customer said they have no
    preference for that attribute (the Boundary scenario). Do not ask about it
    again. Helper: `no_preference_attributes(session_profile)`.
  - anything else is a value the customer moved away from, so treat it as a
    **negative term for retrieval**.
- The same value never appears in a slot and in `rejected` at the same time.

### Example filled state

The target shape, shown for a session that asked for running shoes and then
switched to hiking boots on turn 4. Build your code against this. The stub
extractor cannot produce it yet, so today every slot list comes back empty:

```json
{
  "session_id": "demo-session",
  "turn": 4,
  "session_profile": {
    "category": ["hiking boots"],
    "material": ["full-grain leather"],
    "color": ["black"],
    "size": [],
    "style": [],
    "brand": [],
    "budget": ["<=80"],
    "feature": ["breathable mesh upper"],
    "use_case": ["hiking"],
    "other": [],
    "rejected": ["women's running shoes", "~60", "running"]
  },
  "user_profile": {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability; ratings are usually positive."
  },
  "previous_top_10": ["B000000004"],
  "conflicts_with_previous": true
}
```

### Reading it from another module

`DialogueState` is a dataclass, and `to_dict()` gives you plain
dicts, lists, strings, numbers, and booleans. Convert once at the top of your
function and then work with plain data. Nothing you read needs an import from
this module:

```python
state_dict = state.to_dict()          # JSON-serializable, and a deep copy
terms = state_dict["session_profile"]["category"] + state_dict["session_profile"]["feature"]
```

If you ever need to go the other way, `DialogueState.from_dict(payload)` rebuilds
the object.

Optional helpers, if they save you code:

| Call | Returns |
| --- | --- |
| `state.filled_attributes()` | Slots that hold at least one value. |
| `state.missing_attributes()` | Empty slots, minus the ones the customer declined. Every item is a legal `ask_attribute` value, so it plugs straight into a question-picking policy. |
| `state.query_terms()` | All positive slot values in one flat list. A cheap starting query for retrieval. |
| `no_preference_attributes(session_profile)` | Attributes the customer declined. Takes a **plain dict**. |

---

## 2. What `update()` promises, and what it does not promise yet

```python
update(user_message: str, current_state: DialogueState, turn: int | None = None) -> DialogueState
```

**What it promises now**

- It returns a **new** state object and never changes `current_state`, so you can
  keep the old one and compare.
- `turn` is set from the value you pass, or `current_state.turn + 1` if you pass
  nothing.
- `user_profile` is carried across untouched.
- The `session_profile` rules listed in section 1 still hold after the update, so
  you never have to re-check the shape of the slots.
- Every change is logged. `get_transition_log()` returns a JSON-serializable
  list of `{turn, old_state, new_state, trigger_reason}`, oldest first, with a
  before and after snapshot for debugging.
- Two kinds of reply are still handled without the extractor, because they carry
  no constraint to pull out: `"I don't have a preference for X"` records
  `no_preference:X` in `rejected`, and `"those options are not quite right"` is
  logged as `no_new_information`.
- `conflicts_with_previous` becomes True in three cases:
  1. the extractor named retracted values in its `rejected` key, which is the
     precise path and the one the LLM will use;
  2. the message contains a negation or override word (`actually, instead, no,
     not, forget, ignore, nevermind, changed my mind`, and similar), which is the
     fallback for when the extractor named nothing;
  3. a new value clashes with a slot that already holds one and only takes one
     (`category`, `budget`, `size`).

  In every case the old values move to `rejected` before the new ones go in. When
  the extractor does name retractions, the tracker trusts that list and does not
  also clear slots the extractor chose to keep, so a keyword hit cannot undo a
  precise decision.

**What it does not promise yet**

- **No slot extraction at all.** `extract_slots` returns nothing, so every slot
  stays empty and `state.query_terms()` is always an empty list. The clash half
  of override detection also cannot fire, because a clash needs a new value to
  compare against. This is the next piece of work.
- **Override detection reads words, not meaning, until the extractor exists.**
  The precise path is built and waiting, but with no extractor there is nothing
  to name retractions, so today only the keyword fallback runs: it sets the flag
  and clears nothing, because a keyword cannot say which earlier slot the
  customer meant. So read `conflicts_with_previous` and lower your trust in the
  older slots yourself. Also note that a bare `no` or `not` will sometimes set
  the flag when nothing was really retracted.
- **A message that does two things at once loses one of them.** The two replies
  handled without the extractor return early, so `"I have no preference for
  color, but I do need cotton"` records the declined attribute and drops the
  cotton. The simulator sends these as standalone lines, so it does not bite on
  the public set, but it will matter if the organizer adds paraphrasing.
- **Retracted values are matched by whole words, not exactly.** If the extractor
  retracts `"black"`, any slot value containing the word black goes too, so
  `"black cotton trim"` in `feature` is dropped as well. This is deliberate, so
  that retracting `"running"` also clears `"women's running shoes"`, but it can
  over-reach. Have the extractor copy values as they appear in `session_profile`.
- No handling of phrases like "the second one". No cross-slot consistency checks.
  No confidence scores.
- Nothing is saved to disk, and the per-session cache lives in memory only.

**The one place to swap.** All natural language reading happens in a single
function:

```python
def extract_slots(user_message: str, current_state: DialogueState) -> dict[str, list[str]]
```

It must return `{attribute: [values]}` for the attributes stated in this message,
using keys from `ASK_ATTRIBUTES`, plus one optional extra key:

- `"rejected"`: the values the customer just dropped, copied as they appear in
  `current_state.session_profile`. The tracker removes each one from whichever
  slot holds it, records it as a negative term, and sets
  `conflicts_with_previous`. To clear a whole slot, list all of its current
  values. Leave the key out when nothing was retracted, and the tracker falls
  back to its own keyword check.

Do not emit `no_preference:<attribute>` markers. Those stay the tracker's job.

Write a function with that signature and hand it in. No other file changes:

```python
tracker = DialogueStateTracker(extractor=my_llm_extractor)
```

A good schema for the call is an object whose properties are exactly
`ASK_ATTRIBUTES` plus `rejected`, each one an array of strings. Pass the current
`session_profile` in as context, since the model cannot fill `rejected`
correctly without seeing what is currently held. The numbered TODO list in the
stub's docstring covers the rest, including budget formatting, what to do on an
API error, and token reporting.

Malformed output is tolerated rather than fatal: unknown keys are ignored, a bare
string is accepted where a list was expected, and empty values are dropped
instead of matching every slot.

---

## 3. How to call it

```python
from state.dialogue_state import DialogueStateTracker

tracker = DialogueStateTracker()                                  # once, in Agent.__init__
state = tracker.reset(session_id, user_profile)                   # in Agent.reset
state = tracker.update(user_message, state, turn=turn)            # in Agent.respond
```

Inside `Agent.respond` you only get a `session_id`, so either hold the state
yourself or use the tracker's cache:

```python
state = tracker.update(user_message, tracker.get_state(session_id), turn=turn)
# ... router, retrieval, and rerank read state.to_dict() ...
tracker.record_recommendations(state, [rec["parent_asin"] for rec in recommendations])
```

`record_recommendations` removes duplicates, keeps the first 10, and makes that
list the next turn's `previous_top_10`. `get_state` raises `KeyError` if `reset`
was never called for that session, which is the same rule the starter agent
follows.

To debug a session:

```python
for entry in tracker.get_transition_log(session_id):
    print(entry["turn"], entry["trigger_reason"])
```

With the extractor written, the reasons read like this:

```text
1 slot_filled:feature, slot_filled:category, slot_filled:use_case
2 slot_filled:color, slot_filled:budget
3 negation_marker, slot_filled:material
4 negation_marker, slot_replaced:category, slot_replaced:budget
```

With the current stub, every turn logs `no_slots_extracted` instead, apart from
the two reply types listed in section 2.
