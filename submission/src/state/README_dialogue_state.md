# Dialogue State Tracker (`state`)

Turns what the customer types into a small data object the rest of the agent
reads. One function reads the words. Everything else works from the object.


| File                                         | Job                                             | Calls an LLM |
| -------------------------------------------- | ----------------------------------------------- | ------------ |
| [dialogue_state.py](dialogue_state.py)       | Holds the state, applies changes                | no           |
| [llm_extractor.py](llm_extractor.py)         | Words to slots and Buying/Browsing, in one call | yes          |
| [context_distiller.py](context_distiller.py) | History to what we trust                        | no           |
| [llm_client.py](llm_client.py)               | Shared HTTP plumbing                            | yes          |


`llm_extractor.py` does joint intent detection and slot filling — one schema,
one request, both answers. This is the standard NLU pattern (the two tasks are
correlated: an intent's likely slots inform the slots and vice versa), not two
independent calls for two independent questions.

---



## The workflow, in order

```text
Phase 0  reset()            once per session
   |
Phase 1  extract_slots()    words  ->  {"category": ["boots"], ..., "intent": "Buying"}
   |
Phase 2  update()           slots + intent  ->  new DialogueState
   |
   +--> Phase 3  distill()          ->  {"short_term", "session_summary"}
   |
Phase 4  record_ask(), record_recommendations(), drain_usage()
```

---



### Phase 0: start the session

**What it does:** clears everything and stores the anonymized profile.


|          |                                                         |
| -------- | ------------------------------------------------------- |
| **In**   | `session_id`, `user_profile` from the harness           |
| **Does** | empties all ten slots plus `rejected`, sets `turn` to 0 |
| **Out**  | a fresh `DialogueState`                                 |


```python
tracker = DialogueStateTracker(extractor=extract_slots)   # once, in Agent.__init__
state = tracker.reset(session_id, user_profile)           # in Agent.reset
```

**Example** — this is the running example threaded through every phase below:

```python
reset("sess-42", {
    "purchase_frequency": "occasional",
    "average_prior_rating": 4.2,
    "rating_style": "generous",
    "preference_tags": ["road running", "comfort"],
    "summary": "Casual shopper, buys athletic wear a few times a year.",
})
```

```python
DialogueState(
    session_id="sess-42", turn=0,
    session_profile={"category": [], "material": [], "color": [], "size": [], "style": [],
                      "brand": [], "budget": [], "feature": [], "use_case": [], "other": [],
                      "rejected": []},
    user_profile={"purchase_frequency": "occasional", "average_prior_rating": 4.2,
                  "rating_style": "generous", "preference_tags": ["road running", "comfort"],
                  "summary": "Casual shopper, buys athletic wear a few times a year."},
    previous_top_10=[], previous_ask_attribute="",
    conflicts_with_previous=False, intent=None,
)
```

---



### Phase 1: read the words

**What it does:** the only place in the agent that reads natural language —
and the only LLM call in the whole state layer.


|          |                                                       |
| -------- | ----------------------------------------------------- |
| **In**   | the raw message, plus the slots we already hold       |
| **Does** | one LLM call with a strict JSON schema                |
| **Out**  | a `dict` of just what changed, plus `intent`, or `{}` |


The model is shown the current slots on purpose. "Boots instead" only means
something if you know what it replaces, so showing the slots is what lets the
model name the old value precisely. The same context (how many slots are
already filled) is also what it uses to judge Buying versus Browsing: a
confidently-worded opener with nothing filled in yet is usually still
Browsing, and a vague-sounding message late in a well-specified session is
usually still Buying.

```python
extract_slots("actually hiking boots instead, under $120", state)
# {"category": ["hiking boots"], "budget": ["<=120"], "rejected": ["running shoes"], "intent": "Buying"}
```

**Example** — continuing the running example, turn 1 (`state` is the fresh one from Phase 0):

```python
extract_slots("I need black waterproof hiking boots under $120", state)
# {"intent": "Buying", "category": ["hiking boots"], "color": ["black"],
#  "budget": ["<=120"], "feature": ["waterproof"]}
```

**Example** — turn 2, after `record_ask(state, "size")` set `previous_ask_attribute="size"`.
The bare reply resolves against that context instead of landing in `other`:

```python
extract_slots("US 9", state)
# {"intent": "Buying", "size": ["US 9"]}
```

Four kinds of key come back:


| Key                     | Holds                                                                  |
| ----------------------- | ---------------------------------------------------------------------- |
| the ten attribute names | new or changed values, each a list                                     |
| `rejected`              | **values** the customer dropped, copied exactly from the current slots |
| `no_preference`         | attribute **names** they refuse to constrain ("any colour is fine")    |
| `intent`                | a bare string, `"Buying"` or `"Browsing"` — not a list, and not a slot |


`rejected` is values, `no_preference` is names. That is the easy thing to mix up.

`budget` arrives already normalized as `"<=120"`, `">=25"`, or `"~60"`, so
nothing downstream reads prose prices.

`intent` is two labels, not four. The public set also labels sessions Intent
Override and Boundary, but the state already reports those without any extra
model reasoning (`conflicts_with_previous` and `no_preference_attributes()`).
An override session is still Buying or Browsing on every turn.

`{}` means nothing new and no intent resolved. A failed API call also returns
`{}`, and the two are deliberately identical. There is no keyword fallback
underneath, so a bad turn becomes "no new information" instead of a guess.

---



### Phase 2: apply it to the state

**What it does:** folds the extracted dict into a new state object.


|          |                                                                                    |
| -------- | ---------------------------------------------------------------------------------- |
| **In**   | the raw message (passed straight to Phase 1), the current state                    |
| **Does** | applies retractions, then no-preference marks, then new values, then sets `intent` |
| **Out**  | a **new** `DialogueState`, plus an incremental history update                      |


```python
state = tracker.update(user_message, tracker.get_state(session_id), turn=turn)
```

**Example** — continuing the running example, folding Phase 1's turn-1 output into the
fresh state from Phase 0:

```python
DialogueState(
    session_id="sess-42", turn=1,
    session_profile={"category": ["hiking boots"], "material": [], "color": ["black"],
                      "size": [], "style": [], "brand": [], "budget": ["<=120"],
                      "feature": ["waterproof"], "use_case": [], "other": [], "rejected": []},
    user_profile={...same as Phase 0...},
    previous_top_10=[], previous_ask_attribute="",
    conflicts_with_previous=False, intent="Buying",
)
```

In order:

1. **Retractions.** For each value in `rejected`, remove it from whatever slot
  holds it, move it into the `rejected` list, and set `conflicts_with_previous`.
2. **No preference.** Turn each attribute name into a `no_preference:<name>`
  mark, so `missing_attributes()` stops offering it for the rest of the session.
3. **New values.** Add them. If a value lands in `category`, `budget`, or `size`
  and that slot already holds something different, treat it as a replacement:
   the old value moves to `rejected` and the conflict flag is set. This is the
   safety net for when the extractor adds a value but forgets to name the one it
   replaced.
4. **Intent.** Read straight onto `state.intent`. Not sticky: a turn with no
  resolved intent sets it to `None` rather than carrying the previous label
   forward. Branch on `None`; do not read it as Browsing, since this is a label,
   not a routing decision.

`update()` never looks at the message text itself. Every change above comes from
Phase 1's return value. That is why an API failure is harmless: `{}` in means
nothing changes except `intent`, which resets to `None`.

The old state is never modified, so you can keep it and compare.

---



### Phase 3: distill the history

**What it does:** works out which constraints to trust. No LLM call.


|          |                                                                                          |
| -------- | ---------------------------------------------------------------------------------------- |
| **In**   | the tracker's incremental history summary, plus the current state                        |
| **Does** | reads a handful of small dicts the tracker already maintains — no scan over turn history |
| **Out**  | `{"short_term": ..., "session_summary": ...}`                                            |


Two very different things come back, and the split is what to read *now* versus what to
carry forward *for the rest of this session*:

- `short_term` is a snapshot of right now. It changes every turn, in lockstep with
`session_profile` — it's the same information, just compressed and ordered for a
ranking prompt instead of shaped for `update()` to write to.
- `session_summary` is a judgement about the session's *arc so far*, not this one
turn: has the customer contradicted themselves anywhere, does what they've actually
said support the profile the harness handed you, and — bundled for convenience — what
should the rest of this session keep biasing toward. It is **not** cross-session
memory: `reset_request` gives no identifier linking two sessions for the same
customer, so nothing written here could ever be read back on a future session. Despite
the name, nothing here outlives this one session.

```python
context = distill(tracker.get_history_summary(session_id), state)
```

**Example** — continuing the running example, still turn 1 (everything is freshly stated,
so every constraint has `revisions=0` and `turns_held=1`):

```python
{
  "schema_version": 4, "session_id": "sess-42", "turn": 1, "turns_observed": 1,
  "short_term": {
    "constraints": [
      {"attribute": "budget", "values": ["<=120"], "first_seen_turn": 1, "last_touched_turn": 1, "turns_held": 1, "revisions": 0},
      {"attribute": "category", "values": ["hiking boots"], "first_seen_turn": 1, "last_touched_turn": 1, "turns_held": 1, "revisions": 0},
      {"attribute": "color", "values": ["black"], "first_seen_turn": 1, "last_touched_turn": 1, "turns_held": 1, "revisions": 0},
      {"attribute": "feature", "values": ["waterproof"], "first_seen_turn": 1, "last_touched_turn": 1, "turns_held": 1, "revisions": 0}
    ],
    "avoid": [], "declined_attributes": [],
    "open_attributes": ["material", "size", "style", "brand", "use_case", "other"],
    "unstable_attributes": [],
    "focus_attributes": ["budget", "category", "color", "feature"],
    "digest": "Wants: budget=<=120; category=hiking boots; color=black; feature=waterproof. Just changed: budget, category, color, feature."
  },
  "session_summary": {
    "override_turns": [],
    "profile_corroboration": {"confirmed": [], "unobserved": ["road running", "comfort"]},
    "carry_forward": {
      "prefer": ["hiking boots", "black", "<=120", "waterproof"],
      "avoid": [], "indifferent_attributes": [], "confirmed_tags": [], "led_with": "budget"
    }
  }
}
```

`profile_corroboration` puts both tags in `unobserved` here — the customer hasn't said
anything about running or comfort yet, so the aggregate profile's prior is not yet
confirmed by this session. That would move to `confirmed` the moment either word shows
up in something the customer actually says. That is the kind of judgement `short_term`
cannot make on its own: it only knows the current slots, not whether those slots
corroborate a prior.

`short_term` fields, all recomputed from scratch every turn:


| Field                 | What it adds that the raw state cannot                                                                                                                                                                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `constraints`         | every filled slot, with `first_seen_turn`, `last_touched_turn`, `turns_held`, and `revisions` — plain facts, sorted alphabetically by attribute for a deterministic order. No score: a consumer that wants to prioritize one constraint over another computes that itself from these fields. |
| `avoid`               | each rejected value with the slot it came from and the turn it was dropped                                                                                                                                                                                                                   |
| `declined_attributes` | attributes the customer refused to constrain                                                                                                                                                                                                                                                 |
| `open_attributes`     | still worth asking about                                                                                                                                                                                                                                                                     |
| `unstable_attributes` | slots the customer has already rewritten (`revisions > 0`)                                                                                                                                                                                                                                   |
| `focus_attributes`    | what changed this turn. On an override turn, this is what to weight up.                                                                                                                                                                                                                      |
| `digest`              | all of the above as one line of text                                                                                                                                                                                                                                                         |

`declined_attributes` and `open_attributes` can look redundant at first, since a
declined attribute is already excluded from `open_attributes` (`missing_attributes()`
filters both filled *and* declined slots out). They exist separately because two
different consumers need to tell "not yet asked" apart from "customer explicitly said
they don't care," and neither list alone can distinguish those two cases for an
attribute that's simply absent from it:

- **Question selection** (Pillar II) only needs `open_attributes` — what's still fair
  game to ask. It doesn't need to know *why* something isn't askable.
- **Retrieval/ranking** needs `declined_attributes` specifically, because it changes how
  a candidate product should be scored. A product with no information on an attribute
  the customer never mentioned is a minor unknown — some risk in recommending it. A
  product missing the same information on an attribute the customer explicitly waived
  ("I don't care about material") should carry **zero penalty** for that gap, because the
  customer already said it doesn't matter. Without a separate `declined_attributes` list,
  a ranker scoring "how many stated attributes does this product match" has no way to
  tell those two cases apart and would unfairly penalize products for a gap that was
  never actually in play.

Same underlying fact in both cases (a `no_preference:<attribute>` marker in `rejected`),
surfaced twice because the two jobs need it phrased differently.

`session_summary` fields, each a pattern over the session rather than a fact about
this one turn:


| Field                   | What it's a judgement about                                                                                                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `override_turns`        | has the customer contradicted an earlier stated preference, and on which turns                                                                                                                                                                   |
| `profile_corroboration` | does anything the customer has actually said this session support the `preference_tags` the harness handed you at `reset()`                                                                                                                      |
| `carry_forward`         | one bundled view of what to keep biasing toward for the rest of *this* session — `prefer` / `avoid` / `indifferent_attributes` / `confirmed_tags` — so a caller doesn't re-derive it from `session_profile` and `rejected` separately every turn |


---



### Phase 4: save what you are about to send

**What it does:** lets the next turn understand the reply.

```python
tracker.record_ask(state, ask_attribute)                  # what we asked
tracker.record_recommendations(state, parent_asins)       # what we showed
return {..., "usage": drain_usage()}                      # this turn's one LLM call
```

**Example** — continuing the running example: we ask about `size` next and show two
candidates (a duplicate is deduped automatically):

```python
tracker.record_ask(state, "size")
tracker.record_recommendations(state, ["B001", "B002", "B001"])
```

```python
# state now carries, ready for the next turn's extract_slots() call:
previous_ask_attribute="size"
previous_top_10=["B001", "B002"]
```

This is exactly what makes Phase 1's turn-2 example above work: `extract_slots("US 9", state)` sees `previous_ask_attribute="size"` and resolves the bare reply to the right
slot instead of guessing.

`record_ask` matters more than it looks. If we asked about colour and the
customer replies just "black", the extractor needs to know what the question was.
Verified live: with it, `"black"` lands in `color`. Without it, the model guesses.

Skipping it costs accuracy, not correctness.

---



## Full turn, in code

```python
from state.dialogue_state import DialogueStateTracker
from state.llm_extractor import extract_slots
from state.context_distiller import distill
from state.llm_client import drain_usage

tracker = DialogueStateTracker(extractor=extract_slots)

# Agent.reset
state = tracker.reset(session_id, user_profile)

# Agent.respond
state = tracker.update(user_message, tracker.get_state(session_id), turn=turn)
context = distill(tracker.get_history_summary(session_id), state)

# ... your retrieval and rerank read state.to_dict(), context["short_term"], state.intent ...

tracker.record_ask(state, ask_attribute)
tracker.record_recommendations(state, [r["parent_asin"] for r in recommendations])
return {
    "message": message,
    "ask_attribute": ask_attribute,
    "recommendations": recommendations,
    "usage": drain_usage(),
}
```

---



## Setting up the API key

Nothing works until this is done, but nothing breaks either. You can build
against the interface first.

**1. Get a free Groq key.** [console.groq.com](https://console.groq.com), then
**API Keys** and **Create API Key**. It starts with `gsk_` and is shown once.

**2. Add it to your shell.**

```sh
export LLM_API_KEY='gsk_your_key_here'
```

Put that line in `~/.zshrc` to keep it. The defaults already point at Groq, so
the key is the only variable you need.

**3. Check it.** From the repo root:

```sh
python3 -c "
import sys; sys.path.insert(0, 'submission/src')
from state.dialogue_state import DialogueState
from state.llm_extractor import extract_slots
from state.llm_client import drain_usage
print('slots:', extract_slots('black waterproof hiking boots, max 120 dollars', DialogueState()))
print('usage:', drain_usage())
"
```

Working output looks like this:

```text
slots: {'intent': 'Buying', 'category': ['hiking boots'], 'color': ['black'], 'budget': ['<=120'], 'feature': ['waterproof']}
usage: {'prompt_tokens': 1470, 'completion_tokens': 340}
```

With no key set you get one warning and `slots: {}`, which is the designed no-op.

**Never commit the key.** The competition rules require environment variables.
If one leaks, revoke it in the console and make a new one. Also avoid running
`env` or `export` while screen recording.

### Settings


| Variable               | Default                          | Notes                           |
| ---------------------- | -------------------------------- | ------------------------------- |
| `LLM_API_KEY`          | none                             | required                        |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | no trailing `/chat/completions` |
| `LLM_MODEL`            | `openai/gpt-oss-120b`            | as that provider spells it      |
| `LLM_MAX_TOKENS`       | `2048`                           | keep generous, see below        |
| `LLM_TIMEOUT`          | `20`                             | seconds per attempt             |
| `LLM_MAX_ATTEMPTS`     | `2`                              | only 408/409/425/429/5xx retry  |
| `LLM_REASONING_EFFORT` | unset                            | e.g. `low`, sent only if set    |


Other providers work by changing two variables: OpenRouter
(`https://openrouter.ai/api/v1`), Gemini's OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`), or local Ollama
(`http://localhost:11434/v1` with `LLM_API_KEY=ollama`).

The provider must support `response_format: {"type": "json_schema"}`. The reply
is read with `json.loads` and nothing else. No fence stripping, no prose rescue.
A provider that ignores the schema fails cleanly rather than feeding half-parsed
guesses into the state.

### Cost, and the rate limit you will hit

Measured on Groq free tier with `gpt-oss-120b`. One joint `extract_slots` call
does both jobs (slots and intent), so this is the entire per-turn cost:


| Call                             | Prompt                | Completion |
| -------------------------------- | --------------------- | ---------- |
| `extract_slots` (slots + intent) | ~1470                 | ~200-340   |
| **per turn**                     | **~1700-1800 tokens** |            |


The free tier allows **8000 tokens per minute**, which is about **4.5 turns per
minute** — better than the roughly 3.3 turns per minute two separate calls cost
before this was merged. A full 200 session run still takes a while. Develop
against a subset or a local Ollama, and budget time for the full run.

`LLM_REASONING_EFFORT=low` cuts completion tokens further, but it was measured
emitting `"< =120"` instead of `"<=120"`. `budget_bounds` tolerates the stray
space now, but check quality before relying on low effort.

### If something goes wrong

Turn on logging to see the provider's own error:

```sh
python3 -c "
import logging, sys; logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, 'submission/src')
from state.dialogue_state import DialogueState
from state.llm_extractor import extract_slots
extract_slots('black boots under 120 dollars', DialogueState())
"
```


| Symptom                           | Cause                                                                                              |
| --------------------------------- | -------------------------------------------------------------------------------------------------- |
| `http_403` and `error code: 1010` | Cloudflare, not the API. The `User-Agent` header went missing. urllib's default is blocked.        |
| `http_401`                        | key wrong, expired, or has a stray quote                                                           |
| `http_404` model not found        | `LLM_MODEL` is not available to that key. Free tier model IDs differ per account and get retired.  |
| `http_429`                        | the 8000 tokens per minute limit                                                                   |
| `json_validate_failed`            | `LLM_MAX_TOKENS` too low. The model reasons before emitting JSON, so a tight ceiling truncates it. |
| `returned non-JSON content`       | that model does not support strict JSON schema                                                     |


List what your key can actually reach:

```sh
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $LLM_API_KEY" \
  | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

Tested on this account: `gpt-oss-120b` and `gpt-oss-20b` both work and extract
identically. `qwen3.8-27b` is faster but failed to normalize budget.
`qwen3.6-27b` and `groq/compound-mini` reject strict JSON schema outright.

---



## What other modules should call

Every one of these works on a plain dict, so nothing needs to import
`DialogueState`.


| Your module         | Call                                                                         | You get                                                                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| numeric filter      | `budget_bounds(profile)`                                                     | `{"min_price", "max_price", "target_price"}`, floats or `None`. All `None` means no budget, so skip filtering. Decodes `"<=120"` so you never parse it. |
| BM25 and dense      | `state.query_terms()`                                                        | every positive value in one flat list                                                                                                                   |
| RRF weighting       | `state.intent`                                                               | `"Buying"`, `"Browsing"`, or `None`                                                                                                                     |
| RRF on override     | `state.conflicts_with_previous`, `context["short_term"]["focus_attributes"]` | that an override happened, and which slots changed                                                                                                      |
| negative filtering  | `context["short_term"]["avoid"]`                                             | rejected values with the slot each came from                                                                                                            |
| entropy questions   | `state.missing_attributes()`                                                 | askable slots only, already excluding filled and permanently declined ones                                                                              |
| entropy priors      | `context["session_summary"]["profile_corroboration"]`                        | tags split into `confirmed` and `unobserved`                                                                                                            |
| conflict resolution | `context["short_term"]["constraints"]`                                       | `first_seen_turn`/`turns_held`/`revisions` per attribute, for when two stated constraints disagree                                                      |
| reranker            | `state.to_dict()`                                                            | the raw slots, or `context["short_term"]["digest"]` for a denser version                                                                                |
| agent skeleton      | `record_ask`, `record_recommendations`, `drain_usage`                        | call all three at the end of each turn                                                                                                                  |


Entropy rules map onto what already exists:


| Rule                                         | Where it lives                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------- |
| exclude provided or confirmed attributes     | filled slots are not in `missing_attributes()`                                      |
| refuted **value**, so re-ask the attribute   | the slot stays empty so the attribute stays askable, and the value lands in `avoid` |
| "no more preferences", so ban forever        | `no_preference_attributes()`, excluded for the rest of the session                  |
| it was argmax and the refusal did not inform | `state.previous_ask_attribute`                                                      |


---



## State fields


| Field                     | Type                   | Meaning                                                                                              |
| ------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `session_id`              | `str`                  | from the harness                                                                                     |
| `turn`                    | `int`                  | 1 to 10. `0` means reset happened and no turn ran.                                                   |
| `session_profile`         | `dict[str, list[str]]` | the ten `ask_attribute` names plus `rejected`                                                        |
| `user_profile`            | `dict`                 | anonymized profile, read only, never changed here                                                    |
| `previous_top_10`         | `list[str]`            | what was shown last turn                                                                             |
| `previous_ask_attribute`  | `str`                  | what was asked last turn, or `""`                                                                    |
| `conflicts_with_previous` | `bool`                 | this turn contradicted earlier state. Recomputed each turn, not sticky.                              |
| `intent`                  | `str | None`           | `"Buying"` or `"Browsing"` for this turn, or `None` if unresolved. Recomputed each turn, not sticky. |


Guarantees on `session_profile`:

- every key is always present and every value is always a list, so no `.get()`
guards are needed
- values keep first-seen order, deduped by case, and a short phrase is dropped
when a longer one covers it ("leather" goes when "full-grain leather" arrives)
- `rejected` holds two kinds of string: `no_preference:<attribute>` marks, read
with `no_preference_attributes()`, and everything else, which is a negative
term for retrieval
- a value is never in a slot and in `rejected` at the same time

Helpers: `state.filled_attributes()`, `state.missing_attributes()`,
`state.query_terms()`, `no_preference_attributes(profile)`,
`budget_bounds(profile)`, `state.to_dict()`, `DialogueState.from_dict(payload)`.

---



## Known limits

- **A multi-value override needs a named retraction.** If the extractor says
`color: ["navy"]` without putting `"black"` in `rejected`, you get both colours
and no conflict flag. `category`, `budget`, and `size` are protected by the
replacement rule in Phase 2. The other slots are not, because "black or navy is
fine" is a legitimate two-value answer.
- **Retracted values match on whole words.** Retracting `"black"` also drops
`"black cotton trim"`. Deliberate, so that retracting `"running"` clears
`"women's running shoes"`, but it can over-reach.
- **A pure refinement logs** `no_change`**.** The slot length does not grow, so no
`slot_filled:` reason appears. The log is not a complete record of value edits.
- `previous_top_10` is not cleared on an override, so items shown before a switch
stay in the state.
- No handling of "the second one". Nothing is saved to disk.

---



## Tests

46 tests, no network and no credentials needed. The LLM transport is tested by
replacing `urllib.request.urlopen`, so the real request body and the real parsing
path are both covered.

```sh
PYTHONPATH=submission/src python3 -m unittest tests.test_dialogue_state -v
```

To check the live path instead, use the command in the setup section above.