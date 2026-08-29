# Dialogue State Tracker (`state`)

Turns what the customer types into a small data object the rest of the agent
reads. One function reads the words. Everything else works from the object.

| File | Job | Calls an LLM |
| --- | --- | --- |
| [dialogue_state.py](dialogue_state.py) | Holds the state, applies changes | no |
| [llm_extractor.py](llm_extractor.py) | Words to slots | yes |
| [intent_classifier.py](intent_classifier.py) | Words to Buying/Browsing | yes |
| [context_distiller.py](context_distiller.py) | History to what we trust | no |
| [llm_client.py](llm_client.py) | Shared HTTP plumbing | yes |

---

## The workflow, in order

```text
Phase 0  reset()            once per session
   |
Phase 1  extract_slots()    words  ->  {"category": ["boots"], ...}
   |
Phase 2  update()           slots  ->  new DialogueState
   |
   +--> Phase 3  classify_intent()  ->  "Buying" / "Browsing"
   +--> Phase 4  distill()          ->  {"short_term", "long_term"}
   |
Phase 5  record_ask(), record_recommendations(), drain_usage()
```

Phases 3 and 4 both read the state Phase 2 produced. They do not depend on each
other, so they can run at the same time. Phase 5 always runs last.

---

### Phase 0: start the session

**What it does:** clears everything and stores the anonymized profile.

| | |
| --- | --- |
| **In** | `session_id`, `user_profile` from the harness |
| **Does** | empties all ten slots plus `rejected`, sets `turn` to 0 |
| **Out** | a fresh `DialogueState` |

```python
tracker = DialogueStateTracker(extractor=extract_slots)   # once, in Agent.__init__
state = tracker.reset(session_id, user_profile)           # in Agent.reset
```

---

### Phase 1: read the words

**What it does:** the only place in the agent that reads natural language.

| | |
| --- | --- |
| **In** | the raw message, plus the slots we already hold |
| **Does** | one LLM call with a strict JSON schema |
| **Out** | a `dict` of just what changed, or `{}` |

The model is shown the current slots on purpose. "Boots instead" only means
something if you know what it replaces, so showing the slots is what lets the
model name the old value precisely.

```python
extract_slots("actually hiking boots instead, under $120", state)
# {"category": ["hiking boots"], "budget": ["<=120"], "rejected": ["running shoes"]}
```

Three kinds of key come back:

| Key | Holds |
| --- | --- |
| the ten attribute names | new or changed values |
| `rejected` | **values** the customer dropped, copied exactly from the current slots |
| `no_preference` | attribute **names** they refuse to constrain ("any colour is fine") |

`rejected` is values, `no_preference` is names. That is the easy thing to mix up.

`budget` arrives already normalized as `"<=120"`, `">=25"`, or `"~60"`, so
nothing downstream reads prose prices.

`{}` means nothing new. A failed API call also returns `{}`, and the two are
deliberately identical. There is no keyword fallback underneath, so a bad turn
becomes "no new information" instead of a guess.

---

### Phase 2: apply it to the state

**What it does:** folds the extracted dict into a new state object.

| | |
| --- | --- |
| **In** | the raw message (passed straight to Phase 1), the current state |
| **Does** | applies retractions, then no-preference marks, then new values |
| **Out** | a **new** `DialogueState`, plus a log entry |

```python
state = tracker.update(user_message, tracker.get_state(session_id), turn=turn)
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

`update()` never looks at the message text itself. Every change above comes from
Phase 1's return value. That is why an API failure is harmless: `{}` in means
nothing changes, and the log says `no_slots_extracted`.

The old state is never modified, so you can keep it and compare.

---

### Phase 3: label the intent

**What it does:** says whether the customer is Buying or Browsing.

| | |
| --- | --- |
| **In** | the raw message, plus which attributes are filled and which are missing |
| **Does** | one LLM call with a two-value enum |
| **Out** | `{"intent", "confidence", "signal", "usage", "error"}` |

```python
intent = classify_intent(user_message, state)
# {"intent": "Buying", "confidence": 0.92, "signal": "names a product and a price cap", ...}
```

The model gets attribute **names** only, not their values, because the label
depends on how far the session has converged rather than on what was chosen.

Two labels, not four. The public set also labels sessions Intent Override and
Boundary, but the state already reports those without a model call
(`conflicts_with_previous` and `no_preference_attributes()`). An override session
is still Buying or Browsing on every turn.

On failure `intent` is `None`. Branch on it. Do not read it as Browsing, because
this module deliberately makes no routing decision.

---

### Phase 4: distill the history

**What it does:** works out which constraints to trust. No LLM call.

| | |
| --- | --- |
| **In** | the transition log, plus the current state |
| **Does** | one pass over the log, comparing each turn to the one before |
| **Out** | `{"short_term": ..., "long_term": ...}` |

```python
context = distill(tracker.get_transition_log(session_id), state)
```

**`short_term`** is small enough to paste into a ranking prompt:

| Field | What it adds that the raw state cannot |
| --- | --- |
| `constraints` | every filled slot, **sorted by confidence**. The raw state has no ordering and makes every slot look equally certain. |
| `avoid` | each rejected value with the slot it came from and the turn it was dropped |
| `declined_attributes` | attributes the customer refused to constrain |
| `open_attributes` | still worth asking about |
| `unstable_attributes` | slots the customer has already rewritten |
| `focus_attributes` | what changed this turn. On an override turn, this is what to weight up. |
| `digest` | all of the above as one line of text |

`confidence` starts at 0.55, gains 0.10 per turn survived (capped at 3), loses
0.15 per revision, gains 0.10 if touched this turn, then is clamped to the range
0.15 to 0.95. It is a rough band, not a probability. Ties are normal and break
alphabetically so the output is stable between runs. Use it to decide which
constraint wins a conflict.

**`long_term`** is patterns meant to outlive the session: `stable_preferences`,
`abandoned_preferences`, `refinement_count`, `revision_profile`, `volatility`,
`decisiveness`, `override_turns`, `profile_corroboration` (which
`preference_tags` this session actually supports), and `carry_forward`.

The useful contrast is `refinement_count` against `volatility`. Sharpening a
request ("leather" becoming "full-grain leather") is a customer who knows what
they want. Rewriting it is one who does not. The raw state cannot tell them
apart, because both look like a slot that changed.

---

### Phase 5: save what you are about to send

**What it does:** lets the next turn understand the reply.

```python
tracker.record_ask(state, ask_attribute)                  # what we asked
tracker.record_recommendations(state, parent_asins)       # what we showed
return {..., "usage": drain_usage()}                      # both LLM calls, this turn
```

`record_ask` matters more than it looks. If we asked about colour and the
customer replies just "black", the extractor needs to know what the question was.
Verified live: with it, `"black"` lands in `color`. Without it, the model guesses.

Skipping it costs accuracy, not correctness.

---

## Full turn, in code

```python
from state.dialogue_state import DialogueStateTracker
from state.llm_extractor import extract_slots
from state.intent_classifier import classify_intent
from state.context_distiller import distill
from state.llm_client import drain_usage

tracker = DialogueStateTracker(extractor=extract_slots)

# Agent.reset
state = tracker.reset(session_id, user_profile)

# Agent.respond
state = tracker.update(user_message, tracker.get_state(session_id), turn=turn)
intent = classify_intent(user_message, state)
context = distill(tracker.get_transition_log(session_id), state)

# ... your retrieval and rerank read state.to_dict(), context["short_term"], intent["intent"] ...

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
slots: {'category': ['hiking boots'], 'color': ['black'], 'budget': ['<=120'], 'feature': ['waterproof']}
usage: {'prompt_tokens': 1226, 'completion_tokens': 406}
```

With no key set you get one warning and `slots: {}`, which is the designed no-op.

**Never commit the key.** The competition rules require environment variables.
If one leaks, revoke it in the console and make a new one. Also avoid running
`env` or `export` while screen recording.

### Settings

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_API_KEY` | none | required |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | no trailing `/chat/completions` |
| `LLM_MODEL` | `openai/gpt-oss-120b` | as that provider spells it |
| `LLM_MAX_TOKENS` | `2048` | keep generous, see below |
| `LLM_TIMEOUT` | `20` | seconds per attempt |
| `LLM_MAX_ATTEMPTS` | `2` | only 408/409/425/429/5xx retry |
| `LLM_REASONING_EFFORT` | unset | e.g. `low`, sent only if set |

Other providers work by changing two variables: OpenRouter
(`https://openrouter.ai/api/v1`), Gemini's OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai`), or local Ollama
(`http://localhost:11434/v1` with `LLM_API_KEY=ollama`).

The provider must support `response_format: {"type": "json_schema"}`. The reply
is read with `json.loads` and nothing else. No fence stripping, no prose rescue.
A provider that ignores the schema fails cleanly rather than feeding half-parsed
guesses into the state.

### Cost, and the rate limit you will hit

Measured on Groq free tier with `gpt-oss-120b`:

| Call | Prompt | Completion |
| --- | --- | --- |
| `extract_slots` | ~1220 | ~420 |
| `classify_intent` | ~590 | ~205 |
| **per turn** | **~2440 tokens** | |

The free tier allows **8000 tokens per minute**, which is about **3.3 turns per
minute**. A full 200 session run therefore takes hours. Develop against a subset
or a local Ollama, and budget time for the full run.

`LLM_REASONING_EFFORT=low` drops this to about 1960 per turn, but it was measured
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

| Symptom | Cause |
| --- | --- |
| `http_403` and `error code: 1010` | Cloudflare, not the API. The `User-Agent` header went missing. urllib's default is blocked. |
| `http_401` | key wrong, expired, or has a stray quote |
| `http_404` model not found | `LLM_MODEL` is not available to that key. Free tier model IDs differ per account and get retired. |
| `http_429` | the 8000 tokens per minute limit |
| `json_validate_failed` | `LLM_MAX_TOKENS` too low. The model reasons before emitting JSON, so a tight ceiling truncates it. |
| `returned non-JSON content` | that model does not support strict JSON schema |

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

| Your module | Call | You get |
| --- | --- | --- |
| numeric filter | `budget_bounds(profile)` | `{"min_price", "max_price", "target_price"}`, floats or `None`. All `None` means no budget, so skip filtering. Decodes `"<=120"` so you never parse it. |
| BM25 and dense | `state.query_terms()` | every positive value in one flat list |
| RRF weighting | `classify_intent(...)["intent"]` | `"Buying"`, `"Browsing"`, or `None` |
| RRF on override | `state.conflicts_with_previous`, `context["short_term"]["focus_attributes"]` | that an override happened, and which slots changed |
| negative filtering | `context["short_term"]["avoid"]` | rejected values with the slot each came from |
| entropy questions | `state.missing_attributes()` | askable slots only, already excluding filled and permanently declined ones |
| entropy weighting | `context["short_term"]["constraints"]` | confidence order |
| entropy priors | `context["long_term"]["profile_corroboration"]` | tags split into `confirmed` and `unobserved` |
| reranker | `state.to_dict()` | the raw slots, or `context["short_term"]["digest"]` for a denser version |
| agent skeleton | `record_ask`, `record_recommendations`, `drain_usage` | call all three at the end of each turn |

Entropy rules map onto what already exists:

| Rule | Where it lives |
| --- | --- |
| exclude provided or confirmed attributes | filled slots are not in `missing_attributes()` |
| refuted **value**, so re-ask the attribute | the slot stays empty so the attribute stays askable, and the value lands in `avoid` |
| "no more preferences", so ban forever | `no_preference_attributes()`, excluded for the rest of the session |
| it was argmax and the refusal did not inform | `state.previous_ask_attribute` |

---

## State fields

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | `str` | from the harness |
| `turn` | `int` | 1 to 10. `0` means reset happened and no turn ran. |
| `session_profile` | `dict[str, list[str]]` | the ten `ask_attribute` names plus `rejected` |
| `user_profile` | `dict` | anonymized profile, read only, never changed here |
| `previous_top_10` | `list[str]` | what was shown last turn |
| `previous_ask_attribute` | `str` | what was asked last turn, or `""` |
| `conflicts_with_previous` | `bool` | this turn contradicted earlier state. Recomputed each turn, not sticky. |

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
- **A pure refinement logs `no_change`.** The slot length does not grow, so no
  `slot_filled:` reason appears. `distill()` still counts it in
  `refinement_count`, but the log is not a complete record of value edits.
- `previous_top_10` is not cleared on an override, so items shown before a switch
  stay in the state.
- No handling of "the second one". Nothing is saved to disk.

---

## Tests

44 tests, no network and no credentials needed. The LLM transport is tested by
replacing `urllib.request.urlopen`, so the real request body and the real parsing
path are both covered.

```sh
PYTHONPATH=submission/src python3 -m unittest tests.test_dialogue_state -v
```

To check the live path instead, use the command in the setup section above.
