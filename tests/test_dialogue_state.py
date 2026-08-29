"""Workflow tests for the state package.

Run with the submission source on the path, like the other suites here:

    PYTHONPATH=submission/src python3 -m unittest discover -s tests -v

No network and no credentials. The LLM transport is exercised by replacing
``urllib.request.urlopen``, so the real request body and the real parsing path
are both covered without spending tokens.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, List

from state import llm_client
from state.context_distiller import distill
from state.dialogue_state import (
    ASK_ATTRIBUTES,
    DialogueState,
    DialogueStateTracker,
    budget_bounds,
    no_preference_attributes,
)
from state.intent_classifier import classify_intent
from state.llm_extractor import extract_slots


def scripted(script: Dict[int, Dict[str, Any]]):
    """An extractor that replays canned output, keyed by the turn it runs on."""
    return lambda message, state: dict(script.get(state.turn + 1, {}))


def fake_response(content: str, usage: Dict[str, int]):
    """A stand-in for ``urlopen``'s context manager."""
    payload = json.dumps({"choices": [{"message": {"content": content}}], "usage": usage})

    class _Response:
        def read(self) -> bytes:
            return payload.encode()

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

    return _Response()


class TrackerTests(unittest.TestCase):
    """update() must derive every change from the extractor, and nothing else."""

    def test_slots_fill_and_log_reasons(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"category": ["boots"]}}))
        state = tracker.update("anything", tracker.reset("s", {}), turn=1)
        self.assertEqual(state.session_profile["category"], ["boots"])
        self.assertIn("slot_filled:category", tracker.get_transition_log("s")[0]["trigger_reason"])

    def test_empty_extraction_is_a_no_op(self) -> None:
        # The failure mode: a message with nothing in it, and a failed API call,
        # must be indistinguishable and must not disturb the state.
        tracker = DialogueStateTracker(extractor=scripted({}))
        before = tracker.reset("s", {})
        state = tracker.update("those aren't quite right", before, turn=1)
        self.assertEqual(state.session_profile, before.session_profile)
        self.assertFalse(state.conflicts_with_previous)
        self.assertEqual(tracker.get_transition_log("s")[0]["trigger_reason"], "no_slots_extracted")

    def test_named_retraction_moves_value_to_rejected(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"category": ["running shoes"]},
            2: {"category": ["hiking boots"], "rejected": ["running shoes"]},
        }))
        state = tracker.reset("s", {})
        for turn in (1, 2):
            state = tracker.update("m", state, turn=turn)
        self.assertEqual(state.session_profile["category"], ["hiking boots"])
        self.assertIn("running shoes", state.session_profile["rejected"])
        self.assertTrue(state.conflicts_with_previous)

    def test_single_value_slot_catches_unnamed_override(self) -> None:
        # Safety net for when the extractor adds a value but forgets to name the
        # one it replaces. Structural, not text matching.
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"category": ["running shoes"]},
            2: {"category": ["hiking boots"]},
        }))
        state = tracker.reset("s", {})
        for turn in (1, 2):
            state = tracker.update("m", state, turn=turn)
        self.assertEqual(state.session_profile["category"], ["hiking boots"])
        self.assertIn("running shoes", state.session_profile["rejected"])
        self.assertTrue(state.conflicts_with_previous)

    def test_multi_value_slot_treats_new_value_as_additive(self) -> None:
        # Documented limitation: without a named retraction, a second colour is
        # an addition, not an override. Pinned so a change is deliberate.
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"color": ["black"]},
            2: {"color": ["navy"]},
        }))
        state = tracker.reset("s", {})
        for turn in (1, 2):
            state = tracker.update("m", state, turn=turn)
        self.assertEqual(state.session_profile["color"], ["black", "navy"])
        self.assertFalse(state.conflicts_with_previous)

    def test_no_preference_bans_attribute_for_the_session(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"no_preference": ["brand"]}}))
        state = tracker.update("m", tracker.reset("s", {}), turn=1)
        self.assertEqual(no_preference_attributes(state.session_profile), ["brand"])
        self.assertNotIn("brand", state.missing_attributes())

    def test_unknown_no_preference_name_becomes_other(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"no_preference": ["vibe"]}}))
        state = tracker.update("m", tracker.reset("s", {}), turn=1)
        self.assertEqual(no_preference_attributes(state.session_profile), ["other"])

    def test_bare_string_is_accepted_where_a_list_belongs(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"color": "black"}}))
        state = tracker.update("m", tracker.reset("s", {}), turn=1)
        self.assertEqual(state.session_profile["color"], ["black"])

    def test_current_state_is_never_mutated(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"category": ["boots"]}}))
        before = tracker.reset("s", {})
        snapshot = before.to_dict()
        tracker.update("m", before, turn=1)
        self.assertEqual(before.to_dict(), snapshot)

    def test_conflict_flag_is_not_sticky(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"category": ["a"]},
            2: {"category": ["b"]},
            3: {"color": ["red"]},
        }))
        state = tracker.reset("s", {})
        flags = []
        for turn in (1, 2, 3):
            state = tracker.update("m", state, turn=turn)
            flags.append(state.conflicts_with_previous)
        self.assertEqual(flags, [False, True, False])

    def test_get_state_requires_reset(self) -> None:
        with self.assertRaises(KeyError):
            DialogueStateTracker().get_state("never-reset")

    def test_no_text_matching_against_the_message(self) -> None:
        # The same canned extraction must produce the same state no matter what
        # the customer actually said. That is the Task 0 guarantee.
        results = []
        for message in ["forget it, no, actually not that", "yes please", ""]:
            tracker = DialogueStateTracker(extractor=scripted({1: {"color": ["black"]}}))
            state = tracker.update(message, tracker.reset("s", {}), turn=1)
            results.append(state.to_dict()["session_profile"])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])


class ContractTests(unittest.TestCase):
    """Shape promises other modules are built on."""

    def test_every_slot_is_always_a_present_list(self) -> None:
        state = DialogueState()
        for key in ASK_ATTRIBUTES + ("rejected",):
            self.assertIsInstance(state.session_profile[key], list)

    def test_round_trip_through_dict(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"category": ["boots"]}}))
        state = tracker.update("m", tracker.reset("s", {"preference_tags": ["fit"]}), turn=1)
        tracker.record_ask(state, "color")
        tracker.record_recommendations(state, ["B1", "B1", "B2"])
        payload = state.to_dict()
        json.dumps(payload)
        rebuilt = DialogueState.from_dict(payload)
        self.assertEqual(rebuilt.to_dict(), payload)
        self.assertEqual(rebuilt.previous_top_10, ["B1", "B2"])

    def test_record_ask_rejects_values_outside_the_enum(self) -> None:
        tracker = DialogueStateTracker()
        state = tracker.reset("s", {})
        tracker.record_ask(state, "color")
        self.assertEqual(state.previous_ask_attribute, "color")
        tracker.record_ask(state, "not_an_attribute")
        self.assertEqual(state.previous_ask_attribute, "")
        tracker.record_ask(state, None)
        self.assertEqual(state.previous_ask_attribute, "")

    def test_missing_attributes_are_all_legal_ask_values(self) -> None:
        self.assertTrue(set(DialogueState().missing_attributes()) <= set(ASK_ATTRIBUTES))


class BudgetBoundsTests(unittest.TestCase):
    """The one place the budget spelling is decoded."""

    def test_bounds(self) -> None:
        cases = [
            (["<=120"], (None, 120.0, None)),
            ([">=25", "<=60"], (25.0, 60.0, None)),
            (["~60"], (None, None, 60.0)),
            ([], (None, None, None)),
            (["<=120", "<=80"], (None, 80.0, None)),      # tighter bound wins
            ([">=25", ">=40"], (40.0, None, None)),
            (["< =120"], (None, 120.0, None)),            # observed model slip
            (["under $50"], (None, None, None)),          # unparseable, not fatal
        ]
        for values, (lo, hi, target) in cases:
            with self.subTest(values=values):
                self.assertEqual(
                    budget_bounds({"budget": values}),
                    {"min_price": lo, "max_price": hi, "target_price": target},
                )

    def test_missing_budget_key_is_safe(self) -> None:
        self.assertEqual(
            budget_bounds({}),
            {"min_price": None, "max_price": None, "target_price": None},
        )


class DistillerTests(unittest.TestCase):
    """Derived context, from the transition log only."""

    def setUp(self) -> None:
        self.tracker = DialogueStateTracker(extractor=scripted({
            1: {"category": ["running shoes"], "use_case": ["road running"]},
            2: {"color": ["black"]},
            3: {"no_preference": ["brand"]},
            4: {"category": ["hiking boots"], "rejected": ["running shoes"], "budget": ["<=120"]},
            5: {"color": ["navy"]},
        }))
        self.state = self.tracker.reset("s", {"preference_tags": ["fit", "road running"]})
        for turn in range(1, 6):
            self.state = self.tracker.update("m", self.state, turn=turn)
        self.context = distill(self.tracker.get_transition_log("s"), self.state)

    def test_output_is_json_serializable(self) -> None:
        json.dumps(self.context)

    def test_constraints_are_sorted_by_confidence(self) -> None:
        scores = [c["confidence"] for c in self.context["short_term"]["constraints"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_revised_attribute_ranks_below_a_long_held_one(self) -> None:
        by_attr = {c["attribute"]: c for c in self.context["short_term"]["constraints"]}
        self.assertLess(by_attr["category"]["confidence"], by_attr["use_case"]["confidence"])

    def test_freshly_extended_slot_counts_as_current(self) -> None:
        colour = next(c for c in self.context["short_term"]["constraints"] if c["attribute"] == "color")
        self.assertEqual(colour["first_seen_turn"], 2)
        self.assertEqual(colour["last_touched_turn"], 5)

    def test_avoid_carries_provenance(self) -> None:
        avoid = self.context["short_term"]["avoid"]
        entry = next(a for a in avoid if a["value"] == "running shoes")
        self.assertEqual(entry["attribute"], "category")
        self.assertEqual(entry["dropped_turn"], 4)
        self.assertNotIn("no_preference:brand", [a["value"] for a in avoid])

    def test_override_turn_is_recorded(self) -> None:
        self.assertIn(4, self.context["long_term"]["override_turns"])

    def test_profile_tags_split_by_session_evidence(self) -> None:
        corroboration = self.context["long_term"]["profile_corroboration"]
        self.assertIn("road running", corroboration["confirmed"])
        self.assertIn("fit", corroboration["unobserved"])

    def test_refinement_is_not_counted_as_abandonment(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"material": ["leather"]},
            2: {"material": ["full-grain leather"]},
        }))
        state = tracker.reset("r", {})
        for turn in (1, 2):
            state = tracker.update("m", state, turn=turn)
        context = distill(tracker.get_transition_log("r"), state)
        self.assertEqual(context["long_term"]["refinement_count"], 1)
        self.assertEqual(context["long_term"]["abandoned_preferences"], [])

    def test_empty_log_is_shaped_not_broken(self) -> None:
        context = distill([], DialogueState(session_id="e", turn=0))
        json.dumps(context)
        self.assertEqual(context["short_term"]["constraints"], [])
        self.assertEqual(context["long_term"]["volatility"], 0.0)


class LLMTransportTests(unittest.TestCase):
    """The extractor and classifier, with urlopen replaced."""

    def setUp(self) -> None:
        self._real_urlopen = urllib.request.urlopen
        self._env = {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://example.invalid/v1/",
            "LLM_MODEL": "test-model",
            "LLM_MAX_ATTEMPTS": "1",
        }
        import os
        self._saved = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
        llm_client.reset_usage()
        self.captured: Dict[str, Any] = {}

    def tearDown(self) -> None:
        import os
        urllib.request.urlopen = self._real_urlopen
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _serve(self, content: str, usage: Dict[str, int]) -> None:
        def handler(request, timeout=None):
            self.captured["url"] = request.full_url
            self.captured["auth"] = request.get_header("Authorization")
            self.captured["agent"] = request.get_header("User-agent")
            self.captured["body"] = json.loads(request.data.decode())
            return fake_response(content, usage)
        urllib.request.urlopen = handler

    def test_request_is_schema_constrained(self) -> None:
        self._serve(json.dumps({k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}),
                    {"prompt_tokens": 10, "completion_tokens": 2})
        extract_slots("hello", DialogueState())
        body = self.captured["body"]
        schema = body["response_format"]["json_schema"]
        self.assertEqual(self.captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(self.captured["auth"], "Bearer test-key")
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(
            set(schema["schema"]["properties"]),
            set(ASK_ATTRIBUTES) | {"rejected", "no_preference"},
        )
        self.assertEqual(body["temperature"], 0.0)

    def test_user_agent_is_always_sent(self) -> None:
        # Groq's WAF answers urllib's default agent with 403 / Cloudflare 1010
        # before the request reaches the API, so this header is load bearing.
        self._serve(json.dumps({k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}),
                    {"prompt_tokens": 1, "completion_tokens": 1})
        extract_slots("hello", DialogueState())
        agent = self.captured["agent"]
        self.assertTrue(agent)
        # The blocked form is the bare default, "Python-urllib/3.11". Naming
        # urllib inside a longer, identifiable string is fine and honest.
        self.assertFalse(agent.lower().startswith("python-urllib"))

    def test_undeclared_keys_are_dropped(self) -> None:
        payload = {k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}
        payload["category"] = ["boots"]
        payload["invented"] = ["nope"]
        self._serve(json.dumps(payload), {"prompt_tokens": 5, "completion_tokens": 5})
        slots = extract_slots("boots", DialogueState())
        self.assertEqual(slots, {"category": ["boots"]})

    def test_empty_arrays_are_stripped(self) -> None:
        payload = {k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}
        self._serve(json.dumps(payload), {"prompt_tokens": 5, "completion_tokens": 5})
        self.assertEqual(extract_slots("hello", DialogueState()), {})

    def test_prose_is_not_salvaged(self) -> None:
        self._serve("Sure! The category is boots.", {"prompt_tokens": 5, "completion_tokens": 7})
        self.assertEqual(extract_slots("boots", DialogueState()), {})

    def test_failed_call_is_still_billed(self) -> None:
        self._serve("not json at all", {"prompt_tokens": 5, "completion_tokens": 7})
        extract_slots("boots", DialogueState())
        self.assertEqual(llm_client.drain_usage(), {"prompt_tokens": 5, "completion_tokens": 7})

    def test_http_error_returns_empty_and_does_not_raise(self) -> None:
        def handler(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 400, "Bad", {}, io.BytesIO(b"{}"))
        urllib.request.urlopen = handler
        self.assertEqual(extract_slots("boots", DialogueState()), {})
        self.assertEqual(classify_intent("boots", DialogueState())["error"], "http_400")

    def test_timeout_returns_empty_and_does_not_raise(self) -> None:
        def handler(request, timeout=None):
            raise urllib.error.URLError(TimeoutError("timed out"))
        urllib.request.urlopen = handler
        self.assertEqual(extract_slots("boots", DialogueState()), {})
        self.assertEqual(classify_intent("boots", DialogueState())["error"], "timeout")

    def test_previous_ask_attribute_is_sent_as_context(self) -> None:
        self._serve(json.dumps({k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}),
                    {"prompt_tokens": 1, "completion_tokens": 1})
        state = DialogueState(previous_ask_attribute="color")
        extract_slots("black", state)
        sent = json.loads(self.captured["body"]["messages"][1]["content"])
        self.assertEqual(sent["we_just_asked_about"], "color")

    def test_user_profile_is_withheld_from_the_extractor(self) -> None:
        # Long-term taste must not become a constraint the customer never stated.
        self._serve(json.dumps({k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}),
                    {"prompt_tokens": 1, "completion_tokens": 1})
        extract_slots("hello", DialogueState(user_profile={"preference_tags": ["fit"]}))
        sent = json.loads(self.captured["body"]["messages"][1]["content"])
        self.assertNotIn("user_profile", sent)

    def test_classifier_reads_attribute_names_not_values(self) -> None:
        self._serve(json.dumps({"signal": "names a product", "intent": "Buying", "confidence": 0.9}),
                    {"prompt_tokens": 20, "completion_tokens": 4})
        state = DialogueState(turn=3)
        state.session_profile["category"] = ["boots"]
        label = classify_intent("I want boots under 100", state)
        self.assertEqual(label["intent"], "Buying")
        sent = json.loads(self.captured["body"]["messages"][1]["content"])
        self.assertEqual(sent["filled_attributes"], ["category"])
        self.assertNotIn("boots", json.dumps(sent["filled_attributes"]))

    def test_confidence_is_clamped(self) -> None:
        self._serve(json.dumps({"signal": "x", "intent": "Buying", "confidence": 1.4}),
                    {"prompt_tokens": 1, "completion_tokens": 1})
        self.assertEqual(classify_intent("hi", DialogueState())["confidence"], 1.0)

    def test_label_outside_enum_is_a_failure_not_a_default(self) -> None:
        self._serve(json.dumps({"signal": "x", "intent": "Lurking", "confidence": 0.9}),
                    {"prompt_tokens": 1, "completion_tokens": 1})
        label = classify_intent("hi", DialogueState())
        self.assertIsNone(label["intent"])
        self.assertEqual(label["confidence"], 0.0)

    def test_missing_credentials_degrade_quietly(self) -> None:
        import os
        os.environ.pop("LLM_API_KEY", None)
        self.assertEqual(extract_slots("boots", DialogueState()), {})
        label = classify_intent("boots", DialogueState())
        self.assertIsNone(label["intent"])
        self.assertEqual(label["error"], "no_credentials")

    def test_usage_meter_accumulates_then_drains(self) -> None:
        self._serve(json.dumps({k: [] for k in list(ASK_ATTRIBUTES) + ["rejected", "no_preference"]}),
                    {"prompt_tokens": 100, "completion_tokens": 10})
        extract_slots("a", DialogueState())
        extract_slots("b", DialogueState())
        self.assertEqual(llm_client.drain_usage(), {"prompt_tokens": 200, "completion_tokens": 20})
        self.assertEqual(llm_client.drain_usage(), {"prompt_tokens": 0, "completion_tokens": 0})


class FullTurnLoopTests(unittest.TestCase):
    """One session, all four phases, as the agent will run them."""

    def test_five_turn_session(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({
            1: {"category": ["running shoes"], "use_case": ["road running"]},
            2: {"color": ["black"], "budget": ["<=80"]},
            3: {"no_preference": ["brand"]},
            4: {},
            5: {"category": ["hiking boots"], "rejected": ["running shoes"], "budget": ["<=150"]},
        }))
        asks = {1: "color", 2: "brand", 3: "size", 4: None, 5: None}
        state = tracker.reset("loop", {"preference_tags": ["comfort"]})

        for turn in range(1, 6):
            state = tracker.update("m", state, turn=turn)
            context = distill(tracker.get_transition_log("loop"), state)
            json.dumps(context)
            tracker.record_ask(state, asks[turn])
            tracker.record_recommendations(state, [f"B{turn}"])

        self.assertEqual(state.turn, 5)
        self.assertEqual(state.session_profile["category"], ["hiking boots"])
        self.assertEqual(budget_bounds(state.session_profile)["max_price"], 150.0)
        self.assertIn("running shoes", state.session_profile["rejected"])
        self.assertNotIn("brand", state.missing_attributes())
        self.assertTrue(state.conflicts_with_previous)
        self.assertEqual(state.previous_top_10, ["B5"])

        reasons = [e["trigger_reason"] for e in tracker.get_transition_log("loop")]
        self.assertEqual(reasons[3], "no_slots_extracted")
        self.assertIn(5, context["long_term"]["override_turns"])
        self.assertEqual(context["short_term"]["focus_attributes"], ["budget", "category"])

    def test_sessions_are_isolated(self) -> None:
        tracker = DialogueStateTracker(extractor=scripted({1: {"category": ["boots"]}}))
        a = tracker.update("m", tracker.reset("a", {}), turn=1)
        b = tracker.update("m", tracker.reset("b", {}), turn=1)
        self.assertEqual(len(tracker.get_transition_log("a")), 1)
        self.assertEqual(len(tracker.get_transition_log("b")), 1)
        self.assertEqual(a.session_id, "a")
        self.assertEqual(b.session_id, "b")


if __name__ == "__main__":
    unittest.main()
