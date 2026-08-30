from __future__ import annotations

import unittest

from agent.intent_classifier import LLMIntentClassifier, RegexIntentClassifier
from agent.message_builder import LLMMessageBuilder, TemplateMessageBuilder
from agent.shopping_agent import Agent
from state.dialogue_state import DialogueState, DialogueStateTracker, empty_session_profile


class FakeStateTracker:
    """Minimal StateTracker double: no slot extraction, just turn bookkeeping."""

    def __init__(self) -> None:
        self._states: dict[str, DialogueState] = {}

    def reset(self, session_id: str, user_profile: dict | None = None) -> DialogueState:
        state = DialogueState(session_id=session_id, user_profile=dict(user_profile or {}))
        self._states[session_id] = state
        return state

    def get_state(self, session_id: str) -> DialogueState:
        return self._states[session_id]

    def update(self, user_message: str, current_state: DialogueState, turn: int | None = None) -> DialogueState:
        state = current_state.copy()
        state.turn = turn if turn is not None else current_state.turn + 1
        self._states[state.session_id] = state
        return state

    def record_recommendations(self, state: DialogueState, parent_asins: list[str]) -> DialogueState:
        state.previous_top_10 = list(parent_asins)
        return state


class FakeSearcher:
    """Stands in for the retrieval teammate's black box in these tests only --
    not shipped as part of the agent module."""

    def __init__(self, candidates: list[str], ask_attribute: str | None) -> None:
        self.candidates = candidates
        self.ask_attribute = ask_attribute
        self.calls: list[tuple[DialogueState, str, int]] = []

    def search(self, state: DialogueState, mode: str, top_k: int):
        self.calls.append((state, mode, top_k))
        return self.candidates[:top_k], self.ask_attribute


class FakeIntentClassifier:
    def __init__(self, mode: str = "buy") -> None:
        self.mode = mode
        self.last_usage: dict[str, int] = {}
        self.calls: list[tuple[str, DialogueState]] = []

    def classify(self, user_message: str, state: DialogueState) -> str:
        self.calls.append((user_message, state))
        return self.mode


class FakeMessageBuilder:
    def build(self, ask_attribute, mode, state, candidates):
        return f"asking about {ask_attribute}" if ask_attribute else "presenting results", {}


class AgentWiringTests(unittest.TestCase):
    """Agent.respond() plumbing, fully faked -- no network, no catalog file."""

    def _agent(self, ask_attribute: str | None = "material") -> tuple[Agent, FakeSearcher]:
        searcher = FakeSearcher(["A", "B", "C"], ask_attribute)
        agent = Agent.__new__(Agent)  # bypass __init__
        agent.state_tracker = FakeStateTracker()
        agent.searcher = searcher
        agent.intent_classifier = FakeIntentClassifier("buy")
        agent.message_builder = FakeMessageBuilder()
        return agent, searcher

    def test_respond_shape_matches_contract(self) -> None:
        agent, _ = self._agent()
        agent.reset("s1", {"summary": "test"})
        response = agent.respond("s1", "I need a wool sweater", turn=1, top_k=10)
        self.assertIsInstance(response["message"], str)
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}, {"parent_asin": "C"}])
        self.assertIn("prompt_tokens", response["usage"])
        self.assertIn("completion_tokens", response["usage"])

    def test_respond_truncates_recommendations_to_top_k(self) -> None:
        agent, _ = self._agent()
        agent.reset("s1", {})
        response = agent.respond("s1", "hello", turn=1, top_k=2)
        self.assertEqual(len(response["recommendations"]), 2)

    def test_ask_attribute_none_when_searcher_signals_nothing_left(self) -> None:
        agent, _ = self._agent(ask_attribute=None)
        agent.reset("s1", {})
        response = agent.respond("s1", "hello", turn=1, top_k=10)
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["message"], "presenting results")

    def test_intent_classification_runs_before_search_and_state_update(self) -> None:
        agent, searcher = self._agent()
        agent.reset("s1", {})
        agent.respond("s1", "I need a size 10 boot", turn=1, top_k=10)
        # classify() saw the pre-update state (turn 0, from reset), proving it
        # ran on the prior state, not the state produced by this turn's update.
        classify_calls = agent.intent_classifier.calls
        self.assertEqual(len(classify_calls), 1)
        _, state_seen = classify_calls[0]
        self.assertEqual(state_seen.turn, 0)
        # search() ran against the *updated* state (turn 1).
        search_state, mode, top_k = searcher.calls[0]
        self.assertEqual(search_state.turn, 1)
        self.assertEqual(mode, "buy")
        self.assertEqual(top_k, 10)


class RegexIntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RegexIntentClassifier()
        self.empty_state = DialogueState(session_profile=empty_session_profile())

    def test_concrete_requirement_reads_as_buy(self) -> None:
        message = "I need a waterproof size 10 hiking boot under $80."
        self.assertEqual(self.classifier.classify(message, self.empty_state), "buy")

    def test_vague_opening_reads_as_browse(self) -> None:
        message = "Just looking around, not sure what I want yet."
        self.assertEqual(self.classifier.classify(message, self.empty_state), "browse")

    def test_ambiguous_message_falls_back_to_known_constraints(self) -> None:
        filled_state = DialogueState(session_profile={**empty_session_profile(), "category": ["boots"]})
        self.assertEqual(self.classifier.classify("hmm okay", filled_state), "buy")
        self.assertEqual(self.classifier.classify("hmm okay", self.empty_state), "browse")

    def test_multi_digit_size_reads_as_buy(self) -> None:
        # Regression: \d without + only matched a single digit, so "size 10"
        # (and any two-digit size) silently failed to register as "buy".
        self.assertEqual(self.classifier.classify("size 10 boots please", self.empty_state), "buy")

    def test_bare_dollar_amount_reads_as_buy(self) -> None:
        # Regression: \b immediately before "$" never holds in real text (a
        # space or string-start precedes it, not a word character), so a
        # bare "$80" -- without a leading "under" -- silently failed to match.
        self.assertEqual(self.classifier.classify("$80 budget", self.empty_state), "buy")
        self.assertEqual(self.classifier.classify("$8 budget", self.empty_state), "buy")

    def test_requires_reads_as_buy(self) -> None:
        # Regression: required? only matched "require"/"required", not "requires".
        self.assertEqual(self.classifier.classify("It requires waterproofing", self.empty_state), "buy")

    def test_maybe_no_longer_defeats_a_clear_buy_signal(self) -> None:
        # Regression: "maybe" in _BROWSE_RE fired alongside "i need" in
        # _BUY_RE, turning a clear buy message into a tie.
        self.assertEqual(self.classifier.classify("Maybe I need a size 10 boot", self.empty_state), "buy")


class _FakeUsage:
    def __init__(self, input_tokens: int = 1, output_tokens: int = 1) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessagesAPI:
    """Records the last call's kwargs; never touches the network."""

    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("_FakeResponse", (), {"usage": _FakeUsage(), "content": [_FakeTextBlock(self.reply_text)]})()


class _FakeAnthropicClient:
    def __init__(self, reply_text: str = "buy") -> None:
        self.messages = _FakeMessagesAPI(reply_text)


def _rejected_state() -> DialogueState:
    profile = empty_session_profile()
    profile["color"] = ["black"]
    profile["rejected"] = ["no_preference:material", "red"]
    return DialogueState(session_profile=profile)


class KnownContextExcludesRejectedTests(unittest.TestCase):
    """Regression: the "known so far" summary sent to the LLM used to
    include the "rejected" slot verbatim, presenting declined values (and
    raw "no_preference:<attr>" marker strings) as if they were confirmed
    preferences. Both LLM-backed pieces build this string; neither call
    touches the network here (a fake client records what would be sent)."""

    def test_intent_classifier_excludes_rejected(self) -> None:
        client = _FakeAnthropicClient(reply_text="buy")
        classifier = LLMIntentClassifier(client=client)
        classifier.classify("anything", _rejected_state())
        sent = client.messages.last_kwargs["messages"][0]["content"]
        self.assertIn("color", sent)
        self.assertNotIn("rejected", sent)
        self.assertNotIn("no_preference", sent)

    def test_message_builder_excludes_rejected(self) -> None:
        client = _FakeAnthropicClient(reply_text="Do you have a color preference?")
        builder = LLMMessageBuilder(client=client)
        builder.build("color", "buy", _rejected_state(), ["A"])
        sent = client.messages.last_kwargs["messages"][0]["content"]
        self.assertIn("color", sent)
        self.assertNotIn("rejected", sent)
        self.assertNotIn("no_preference", sent)


class TemplateMessageBuilderTests(unittest.TestCase):
    def test_known_attribute_uses_its_template(self) -> None:
        builder = TemplateMessageBuilder()
        message, usage = builder.build("material", "buy", DialogueState(), ["A"])
        self.assertEqual(message, "Do you have a material preference?")
        self.assertEqual(usage, {})

    def test_no_attribute_presents_results(self) -> None:
        builder = TemplateMessageBuilder()
        message, usage = builder.build(None, "buy", DialogueState(), ["A"])
        self.assertEqual(message, "Here are the closest matches I found.")


class AgentOfflineSwapTests(unittest.TestCase):
    """Real DialogueStateTracker + a fake searcher + the offline
    regex/template swap -- confirms the non-LLM wiring runs end-to-end with
    no network dependency."""

    def test_offline_swap_runs_end_to_end(self) -> None:
        agent = Agent(
            searcher=FakeSearcher(["A", "B"], "material"),
            intent_classifier=RegexIntentClassifier(),
            message_builder=TemplateMessageBuilder(),
        )
        self.assertIsInstance(agent.state_tracker, DialogueStateTracker)
        agent.reset("s1", {"summary": "test"})
        response = agent.respond("s1", "I need a wool sweater", turn=1, top_k=10)
        self.assertEqual(response["message"], "Do you have a material preference?")
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}])
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
