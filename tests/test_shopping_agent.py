from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.message_builder import LLMMessageBuilder, TemplateMessageBuilder
from agent.shopping_agent import Agent
from state.dialogue_state import DialogueState, DialogueStateTracker, empty_session_profile


class FakeStateTracker:
    """Minimal StateTracker double: no slot extraction, just turn bookkeeping.

    ``default_intent`` lets a test control what ``update()`` sets
    ``state.intent`` to -- standing in for the real joint intent+slot
    extractor, so tests can confirm ``Agent`` reads mode off state rather
    than calling a separate classifier.
    """

    def __init__(self, default_intent: str | None = None) -> None:
        self._states: dict[str, DialogueState] = {}
        self.default_intent = default_intent

    def reset(self, session_id: str, user_profile: dict | None = None) -> DialogueState:
        state = DialogueState(session_id=session_id, user_profile=dict(user_profile or {}))
        self._states[session_id] = state
        return state

    def get_state(self, session_id: str) -> DialogueState:
        return self._states[session_id]

    def update(self, user_message: str, current_state: DialogueState, turn: int | None = None) -> DialogueState:
        state = current_state.copy()
        state.turn = turn if turn is not None else current_state.turn + 1
        state.intent = self.default_intent
        self._states[state.session_id] = state
        return state

    def record_recommendations(self, state: DialogueState, parent_asins: list[str]) -> DialogueState:
        state.previous_top_10 = list(parent_asins)
        return state


class FakeSearcher:
    """Stands in for the retrieval teammate's black box in these tests only --
    not shipped as part of the agent module. Matches the real module-level
    ``search(state) -> (candidates, ask_attribute, diagnostics)`` contract, and
    is wired in via ``patch("agent.shopping_agent.search", ...)`` rather than
    an injected attribute -- ``Agent`` calls the imported function directly,
    it does not hold a ``self.searcher``."""

    def __init__(self, candidates: list[str], ask_attribute: str | None) -> None:
        self.candidates = candidates
        self.ask_attribute = ask_attribute
        self.calls: list[DialogueState] = []

    def search(self, state: DialogueState):
        self.calls.append(state)
        return self.candidates, self.ask_attribute, {}


class FakeMessageBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def build(self, ask_attribute, mode, state, candidates):
        self.calls.append((ask_attribute, mode, state, candidates))
        return f"asking about {ask_attribute}" if ask_attribute else "presenting results", {}


class AgentWiringTests(unittest.TestCase):
    """Agent.respond() plumbing, fully faked -- no network, no catalog file."""

    def _agent(self, ask_attribute: str | None = "material", candidates: list[str] | None = None) -> tuple[Agent, FakeSearcher]:
        searcher = FakeSearcher(candidates if candidates is not None else ["A", "B", "C"], ask_attribute)
        agent = Agent.__new__(Agent)  # bypass __init__
        agent.state_tracker = FakeStateTracker()
        agent.message_builder = FakeMessageBuilder()
        return agent, searcher

    def test_respond_shape_matches_contract(self) -> None:
        agent, searcher = self._agent()
        agent.reset("s1", {"summary": "test"})
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            response = agent.respond("s1", "I need a wool sweater", turn=1, top_k=10)
        self.assertIsInstance(response["message"], str)
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}, {"parent_asin": "C"}])
        self.assertIn("prompt_tokens", response["usage"])
        self.assertIn("completion_tokens", response["usage"])

    def test_respond_truncates_recommendations_to_top_k(self) -> None:
        agent, searcher = self._agent()
        agent.reset("s1", {})
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            response = agent.respond("s1", "hello", turn=1, top_k=2)
        self.assertEqual(len(response["recommendations"]), 2)

    def test_ask_attribute_none_when_searcher_signals_nothing_left(self) -> None:
        agent, searcher = self._agent(ask_attribute=None)
        agent.reset("s1", {})
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            response = agent.respond("s1", "hello", turn=1, top_k=10)
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(response["message"], "presenting results")

    def test_mode_comes_from_state_intent_not_a_separate_classifier(self) -> None:
        # No intent_classifier exists on Agent anymore -- mode for the message
        # builder is read straight off state.intent, which the (real) joint
        # intent+slot extractor sets as part of state_tracker.update().
        agent, searcher = self._agent()
        agent.state_tracker = FakeStateTracker(default_intent="buying")
        agent.reset("s1", {})
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            agent.respond("s1", "I need a size 10 boot", turn=1, top_k=10)
        self.assertEqual(len(agent.message_builder.calls), 1)
        _, mode_seen, state_seen, _ = agent.message_builder.calls[0]
        self.assertEqual(mode_seen, "buying")
        # search() ran against the *updated* state (turn 1), same state.intent
        # SearchPipeline itself reads for RRF weighting.
        self.assertEqual(len(searcher.calls), 1)
        self.assertEqual(searcher.calls[0].turn, 1)
        self.assertEqual(searcher.calls[0].intent, "buying")

    def test_mode_defaults_to_browsing_when_state_intent_is_unset(self) -> None:
        agent, searcher = self._agent()  # FakeStateTracker() default_intent=None
        agent.reset("s1", {})
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            agent.respond("s1", "hello", turn=1, top_k=10)
        _, mode_seen, _, _ = agent.message_builder.calls[0]
        self.assertEqual(mode_seen, "browsing")


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
    preferences. This call never touches the network here (a fake client
    records what would be sent)."""

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
        agent = Agent(message_builder=TemplateMessageBuilder())
        self.assertIsInstance(agent.state_tracker, DialogueStateTracker)
        agent.reset("s1", {"summary": "test"})
        searcher = FakeSearcher(["A", "B"], "material")
        with patch("agent.shopping_agent.search", side_effect=searcher.search):
            response = agent.respond("s1", "I need a wool sweater", turn=1, top_k=10)
        self.assertEqual(response["message"], "Do you have a material preference?")
        self.assertEqual(response["ask_attribute"], "material")
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}, {"parent_asin": "B"}])
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main()
