from __future__ import annotations

import unittest

from scripts.tune_rrf_weights import (
    grid_search_mode,
    import_agent_factory,
    score_weights,
    technical_score_from_summary,
)


class TechnicalScoreFromSummaryTests(unittest.TestCase):
    def test_matches_evaluate_s_own_formula(self) -> None:
        # Same numbers evaluate() would compute for hit_rate=0.8, mrr=0.6,
        # mttc=3.0 -> efficiency = (11-3)/10 = 0.8
        summary = {"sample_count": 10, "hit_rate_at_10": 0.8, "mrr": 0.6, "mttc": 3.0}
        expected = 0.50 * 0.8 + 0.30 * 0.6 + 0.20 * 0.8
        self.assertAlmostEqual(technical_score_from_summary(summary), expected)

    def test_returns_none_for_empty_scenario(self) -> None:
        summary = {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
        self.assertIsNone(technical_score_from_summary(summary))

    def test_efficiency_is_clipped_to_zero_for_a_very_slow_mttc(self) -> None:
        summary = {"sample_count": 5, "hit_rate_at_10": 0.5, "mrr": 0.3, "mttc": 50.0}
        # efficiency would go negative uncapped; must clip to 0
        expected = 0.50 * 0.5 + 0.30 * 0.3 + 0.20 * 0.0
        self.assertAlmostEqual(technical_score_from_summary(summary), expected)


class ImportAgentFactoryTests(unittest.TestCase):
    def test_imports_a_real_callable(self) -> None:
        factory = import_agent_factory("retrieval.rrf:weights_for_mode")
        self.assertTrue(callable(factory))

    def test_missing_colon_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            import_agent_factory("retrieval.rrf.weights_for_mode")

    def test_unknown_attribute_raises(self) -> None:
        with self.assertRaises(AttributeError):
            import_agent_factory("retrieval.rrf:this_does_not_exist")

    def test_unknown_module_raises(self) -> None:
        with self.assertRaises(ImportError):
            import_agent_factory("not_a_real_module:foo")


class _FakeAgent:
    def __init__(self, weight_table: dict) -> None:
        self.weight_table = weight_table


def _fake_evaluate_fixed(agent, samples, catalog_ids, categories, products) -> dict:
    return {
        "scenario_metrics": {
            "buying": {"sample_count": 80, "hit_rate_at_10": 0.7, "mrr": 0.5, "mttc": 4.0},
            "browsing": {"sample_count": 80, "hit_rate_at_10": 0.4, "mrr": 0.2, "mttc": 6.0},
            "boundary": {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None},
        }
    }


class ScoreWeightsTests(unittest.TestCase):
    def test_extracts_technical_score_per_scenario(self) -> None:
        weight_table = {"buying": {"bm25": 3.0, "dense": 1.0}, "browsing": {"bm25": 1.0, "dense": 2.0}}
        scores = score_weights(
            weight_table, _FakeAgent, samples=[], catalog_ids=set(), categories={}, products={},
            evaluate_fn=_fake_evaluate_fixed,
        )
        self.assertAlmostEqual(scores["buying"], technical_score_from_summary(
            {"sample_count": 80, "hit_rate_at_10": 0.7, "mrr": 0.5, "mttc": 4.0}
        ))
        self.assertIsNone(scores["boundary"])

    def test_builds_agent_with_the_given_weight_table(self) -> None:
        captured = {}

        def factory(weight_table):
            captured["weight_table"] = weight_table
            return _FakeAgent(weight_table)

        weight_table = {"buying": {"bm25": 5.0, "dense": 1.0}, "browsing": {"bm25": 1.0, "dense": 1.0}}
        score_weights(
            weight_table, factory, samples=[], catalog_ids=set(), categories={}, products={},
            evaluate_fn=_fake_evaluate_fixed,
        )
        self.assertEqual(captured["weight_table"], weight_table)


class GridSearchModeTests(unittest.TestCase):
    def test_picks_the_best_candidate(self) -> None:
        def evaluate_fn(agent, samples, catalog_ids, categories, products):
            # "Best" candidate is bm25=3, dense=1 -- everything else scores lower.
            buying = agent.weight_table["buying"]
            is_best = buying["bm25"] == 3.0 and buying["dense"] == 1.0
            score = 0.9 if is_best else 0.3
            return {"scenario_metrics": {
                "buying": {"sample_count": 10, "hit_rate_at_10": score, "mrr": score, "mttc": 2.0},
            }}

        results = grid_search_mode(
            "buying", other_mode_weights={"browsing": {"bm25": 1.0, "dense": 1.0}},
            agent_factory=_FakeAgent, samples=[], catalog_ids=set(), categories={}, products={},
            candidates=[1.0, 3.0], evaluate_fn=evaluate_fn,
        )
        self.assertEqual(results[0]["bm25"], 3.0)
        self.assertEqual(results[0]["dense"], 1.0)
        self.assertEqual(len(results), 4)  # 2x2 grid

    def test_other_mode_weights_stay_fixed_across_every_candidate(self) -> None:
        seen_browsing_weights = []

        def evaluate_fn(agent, samples, catalog_ids, categories, products):
            seen_browsing_weights.append(agent.weight_table["browsing"])
            return {"scenario_metrics": {
                "buying": {"sample_count": 10, "hit_rate_at_10": 0.5, "mrr": 0.5, "mttc": 3.0},
            }}

        fixed_browsing = {"bm25": 7.0, "dense": 2.0}
        grid_search_mode(
            "buying", other_mode_weights={"browsing": fixed_browsing},
            agent_factory=_FakeAgent, samples=[], catalog_ids=set(), categories={}, products={},
            candidates=[1.0, 2.0], evaluate_fn=evaluate_fn,
        )
        self.assertTrue(all(w == fixed_browsing for w in seen_browsing_weights))

    def test_none_scores_sort_last_not_crashing(self) -> None:
        def evaluate_fn(agent, samples, catalog_ids, categories, products):
            buying = agent.weight_table["buying"]
            if buying["bm25"] == 1.0:
                # Simulate a scenario with zero sessions for this candidate somehow.
                return {"scenario_metrics": {"buying": {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}}}
            return {"scenario_metrics": {"buying": {"sample_count": 5, "hit_rate_at_10": 0.5, "mrr": 0.5, "mttc": 3.0}}}

        results = grid_search_mode(
            "buying", other_mode_weights={},
            agent_factory=_FakeAgent, samples=[], catalog_ids=set(), categories={}, products={},
            candidates=[1.0, 2.0], evaluate_fn=evaluate_fn,
        )
        self.assertIsNone(results[-1]["technical_score"])


if __name__ == "__main__":
    unittest.main()
