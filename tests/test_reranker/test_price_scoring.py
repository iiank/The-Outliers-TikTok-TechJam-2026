from __future__ import annotations

import unittest

from reranker.reranker import price_multiplier


class PriceMultiplierTests(unittest.TestCase):
    """Pure-function tests for reranker.price_multiplier -- no model load needed.

    Test vectors straight from target-price-scoring-spec.md
    (target_price=60, decay_rate=1.0).
    """

    def test_exact_match_is_one(self) -> None:
        self.assertAlmostEqual(price_multiplier(60, 60), 1.000, places=3)

    def test_one_dollar_over(self) -> None:
        self.assertAlmostEqual(price_multiplier(61, 60), 0.983, places=3)

    def test_thirty_under(self) -> None:
        self.assertAlmostEqual(price_multiplier(30, 60), 0.607, places=3)

    def test_thirty_over(self) -> None:
        self.assertAlmostEqual(price_multiplier(90, 60), 0.607, places=3)

    def test_symmetric_under_and_over_are_equal(self) -> None:
        # Deliberate design decision (spec Open Decision 2), not a
        # coincidence of the two cases above -- assert the symmetry itself.
        self.assertEqual(price_multiplier(30, 60), price_multiplier(90, 60))

    def test_ninety_over(self) -> None:
        self.assertAlmostEqual(price_multiplier(150, 60), 0.223, places=3)

    def test_missing_price_is_neutral(self) -> None:
        self.assertEqual(price_multiplier(None, 60), 1.0)

    def test_missing_target_price_is_neutral(self) -> None:
        self.assertEqual(price_multiplier(60, None), 1.0)

    def test_missing_price_never_defaults_to_zero(self) -> None:
        # Regression guard for the exact footgun the spec calls out:
        # defaulting missing price to 0 would compute multiplier ~0.37,
        # not the required 1.0.
        self.assertNotAlmostEqual(price_multiplier(None, 60), price_multiplier(0, 60), places=2)

    def test_decay_rate_controls_steepness(self) -> None:
        gentle = price_multiplier(90, 60, decay_rate=5.0)
        steep = price_multiplier(90, 60, decay_rate=0.2)
        self.assertGreater(gentle, steep)


class RankPriceIntegrationTests(unittest.TestCase):
    """Tests Reranker.rank()'s price-adjustment step without loading a real
    CrossEncoder model -- constructs a bare instance and calls rank() with
    a stubbed-out model.predict, since loading the real model needs a
    network call and is out of scope for a fast unit test.
    """

    @staticmethod
    def _bare_reranker(scores):
        from reranker.reranker import Reranker

        instance = Reranker.__new__(Reranker)  # skip __init__, no model load

        class _StubModel:
            def predict(self, pairs, **kwargs):
                return list(scores)

        instance.model = _StubModel()
        instance.batch_size = 64
        return instance

    def test_target_price_reorders_the_final_ranking(self) -> None:
        # "far" scores higher on relevance but is priced way off target;
        # "near" scores lower on relevance but sits right on target.
        reranker = self._bare_reranker(scores=[1.0, 0.9])
        candidates = [
            {"parent_asin": "far", "document": "d", "price": 500.0},
            {"parent_asin": "near", "document": "d", "price": 60.0},
        ]
        result = reranker.rank(query="q", candidates=candidates, top_k=2, target_price=60.0)
        self.assertEqual(result[0]["parent_asin"], "near")

    def test_none_target_price_leaves_relevance_order_untouched(self) -> None:
        reranker = self._bare_reranker(scores=[0.5, 0.9])
        candidates = [
            {"parent_asin": "A", "document": "d", "price": 9999.0},
            {"parent_asin": "B", "document": "d", "price": 9999.0},
        ]
        result = reranker.rank(query="q", candidates=candidates, top_k=2, target_price=None)
        self.assertEqual(result[0]["parent_asin"], "B")  # pure relevance order

    def test_returned_score_is_the_price_adjusted_value(self) -> None:
        reranker = self._bare_reranker(scores=[1.0])
        candidates = [{"parent_asin": "A", "document": "d", "price": 90.0}]
        result = reranker.rank(query="q", candidates=candidates, top_k=1, target_price=60.0)
        self.assertAlmostEqual(result[0]["score"], price_multiplier(90.0, 60.0), places=6)

    def test_never_excludes_a_candidate(self) -> None:
        reranker = self._bare_reranker(scores=[1.0, 1.0])
        candidates = [
            {"parent_asin": "A", "document": "d", "price": 60.0},
            {"parent_asin": "B", "document": "d", "price": 100000.0},
        ]
        result = reranker.rank(query="q", candidates=candidates, top_k=2, target_price=60.0)
        self.assertEqual({c["parent_asin"] for c in result}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
