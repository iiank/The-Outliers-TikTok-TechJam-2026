from __future__ import annotations

import unittest

from retrieval.rrf import (
    BROWSING_WEIGHTS,
    BUYING_WEIGHTS,
    ROUTE_ORDER,
    reciprocal_rank_fusion,
    weights_for_mode,
)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_single_ranking_preserves_order(self) -> None:
        ranking = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        fused = reciprocal_rank_fusion([ranking])
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b", "c"])

    def test_agreement_across_routes_wins(self) -> None:
        # "b" is mediocre on both routes; "a" is great on one and absent
        # from the other. Agreement should still be able to outrank a
        # single strong showing, which is the entire point of RRF.
        dense = [("a", 0.99), ("b", 0.5), ("c", 0.4)]
        bm25 = [("b", 10.0), ("c", 8.0), ("d", 1.0)]
        fused = reciprocal_rank_fusion([dense, bm25])
        ids = [doc_id for doc_id, _ in fused]
        self.assertLess(ids.index("b"), ids.index("a"))

    def test_score_matches_formula(self) -> None:
        dense = [("a", 1.0)]
        bm25 = [("a", 1.0)]
        fused = reciprocal_rank_fusion([dense, bm25], rrf_k=60)
        self.assertEqual(fused, [("a", 2 / 61)])

    def test_doc_absent_from_a_route_is_not_penalized_to_zero_contribution(self) -> None:
        dense = [("a", 1.0), ("b", 0.9)]
        bm25: list[tuple[str, float]] = []
        fused = reciprocal_rank_fusion([dense, bm25])
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b"])

    def test_empty_rankings_returns_empty(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_ties_break_by_first_appearance(self) -> None:
        # Both routes rank these identically relative to each other, so
        # fused scores tie; order must stay deterministic.
        dense = [("x", 1.0), ("y", 0.5)]
        fused = reciprocal_rank_fusion([dense, dense])
        self.assertEqual([doc_id for doc_id, _ in fused], ["x", "y"])

    def test_rrf_k_changes_relative_weighting(self) -> None:
        dense = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        bm25 = [("c", 1.0), ("b", 0.9), ("a", 0.8)]
        # Small rrf_k sharpens the gap between rank 1 and rank 3.
        fused_small_k = dict(reciprocal_rank_fusion([dense, bm25], rrf_k=1))
        fused_large_k = dict(reciprocal_rank_fusion([dense, bm25], rrf_k=1000))
        spread_small = max(fused_small_k.values()) - min(fused_small_k.values())
        spread_large = max(fused_large_k.values()) - min(fused_large_k.values())
        self.assertGreater(spread_small, spread_large)


class WeightedFusionTests(unittest.TestCase):
    def test_unweighted_call_is_unchanged(self) -> None:
        ranking = [("a", 0.9), ("b", 0.5)]
        self.assertEqual(reciprocal_rank_fusion([ranking]), reciprocal_rank_fusion([ranking], weights=None))

    def test_weight_scales_a_routes_contribution(self) -> None:
        # Both routes agree "a" beats "b", but weighting bm25 higher should
        # widen the score gap, not just preserve the order.
        bm25 = [("a", 1.0), ("b", 0.5)]
        dense = [("a", 1.0), ("b", 0.5)]
        unweighted = dict(reciprocal_rank_fusion([bm25, dense]))
        weighted = dict(reciprocal_rank_fusion([bm25, dense], weights=[5.0, 1.0]))
        unweighted_gap = unweighted["a"] - unweighted["b"]
        weighted_gap = weighted["a"] - weighted["b"]
        self.assertGreater(weighted_gap, unweighted_gap)

    def test_zero_weight_route_is_effectively_ignored(self) -> None:
        bm25 = [("a", 1.0)]
        dense = [("b", 1.0)]
        fused = reciprocal_rank_fusion([bm25, dense], weights=[1.0, 0.0])
        ids = [doc_id for doc_id, _ in fused]
        self.assertEqual(ids[0], "a")
        self.assertEqual(dict(fused)["b"], 0.0)

    def test_mismatched_weights_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])


class WeightsForModeTests(unittest.TestCase):
    def test_buying_and_browsing_resolve_to_configured_weights(self) -> None:
        self.assertEqual(
            weights_for_mode("buying"),
            [BUYING_WEIGHTS[route] for route in ROUTE_ORDER],
        )
        self.assertEqual(
            weights_for_mode("browsing"),
            [BROWSING_WEIGHTS[route] for route in ROUTE_ORDER],
        )

    def test_route_order_is_bm25_then_dense(self) -> None:
        self.assertEqual(ROUTE_ORDER, ("bm25", "dense"))

    def test_buying_weights_favor_bm25_over_dense(self) -> None:
        # Per the build brief: buying leans on BM25 (+ the upstream hard
        # filter), dense is along only to catch nuance.
        self.assertGreater(BUYING_WEIGHTS["bm25"], BUYING_WEIGHTS["dense"])

    def test_browsing_weights_favor_dense_over_bm25(self) -> None:
        self.assertGreater(BROWSING_WEIGHTS["dense"], BROWSING_WEIGHTS["bm25"])

    def test_unknown_mode_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            weights_for_mode("not_a_real_mode")


if __name__ == "__main__":
    unittest.main()
