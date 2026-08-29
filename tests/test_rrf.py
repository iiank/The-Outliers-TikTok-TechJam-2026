from __future__ import annotations

import unittest

from retrieval.rrf import reciprocal_rank_fusion


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


if __name__ == "__main__":
    unittest.main()
