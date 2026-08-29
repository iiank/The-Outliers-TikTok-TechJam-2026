from __future__ import annotations

import unittest

from retrieval.catalog_ids import CatalogIndex
from retrieval.category_filter import CategoryLookup
from retrieval.pipeline import DEFAULT_ENTROPY_POOL_SIZE, DEFAULT_RERANKER_POOL_SIZE, Retriever


class FakeRoute:
    """A canned ranking, ignoring the query text -- enough to test fusion wiring."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self._ranking = ranking
        self.seen_top_k: list[int] = []
        self.seen_candidate_ids: list[set[str] | None] = []

    def search(self, query: str, top_k: int, candidate_ids: set[str] | None = None) -> list[tuple[str, float]]:
        self.seen_top_k.append(top_k)
        self.seen_candidate_ids.append(candidate_ids)
        ranking = self._ranking
        if candidate_ids is not None:
            ranking = [(doc_id, score) for doc_id, score in ranking if doc_id in candidate_ids]
        return ranking[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_index = CatalogIndex(["a", "b", "c", "d", "e"])

    def test_retrieve_returns_entropy_and_reranker_pools(self) -> None:
        bm25 = FakeRoute([("a", 0.9), ("b", 0.5)])
        dense = FakeRoute([("b", 10.0), ("a", 8.0)])
        retriever = Retriever(bm25=bm25, dense=dense, catalog_index=self.catalog_index)
        result = retriever.retrieve("query", mode="buying")
        self.assertEqual([r["parent_asin"] for r in result.entropy_pool], ["a", "b"])
        self.assertEqual([r["catalog_index"] for r in result.entropy_pool], [0, 1])
        self.assertEqual([idx for idx, _score in result.reranker_pool], [0, 1])

    def test_reranker_pool_is_index_score_pairs(self) -> None:
        bm25 = FakeRoute([("c", 1.0)])
        dense = FakeRoute([])
        retriever = Retriever(bm25=bm25, dense=dense, catalog_index=self.catalog_index)
        result = retriever.retrieve("query", mode="buying")
        self.assertEqual(len(result.reranker_pool), 1)
        index, score = result.reranker_pool[0]
        self.assertEqual(index, 2)
        self.assertIsInstance(score, float)

    def test_pool_sizes_are_configurable(self) -> None:
        bm25 = FakeRoute([(str(i), 1.0) for i in range(600)])
        catalog = CatalogIndex([str(i) for i in range(600)])
        retriever = Retriever(bm25=bm25, dense=FakeRoute([]), catalog_index=catalog)
        result = retriever.retrieve("query", mode="buying", entropy_pool_size=50, reranker_pool_size=10)
        self.assertEqual(len(result.entropy_pool), 50)
        self.assertEqual(len(result.reranker_pool), 10)

    def test_default_pool_sizes_are_500_and_100(self) -> None:
        self.assertEqual(DEFAULT_ENTROPY_POOL_SIZE, 500)
        self.assertEqual(DEFAULT_RERANKER_POOL_SIZE, 100)

    def test_reranker_pool_is_prefix_of_entropy_pool(self) -> None:
        bm25 = FakeRoute([(str(i), 1.0 / (i + 1)) for i in range(50)])
        catalog = CatalogIndex([str(i) for i in range(50)])
        retriever = Retriever(bm25=bm25, dense=FakeRoute([]), catalog_index=catalog)
        result = retriever.retrieve("query", mode="buying", entropy_pool_size=20, reranker_pool_size=5)
        entropy_ids = [r["parent_asin"] for r in result.entropy_pool[:5]]
        reranker_ids = [catalog.ids[idx] for idx, _score in result.reranker_pool]
        self.assertEqual(entropy_ids, reranker_ids)

    def test_buying_and_browsing_modes_can_produce_different_order(self) -> None:
        # bm25 favors "a"; dense favors "b". Buying weights bm25 higher,
        # browsing weights dense higher -- the two modes should be able to
        # disagree on which one wins.
        bm25 = FakeRoute([("a", 1.0), ("b", 0.9)])
        dense = FakeRoute([("b", 1.0), ("a", 0.9)])
        retriever = Retriever(bm25=bm25, dense=dense, catalog_index=self.catalog_index)
        buying = [r["parent_asin"] for r in retriever.retrieve("query", mode="buying").entropy_pool]
        browsing = [r["parent_asin"] for r in retriever.retrieve("query", mode="browsing").entropy_pool]
        self.assertEqual(buying[0], "a")
        self.assertEqual(browsing[0], "b")

    def test_unknown_mode_raises(self) -> None:
        retriever = Retriever(bm25=FakeRoute([]), dense=FakeRoute([]), catalog_index=self.catalog_index)
        with self.assertRaises(KeyError):
            retriever.retrieve("query", mode="not_a_real_mode")

    def test_no_message_means_unscoped_search(self) -> None:
        bm25 = FakeRoute([("a", 1.0)])
        lookup = CategoryLookup({"a": {"shoes"}, "b": {"dresses"}})
        retriever = Retriever(
            bm25=bm25, dense=FakeRoute([]), catalog_index=self.catalog_index, category_lookup=lookup
        )
        retriever.retrieve("query", mode="buying")  # message=None by default
        self.assertEqual(bm25.seen_candidate_ids, [None])

    def test_recognized_category_message_restricts_both_routes(self) -> None:
        bm25 = FakeRoute([("a", 1.0), ("b", 0.5)])
        dense = FakeRoute([("a", 1.0), ("b", 0.5)])
        lookup = CategoryLookup({"a": {"shoes", "running"}, "b": {"dresses"}})
        retriever = Retriever(
            bm25=bm25, dense=dense, catalog_index=self.catalog_index, category_lookup=lookup
        )
        result = retriever.retrieve(
            "query", mode="buying", message="I'm looking for running shoes, but I'm still exploring."
        )
        self.assertEqual([r["parent_asin"] for r in result.entropy_pool], ["a"])

    def test_unrecognized_message_shape_is_unscoped(self) -> None:
        bm25 = FakeRoute([("a", 1.0), ("b", 0.5)])
        lookup = CategoryLookup({"a": {"shoes"}, "b": {"dresses"}})
        retriever = Retriever(
            bm25=bm25, dense=FakeRoute([]), catalog_index=self.catalog_index, category_lookup=lookup
        )
        message = "im looking for something to buy since im going to a party. im just browsing. help me find something"
        result = retriever.retrieve("query", mode="buying", message=message)
        self.assertEqual({r["parent_asin"] for r in result.entropy_pool}, {"a", "b"})

    def test_dense_route_overfetches_and_filters_when_category_active(self) -> None:
        dense = FakeRoute([("a", 1.0), ("b", 0.9), ("c", 0.8)])
        lookup = CategoryLookup({"a": {"shoes"}, "b": {"shoes"}, "c": {"dresses"}})
        retriever = Retriever(
            bm25=FakeRoute([]), dense=dense, catalog_index=self.catalog_index, category_lookup=lookup
        )
        result = retriever.retrieve(
            "query", mode="buying", message="I'm looking for shoes, but I'm still exploring.",
            entropy_pool_size=2, reranker_pool_size=2,
        )
        # dense.search was asked for MORE than 2 (overfetch), not exactly 2 --
        # candidate_ids isn't passed to the dense route's own search() call.
        self.assertGreater(dense.seen_top_k[0], 2)
        self.assertEqual({r["parent_asin"] for r in result.entropy_pool}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
