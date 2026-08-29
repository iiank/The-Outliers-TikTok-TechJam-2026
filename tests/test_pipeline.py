from __future__ import annotations

import unittest

from retrieval.catalog_ids import CatalogIndex
from retrieval.pipeline import Retriever


class FakeRoute:
    """A canned ranking, ignoring the query text -- enough to test fusion wiring."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self._ranking = ranking

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self._ranking[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog_index = CatalogIndex(["a", "b", "c", "d", "e"])

    def test_retrieve_returns_parent_asin_and_catalog_index(self) -> None:
        dense = FakeRoute([("a", 0.9), ("b", 0.5)])
        bm25 = FakeRoute([("b", 10.0), ("a", 8.0)])
        retriever = Retriever(routes=[dense, bm25], catalog_index=self.catalog_index)
        results = retriever.retrieve("query", top_k=2)
        self.assertEqual([r["parent_asin"] for r in results], ["a", "b"])
        self.assertEqual([r["catalog_index"] for r in results], [0, 1])

    def test_top_k_is_configurable(self) -> None:
        dense = FakeRoute([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 1.0)])
        retriever = Retriever(routes=[dense], catalog_index=self.catalog_index)
        self.assertEqual(len(retriever.retrieve("query", top_k=3)), 3)
        self.assertEqual(len(retriever.retrieve("query", top_k=5)), 5)

    def test_pool_multiplier_widens_the_per_route_query(self) -> None:
        seen_top_k: list[int] = []

        class RecordingRoute:
            def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
                seen_top_k.append(top_k)
                return []

        retriever = Retriever(routes=[RecordingRoute()], catalog_index=self.catalog_index, pool_multiplier=4)
        retriever.retrieve("query", top_k=100)
        self.assertEqual(seen_top_k, [400])

    def test_unknown_id_gets_none_catalog_index(self) -> None:
        dense = FakeRoute([("not_in_catalog", 1.0)])
        retriever = Retriever(routes=[dense], catalog_index=self.catalog_index)
        results = retriever.retrieve("query", top_k=1)
        self.assertIsNone(results[0]["catalog_index"])

    def test_top_k_zero_returns_empty(self) -> None:
        retriever = Retriever(routes=[FakeRoute([("a", 1.0)])], catalog_index=self.catalog_index)
        self.assertEqual(retriever.retrieve("query", top_k=0), [])

    def test_single_route_still_works(self) -> None:
        dense = FakeRoute([("c", 1.0), ("a", 0.5)])
        retriever = Retriever(routes=[dense], catalog_index=self.catalog_index)
        results = retriever.retrieve("query", top_k=2)
        self.assertEqual([r["parent_asin"] for r in results], ["c", "a"])


if __name__ == "__main__":
    unittest.main()
