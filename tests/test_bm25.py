from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval.bm25 import BM25Index


def _write_catalog(root: Path, rows: list[dict]) -> Path:
    catalog_path = root / "catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return catalog_path


class BM25IndexTests(unittest.TestCase):
    def test_search_ranks_matching_product_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "Blue running shoe", "categories": ["Shoes"],
                 "features": [], "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "B", "title": "Red winter jacket", "categories": ["Jackets"],
                 "features": [], "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            results = index.search("running shoe", top_k=10)
            self.assertEqual(results[0][0], "A")

    def test_search_returns_empty_list_for_stopword_only_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "Blue shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            self.assertEqual(index.search("the a an", top_k=10), [])

    def test_search_respects_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"parent_asin": str(i), "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []}
                for i in range(5)
            ]
            catalog_path = _write_catalog(Path(directory), rows)
            index = BM25Index(catalog_path)
            self.assertEqual(len(index.search("running shoe", top_k=2)), 2)

    def test_candidate_ids_restricts_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "B", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "C", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            results = index.search("running shoe", top_k=10, candidate_ids={"A", "C"})
            self.assertEqual({doc_id for doc_id, _ in results}, {"A", "C"})

    def test_empty_candidate_ids_returns_empty_not_unscoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            self.assertEqual(index.search("running shoe", top_k=10, candidate_ids=set()), [])

    def test_none_candidate_ids_is_unscoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "B", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            results = index.search("running shoe", top_k=10, candidate_ids=None)
            self.assertEqual({doc_id for doc_id, _ in results}, {"A", "B"})

    def test_repeated_searches_with_different_candidate_ids_dont_leak(self) -> None:
        # The temp table is reused across calls -- make sure DELETE really
        # clears it rather than accumulating candidates from prior calls.
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "B", "title": "running shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            index.search("running shoe", top_k=10, candidate_ids={"A"})
            second = index.search("running shoe", top_k=10, candidate_ids={"B"})
            self.assertEqual({doc_id for doc_id, _ in second}, {"B"})

    def test_higher_score_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = _write_catalog(Path(directory), [
                {"parent_asin": "A", "title": "running running running shoe", "categories": [],
                 "features": [], "details": {}, "store": "Acme", "description": []},
                {"parent_asin": "B", "title": "shoe", "categories": [], "features": [],
                 "details": {}, "store": "Acme", "description": []},
            ])
            index = BM25Index(catalog_path)
            results = dict(index.search("running shoe", top_k=10))
            self.assertGreater(results["A"], results["B"])


if __name__ == "__main__":
    unittest.main()
