from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval.catalog_ids import CatalogIndex


class CatalogIndexTests(unittest.TestCase):
    def test_load_preserves_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            with catalog_path.open("w", encoding="utf-8") as handle:
                for asin in ["B", "A", "C"]:
                    handle.write(json.dumps({"parent_asin": asin}) + "\n")
            index = CatalogIndex.load(catalog_path)
            self.assertEqual(index.ids, ["B", "A", "C"])
            self.assertEqual(index.index_of("A"), 1)
            self.assertEqual(index.index_of("C"), 2)

    def test_index_of_returns_none_for_unknown_id(self) -> None:
        index = CatalogIndex(["A", "B"])
        self.assertIsNone(index.index_of("nope"))

    def test_load_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text('{"parent_asin": "A"}\n\n{"parent_asin": "B"}\n', encoding="utf-8")
            index = CatalogIndex.load(catalog_path)
            self.assertEqual(index.ids, ["A", "B"])

    def test_len(self) -> None:
        self.assertEqual(len(CatalogIndex(["A", "B", "C"])), 3)

    def test_index_order_is_stable_across_reloads(self) -> None:
        # Task 2's load-bearing assumption: index i must resolve to the
        # same product every time the catalog is loaded, since fusion,
        # the reranker, and whatever converts index -> parent_asin
        # downstream all have to agree on one catalog order.
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            with catalog_path.open("w", encoding="utf-8") as handle:
                for asin in ["X", "Y", "Z"]:
                    handle.write(json.dumps({"parent_asin": asin}) + "\n")
            first = CatalogIndex.load(catalog_path)
            second = CatalogIndex.load(catalog_path)
            self.assertEqual(first.ids, second.ids)
            for asin in ["X", "Y", "Z"]:
                self.assertEqual(first.index_of(asin), second.index_of(asin))


if __name__ == "__main__":
    unittest.main()
