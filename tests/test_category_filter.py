from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval.category_filter import CategoryLookup, extract_coarse_category


class ExtractCoarseCategoryTests(unittest.TestCase):
    def test_browsing_template_extracts_category(self) -> None:
        message = "I'm looking for running shoes, but I'm still exploring."
        self.assertEqual(extract_coarse_category(message), "running shoes")

    def test_buying_template_extracts_category(self) -> None:
        message = "I'm looking for hiking boots. A key requirement is: waterproof."
        self.assertEqual(extract_coarse_category(message), "hiking boots")

    def test_missing_apostrophe_still_matches(self) -> None:
        message = "im looking for dresses, but im still exploring."
        self.assertEqual(extract_coarse_category(message), "dresses")

    def test_freeform_message_with_wrong_period_placement_returns_none(self) -> None:
        # The exact adversarial example from the build brief: a period
        # lands in the wrong place and neither recognized suffix appears,
        # so this must NOT extract "something to buy since im going to a party".
        message = (
            "im looking for something to buy since im going to a party. "
            "im just browsing. help me find something"
        )
        self.assertIsNone(extract_coarse_category(message))

    def test_intent_override_shape_returns_none(self) -> None:
        # "I'm looking for {category}. {old_value}" is structurally
        # ambiguous (old_value is arbitrary text) -- must not guess.
        message = "I'm looking for running shoes. I prefer a different style."
        self.assertIsNone(extract_coarse_category(message))

    def test_unrelated_message_returns_none(self) -> None:
        self.assertIsNone(extract_coarse_category("Do you have this in blue?"))

    def test_empty_message_returns_none(self) -> None:
        self.assertIsNone(extract_coarse_category(""))


class CategoryLookupTests(unittest.TestCase):
    def test_matching_ids_requires_all_phrase_words(self) -> None:
        lookup = CategoryLookup({
            "A": {"shoes", "running", "mens"},
            "B": {"shoes", "dress"},
            "C": {"jewelry"},
        })
        self.assertEqual(lookup.matching_ids("running shoes"), {"A"})

    def test_matching_ids_is_case_and_order_insensitive(self) -> None:
        lookup = CategoryLookup({"A": {"shoes", "running"}})
        self.assertEqual(lookup.matching_ids("Running Shoes"), {"A"})
        self.assertEqual(lookup.matching_ids("shoes running"), {"A"})

    def test_no_matches_returns_empty_set_not_everything(self) -> None:
        lookup = CategoryLookup({"A": {"jewelry"}, "B": {"dresses"}})
        self.assertEqual(lookup.matching_ids("hiking boots"), set())

    def test_empty_phrase_returns_empty_set(self) -> None:
        lookup = CategoryLookup({"A": {"shoes"}})
        self.assertEqual(lookup.matching_ids(""), set())

    def test_load_builds_tokens_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            rows = [
                {"parent_asin": "A", "categories": ["Clothing", "Shoes", "Running"]},
                {"parent_asin": "B", "categories": ["Clothing", "Dresses"]},
            ]
            with catalog_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            lookup = CategoryLookup.load(catalog_path)
            self.assertEqual(lookup.matching_ids("running shoes"), {"A"})
            self.assertEqual(lookup.matching_ids("dresses"), {"B"})


if __name__ == "__main__":
    unittest.main()
