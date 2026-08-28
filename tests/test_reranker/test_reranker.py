# tests/test_reranker.py
import os
import time
import unittest
from typing import Any, Dict, List
import torch

from submission.src.reranker.reranker import ProductReranker, build_reranker_query, load_catalog


class TestProductReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the reranker once across tests to avoid redundant model loading."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.reranker = ProductReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=cls.device,
            max_length=192,
            batch_size=64,
            use_fp16=True,
        )

        cls.sample_state: Dict[str, Any] = {
            "session_id": "test-session-001",
            "turn": 4,
            "session_profile": {
                "category": ["hiking boots"],
                "material": ["full-grain leather"],
                "color": ["black"],
                "size": ["10"],
                "style": ["waterproof outdoor"],
                "brand": ["Timberland"],
                "budget": ["<=120"],
                "feature": ["breathable mesh lining", "vibram rubber sole"],
                "use_case": ["trail hiking"],
                "other": [],
                "rejected": ["running shoes", "synthetic"],
            },
            "user_profile": {
                "purchase_frequency": "frequent",
                "average_prior_rating": 4.8,
                "rating_style": "constructive",
                "preference_tags": ["durability", "comfort", "arch support"],
                "summary": "Values long-lasting durability and arch support.",
            },
            "previous_top_10": ["B001", "B002"],
            "conflicts_with_previous": False,
        }

        cls.mock_catalog: List[Dict[str, Any]] = [
            {
                "parent_asin": "B00_TARGET",
                "title": "Timberland Mens Waterproof Leather Hiking Boots",
                "document": "Title: Timberland Mens Waterproof Leather Hiking Boots | Category: Boots, Hiking | Material: Full-grain leather | Features: Vibram rubber sole, breathable mesh lining | Preferences: durability, comfort",
            },
            {
                "parent_asin": "B01_PARTIAL",
                "title": "Mens Black Synthetic Trail Running Shoes",
                "document": "Title: Mens Black Synthetic Trail Running Shoes | Category: Shoes, Trail Running | Material: Mesh, Synthetic | Features: lightweight, breathable | Preferences: comfort",
            },
            {
                "parent_asin": "B02_IRRELEVANT_1",
                "title": "Floral Print Summer Silk Dress",
                "document": "Title: Floral Print Summer Silk Dress | Category: Dresses, Casual | Material: 100% Silk | Features: sleeveless, knee-length",
            },
            {
                "parent_asin": "B03_IRRELEVANT_2",
                "title": "Stainless Steel Kitchen Chef Knife 8 Inch",
                "document": "Title: Stainless Steel Kitchen Chef Knife 8 Inch | Category: Kitchen, Cutlery | Material: High Carbon Stainless Steel",
            },
        ]

    # -------------------------------------------------------------------------
    # 1. End-to-End Functionality Tests
    # -------------------------------------------------------------------------
    def test_build_reranker_query_structure(self):
        """Verifies state parsing ignores rejected/budget metadata and includes relevant slots."""
        query = build_reranker_query(self.sample_state)

        # High-signal terms must be present
        self.assertIn("hiking boots", query)
        self.assertIn("full-grain leather", query)
        self.assertIn("black", query)
        self.assertIn("durability", query)

        # Operational/rejected metadata must be dropped
        self.assertNotIn("<=120", query)
        self.assertNotIn("running shoes", query)
        self.assertNotIn("test-session-001", query)

    def test_end_to_end_ranking_pipeline(self):
        """Tests end-to-end execution of candidate ranking from state."""
        candidate_indices = list(range(len(self.mock_catalog)))
        ranked_results = self.reranker.rank_from_state(
            state=self.sample_state,
            candidate_indices=candidate_indices,
            catalog=self.mock_catalog,
            top_k=2,
        )

        self.assertEqual(len(ranked_results), 2)
        self.assertIn("score", ranked_results[0])
        self.assertIn("parent_asin", ranked_results[0])
        self.assertIn("index", ranked_results[0])
        # Verify scores are sorted in descending order
        self.assertGreaterEqual(ranked_results[0]["score"], ranked_results[1]["score"])

    # -------------------------------------------------------------------------
    # 2. Hardware / GPU Acceleration Tests
    # -------------------------------------------------------------------------
    def test_device_placement_and_precision(self):
        """Verifies the model is placed on CUDA when available and checks precision."""
        expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        model_device = self.reranker.model.model.device.type

        self.assertEqual(
            model_device,
            expected_device_type,
            f"Expected model on {expected_device_type}, but found on {model_device}",
        )

        if expected_device_type == "cuda":
            param_dtype = next(self.reranker.model.model.parameters()).dtype
            self.assertEqual(
                param_dtype,
                torch.float16,
                "Model parameters should be in float16 when use_fp16=True on CUDA.",
            )

    # -------------------------------------------------------------------------
    # 3. Accuracy and Goal Product Matching Tests
    # -------------------------------------------------------------------------
    def test_target_item_scores_highest(self):
        """Tests if the exact matching goal item outscores partially matching and irrelevant items."""
        query = build_reranker_query(self.sample_state)
        ranked = self.reranker.rank(query=query, candidates=self.mock_catalog, top_k=4)

        top_asin = ranked[0]["parent_asin"]
        self.assertEqual(
            top_asin,
            "B00_TARGET",
            f"Expected B00_TARGET as #1 recommendation, but got {top_asin}",
        )

        # Quantifiable margin between target and completely irrelevant product
        target_score = ranked[0]["score"]
        irrelevant_score = [r["score"] for r in ranked if r["parent_asin"] == "B02_IRRELEVANT_1"][0]
        self.assertGreater(
            target_score - irrelevant_score,
            3.0,
            "Target item score margin over irrelevant item is too narrow.",
        )

    # -------------------------------------------------------------------------
    # 4. Ranking Improvement (MRR / Rank Shift) Tests
    # -------------------------------------------------------------------------
    def test_reranker_improves_retrieval_order(self):
        """Tests if the reranker rescues a relevant target positioned low in the retrieval candidate list."""
        # Simulate retrieval output where target item is buried at the end (index 49)
        distractor_item = {
            "parent_asin": "B_DISTRACTOR",
            "title": "Random Product",
            "document": "Title: Generic Cotton Casual Socks | Category: Clothing | Material: Cotton",
        }
        retrieval_candidates = [distractor_item.copy() for _ in range(49)]
        target_item = self.mock_catalog[0].copy()
        retrieval_candidates.append(target_item)  # Target is at index 49 (Rank 50)

        initial_target_rank = 50
        initial_mrr = 1.0 / initial_target_rank

        query = build_reranker_query(self.sample_state)
        reranked = self.reranker.rank(
            query=query, candidates=retrieval_candidates, top_k=10
        )

        reranked_asins = [item["parent_asin"] for item in reranked]
        self.assertIn(
            "B00_TARGET",
            reranked_asins,
            "Target item failed to reach Top 10 after reranking.",
        )

        new_target_rank = reranked_asins.index("B00_TARGET") + 1
        new_mrr = 1.0 / new_target_rank

        # Verify ranking metric improvement
        self.assertLess(new_target_rank, initial_target_rank)
        self.assertGreater(new_mrr, initial_mrr)