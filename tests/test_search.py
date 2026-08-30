from __future__ import annotations

import importlib
import unittest

# NOTE: `import search.search as search_mod` looks equivalent but is not --
# `submission/src/search/__init__.py` does `from .search import search`,
# which rebinds the *attribute* `search` on the `search` package to that
# function. Since `import a.b as x` is sugar for `import a.b; x = a.b`, the
# trailing attribute access resolves to the shadowed function, not the
# submodule. importlib.import_module() goes through sys.modules directly and
# is unaffected by the shadowing.
search_mod = importlib.import_module("search.search")


class _FakeRetrievalResult:
    def __init__(self, size: int = 5) -> None:
        self.reranker_pool = [(i, 1.0) for i in range(size)]
        self.entropy_pool = [{"catalog_index": i} for i in range(size)]


class _FakeRetriever:
    def retrieve(self, **kwargs):
        return _FakeRetrievalResult()


class _FakeReranker:
    def rank_from_state(self, state, candidate_indices, catalog, top_k):
        return [{"parent_asin": f"B{idx}"} for idx in candidate_indices]


def _make_pipeline(entropy_gen) -> search_mod.SearchPipeline:
    pipeline = search_mod.SearchPipeline.__new__(search_mod.SearchPipeline)
    pipeline.catalog_index = None
    pipeline.catalog = []
    pipeline.retriever = _FakeRetriever()
    pipeline.reranker = _FakeReranker()
    pipeline.entropy_gen = entropy_gen
    return pipeline


class SelectGateWiringTests(unittest.TestCase):
    """B2 wiring: search.py must let a clean None through, but still
    fall back to the priority-list heuristic when select() itself breaks."""

    def setUp(self) -> None:
        search_mod.get_failed_hard_filter_asins = lambda state, catalog: []
        search_mod.apply_target_price_scoring = lambda docs, target_price: docs
        search_mod.budget_bounds = lambda profile: {}

    def test_clean_none_from_select_propagates(self) -> None:
        class _EntropyGen:
            def select(self, state, top_500_candidate_indices):
                return None, []

        pipeline = _make_pipeline(_EntropyGen())
        _, attr = pipeline.search({"session_profile": {}, "turn": 2})
        self.assertIsNone(attr)

    def test_exception_falls_back_to_heuristic(self) -> None:
        class _EntropyGen:
            def select(self, state, top_500_candidate_indices):
                raise RuntimeError("boom")

        pipeline = _make_pipeline(_EntropyGen())
        _, attr = pipeline.search({"session_profile": {}, "turn": 2})
        self.assertTrue(attr in search_mod.DEFAULT_ATTRIBUTE_PRIORITY or attr == "feature")

    def test_out_of_enum_attribute_falls_back_to_heuristic(self) -> None:
        class _EntropyGen:
            def select(self, state, top_500_candidate_indices):
                return "not_a_real_attribute", []

        pipeline = _make_pipeline(_EntropyGen())
        _, attr = pipeline.search({"session_profile": {}, "turn": 2})
        self.assertTrue(attr in search_mod.DEFAULT_ATTRIBUTE_PRIORITY or attr == "feature")

    def test_heuristic_fallback_skips_already_filled_attributes(self) -> None:
        class _EntropyGen:
            def select(self, state, top_500_candidate_indices):
                raise RuntimeError("boom")

        pipeline = _make_pipeline(_EntropyGen())
        filled = {attr: ["x"] for attr in search_mod.DEFAULT_ATTRIBUTE_PRIORITY[:-1]}
        _, attr = pipeline.search({"session_profile": filled, "turn": 2})
        self.assertEqual(attr, search_mod.DEFAULT_ATTRIBUTE_PRIORITY[-1])


if __name__ == "__main__":
    unittest.main()
