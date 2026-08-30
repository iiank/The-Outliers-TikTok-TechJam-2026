import re
from typing import Any, Callable, Dict, List, Optional, Set, Union, Sequence, Tuple
import logging

from generation.weighted_entropy import WeightedEntropy
from reranker.reranker import Reranker, load_reranker_catalog
from retrieval.bm25 import BM25Index
from retrieval.catalog_ids import CatalogIndex
from retrieval.pipeline import Retriever
from retrieval.price_scoring import apply_target_price_scoring
from embed.store import load_store
from state.dialogue_state import budget_bounds

logger = logging.getLogger(__name__)

ALLOWED_ATTRIBUTES: Set[str] = {"category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other",}
# NOTE: Need to redefine this list with rationale
DEFAULT_ATTRIBUTE_PRIORITY: List[str] = ["material", "color", "size", "brand", "style", "feature", "use_case", "budget",]

"""
HARD FILTERING
"""

def is_category_satisfied(
    catalog_category: Optional[Union[List[str], str]],
    target_categories: List[str]
) -> bool:
    """
    Evaluates whether a product's catalog category satisfies state category constraints.
    Missing/null categories pass by default.
    """
    if not target_categories:
        return True

    if catalog_category is None:
        return True

    # Flatten catalog category to lowercased text
    if isinstance(catalog_category, list):
        if not catalog_category:
            return True
        cat_text = " ".join(str(c) for c in catalog_category).lower()
    else:
        cat_text = str(catalog_category).lower().strip()
        if not cat_text:
            return True

    # Check if item satisfies at least one target category constraint
    for target in target_categories:
        target_clean = str(target).lower().strip()
        if not target_clean:
            continue

        # Exact substring match
        if target_clean in cat_text:
            return True

        # all words in the target category must appear in path
        # e.g. target = "hiking boots" -> both "hiking" and "boots" must appear
        tokens = [t for t in re.findall(r"\w+", target_clean) if len(t) > 1]
        if tokens and all(t in cat_text for t in tokens):
            return True

    return False

def get_failed_hard_filter_asins(
    state: Any,
    reranker_catalog: List[Dict[str, Any]]
) -> List[str]:
    """
    Extracts category hard filters from state, applies them to reranker_catalog,
    and returns parent_asins that failed either constraint.
    """
    session_profile = state.get("session_profile", {})
    target_categories = [c for c in session_profile.get("category", []) if c]

    # If no hard filters exist, skip filtering
    if not target_categories:
        return []

    failed_asins: List[str] = []

    for item in reranker_catalog:
        parent_asin = item.get("parent_asin")
        if not parent_asin:
            continue

        item_category = item.get("category")

        # Evaluate Category constraint
        if target_categories and not is_category_satisfied(item_category, target_categories):
            failed_asins.append(parent_asin)
            continue

    return failed_asins

"""
SEARCH(STATE) -> List[str] TOP10, str ATTRIBUTE

Search and clarification orchestration pipeline.
Coordinates hard filtering, BM25 + semantic + RRF, cross-encoder reranking,
and entropy-based attribute inquiry.
"""

def _to_dict(state: Any) -> Dict[str, Any]:
    """Converts DialogueState object or dictionary into a standard Dict[str, Any]."""
    if isinstance(state, dict):
        return state
    if hasattr(state, "__dict__"):
        return state.__dict__
    return dict(state)


def build_retrieval_query(state_dict: Dict[str, Any]) -> str:
    """Flattens session_profile's disclosed values into a plain search string."""
    session_profile = state_dict.get("session_profile", {})
    terms: List[str] = []
    for values in session_profile.values():
        terms.extend(v for v in values if v)
    return " ".join(terms)


class SearchPipeline:
    """Encompasses catalog, retriever, reranker, and entropy modules."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        entropy_gen: Optional[WeightedEntropy] = None,
        catalog: Optional[List[Dict[str, Any]]] = None,
    ):
        # CatalogIndex is the shared parent_asin <-> index mapping.
        self.catalog_index = CatalogIndex.load("data/catalog.jsonl")

        if catalog is not None:
            self.catalog: List[Dict[str, Any]] = catalog
        else:
            _by_asin = {row["parent_asin"]: row for row in load_reranker_catalog()}
            self.catalog = [_by_asin[asin] for asin in self.catalog_index.ids]

        if retriever is None:
            bm25 = BM25Index("data/catalog.jsonl")
            dense = load_store()
            retriever = Retriever(bm25=bm25, dense=dense, catalog_index=self.catalog_index)
        self.retriever: Retriever = retriever

        self.reranker: Reranker = reranker or Reranker()
        self.entropy_gen: WeightedEntropy = entropy_gen or WeightedEntropy()

    def search(self, state: Any) -> Tuple[List[str], Optional[str]]:
        """
        Executes end-to-end multi-stage recommendation and question generation.

        Returns:     Tuple[List[str], Optional[str]]: (Top 10 recommended parent_asins, selected attribute to inquire, or None to show results without asking)
        """
        state_dict = _to_dict(state)
        session_profile = state_dict.get("session_profile", {})

        # ---------------------------------------------------------------------
        # 1. Hard Filtering
        # ---------------------------------------------------------------------
        failed_asins: List[str] = get_failed_hard_filter_asins(state_dict, self.catalog)

        # ---------------------------------------------------------------------
        # 2. Hybrid Retrieval (Pre-filtered candidate pool -> BM25 + Dense -> RRF)
        # ---------------------------------------------------------------------
        intent_mode = state_dict.get("intent") or "browsing"
        if intent_mode not in ("buying", "browsing"):
            intent_mode = "browsing"

        query_terms = build_retrieval_query(state_dict)

        retrieval_result = self.retriever.retrieve(
            query_terms=query_terms,
            mode=intent_mode,
            entropy_pool_size=500,
            reranker_pool_size=100,
            failed_asins=failed_asins,
        )

        # ---------------------------------------------------------------------
        # 3. Candidate Index Extraction
        # ---------------------------------------------------------------------
        reranker_indices: List[int] = [idx for idx, _ in retrieval_result.reranker_pool]

        entropy_indices: List[int] = []
        for item in retrieval_result.entropy_pool:
            idx = item.get("catalog_index")
            if idx is not None:
                entropy_indices.append(idx)

        # ---------------------------------------------------------------------
        # 4. Cross-Encoder Reranking -> Target-Price Scoring -> Top 10
        # ---------------------------------------------------------------------
        # Reranked at the full candidate-pool width, not just 10: target-price
        # scoring (below) needs room to move a candidate up from outside the
        # cutoff, which it can't do if the list is already truncated to 10.
        reranked_docs = self.reranker.rank_from_state(
            state=state_dict,
            candidate_indices=reranker_indices,
            catalog=self.catalog,
            top_k=len(reranker_indices),
        )

        # Soft price adjustment (target-price-scoring-spec.md), never a hard
        # filter -- multiplies each candidate's rerank score by a
        # closeness-to-target factor, then re-sorts. A no-op when the
        # customer hasn't stated a target price this turn.
        target_price = budget_bounds(session_profile).get("target_price")
        price_scored_docs = apply_target_price_scoring(reranked_docs, target_price=target_price)

        top_10_asins = [
            doc["parent_asin"] for doc in price_scored_docs[:10] if "parent_asin" in doc
        ]

        # ---------------------------------------------------------------------
        # 5. Question Generation via Weighted Entropy
        # ---------------------------------------------------------------------
        selected_attribute: Optional[str] = None
        needs_heuristic_fallback = False
        try:
            selected_attribute, _scored = self.entropy_gen.select(
                state=state_dict,
                top_500_candidate_indices=entropy_indices,
            )
            if selected_attribute is not None and selected_attribute not in ALLOWED_ATTRIBUTES:
                selected_attribute = None
                needs_heuristic_fallback = True
        except Exception as e:
            logger.warning(f"WeightedEntropy generation failed: {e}. Falling back to heuristic.")
            needs_heuristic_fallback = True

        # Only fall back to the priority-list heuristic when select() itself
        # broke or returned something outside the contract's enum -- a clean
        # None from select() (nothing left worth asking, or fatigued past
        # HARD_REFUSAL_LIMIT) must be allowed through as "show results".
        if selected_attribute is None and needs_heuristic_fallback:
            for attr in DEFAULT_ATTRIBUTE_PRIORITY:
                if not session_profile.get(attr):
                    selected_attribute = attr
                    break
            if selected_attribute is None:
                selected_attribute = "feature"

        return top_10_asins, selected_attribute

# -----------------------------------------------------------------------------
# Module Singleton Wrapper
# -----------------------------------------------------------------------------
_PIPELINE_INSTANCE: Optional[SearchPipeline] = None

def search(state: Any) -> Tuple[List[str], Optional[str]]:
    """
    Overarching search API called by agent.py.
    Maintains persistent memory instances of models and catalog.
    """
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = SearchPipeline()
    return _PIPELINE_INSTANCE.search(state)