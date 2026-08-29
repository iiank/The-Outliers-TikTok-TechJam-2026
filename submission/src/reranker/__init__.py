"""
Reranker: Uses a cross-encoder (default ms-marco-MiniLM-L-6-v2) with state_dict, candidate_indices,
reranker_catalog, and top_k as input. Outputs top_k of reranked documents.
"""

from src.reranker.reranker import (
    build_reranker_query,
    load_reranker_catalog,
    Reranker,
)

__all__ = [
    "build_reranker_query",
    "load_reranker_catalog",
    "Reranker",
    "rank_from_state",
]
