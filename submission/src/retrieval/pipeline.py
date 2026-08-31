"""Wires the BM25 and dense routes together through weighted RRF.

``Retriever.retrieve(query_terms, mode, failed_asins=None)`` is the single
entry point.

1) fusion is weighted differently for ``"buying"`` vs ``"browsing"``
(see ``retrieval.rrf.weights_for_mode``).
2) one retrieval pass produces two slices: a top-500 pool for entropy calculations,
and a top-100 ``(catalog_index, rrf_score)`` pool for the reranker.
3) ``search.py``'s ``get_failed_hard_filter_asins()`` executes hard
filtering on category and returns ``parent_asin``s to exclude. This module
takes the list to exclude in both keyword/semantic search routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from .bm25 import BM25Index
from .catalog_ids import CatalogIndex
from .rrf import ROUTE_ORDER, reciprocal_rank_fusion, weights_for_mode

__all__ = ["Retriever", "RetrievalResult", "DenseRoute", "DEFAULT_ENTROPY_POOL_SIZE", "DEFAULT_RERANKER_POOL_SIZE"]

DEFAULT_ENTROPY_POOL_SIZE = 500
DEFAULT_RERANKER_POOL_SIZE = 100

# How much deeper to query the dense route than the target pool size
# Compensates for the overfetch+filter-in-Python approach
_DENSE_OVERFETCH_MULTIPLIER = 5


class DenseRoute(Protocol):
    """Structural contract ``embed.store.VectorStore`` satisfies."""

    def search(self, query: str, top_k: int) -> Sequence[Tuple[str, float]]: ...


@dataclass
class RetrievalResult:
    """Two slices of the one retrieval pass.

    ``entropy_pool``: up to ``entropy_pool_size`` results, each
    ``{"parent_asin": str, "catalog_index": int | None, "score": float}``

    ``reranker_pool``: up to ``reranker_pool_size`` ``(catalog_index,
    rrf_score)`` pairs, in ranked order -- catalog indices, not
    ``parent_asin`` strings (mappings resolved using``CatalogIndex``).
    """

    entropy_pool: List[Dict[str, Any]] = field(default_factory=list)
    reranker_pool: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class Retriever:
    """Fuses BM25 + dense via mode-weighted RRF"""

    bm25: BM25Index
    dense: DenseRoute
    catalog_index: CatalogIndex
    rrf_k: int = 60

    def retrieve(
        self,
        query_terms: str,
        mode: str,
        entropy_pool_size: int = DEFAULT_ENTROPY_POOL_SIZE,
        reranker_pool_size: int = DEFAULT_RERANKER_POOL_SIZE,
        failed_asins: Optional[Sequence[str]] = None,
    ) -> RetrievalResult:
        """Runs full retrieval + fusion pipeline for one turn.

        ``query_terms`` built from state is input for both routes

        ``failed_asins``, if given, is a list of ``parent_asin``s to
        exclude before either route searches

        ``mode`` must be ``"buying"`` or ``"browsing"``
        """
        pool_size = max(entropy_pool_size, reranker_pool_size)
        candidate_ids = self._resolve_candidate_ids(failed_asins)

        bm25_ranking = self.bm25.search(query_terms, pool_size, candidate_ids=candidate_ids)
        dense_ranking = self._search_dense(query_terms, pool_size, candidate_ids)

        weights = weights_for_mode(mode, ROUTE_ORDER)
        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], weights=weights, rrf_k=self.rrf_k)

        entropy_pool = [
            {
                "parent_asin": parent_asin,
                "catalog_index": self.catalog_index.index_of(parent_asin),
                "score": score,
            }
            for parent_asin, score in fused[:entropy_pool_size]
        ]

        reranker_pool: List[Tuple[int, float]] = []
        for parent_asin, score in fused[:reranker_pool_size]:
            index = self.catalog_index.index_of(parent_asin)
            if index is not None:
                reranker_pool.append((index, score))

        return RetrievalResult(entropy_pool=entropy_pool, reranker_pool=reranker_pool)

    def _resolve_candidate_ids(self, failed_asins: Optional[Sequence[str]]) -> Optional[Set[str]]:
        """``failed_asins`` -> allowed-ID set (catalog minus excluded), or ``None`` (unscoped)."""
        if not failed_asins:
            return None
        excluded = set(failed_asins)
        return {asin for asin in self.catalog_index.ids if asin not in excluded}

    def _search_dense(
        self,
        query_terms: str,
        top_k: int,
        candidate_ids: Optional[Set[str]],
    ) -> List[Tuple[str, float]]:
        if candidate_ids is None:
            return list(self.dense.search(query_terms, top_k))
        if not candidate_ids:
            return []
        overfetch_k = top_k * _DENSE_OVERFETCH_MULTIPLIER
        raw = self.dense.search(query_terms, overfetch_k)
        filtered = [(doc_id, score) for doc_id, score in raw if doc_id in candidate_ids]
        return filtered[:top_k]
