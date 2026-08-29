"""Wires the BM25 and dense routes together through weighted RRF.

``Retriever.retrieve(query_terms, mode, message=None)`` is the single
entry point. Per the team's Aug 2026 build brief:

* Task 1 -- fusion is weighted differently for ``"buying"`` vs
  ``"browsing"`` (see ``retrieval.rrf.weights_for_mode``).
* Task 2 -- one fused computation produces two slices: a top-500 pool for
  the entropy/ask-attribute policy, and a top-100 ``(catalog_index,
  rrf_score)`` pool for the reranker. Both are prefixes of the same fused
  list, not two separate retrieval passes.
* Task 3 -- when a coarse category is extractable from ``message``, both
  routes search only within matching products, applied identically to
  both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from .bm25 import BM25Index
from .category_filter import CategoryLookup, extract_coarse_category
from .catalog_ids import CatalogIndex
from .rrf import ROUTE_ORDER, reciprocal_rank_fusion, weights_for_mode

__all__ = ["Retriever", "RetrievalResult", "DenseRoute", "DEFAULT_ENTROPY_POOL_SIZE", "DEFAULT_RERANKER_POOL_SIZE"]

#: Named config, not magic numbers -- per the build brief, 500/100 are
#: adjustable and only worth revisiting if there's spare time.
DEFAULT_ENTROPY_POOL_SIZE = 500
DEFAULT_RERANKER_POOL_SIZE = 100

#: How much deeper to query the dense route than the target pool size
#: when a category filter is active, to compensate for the overfetch+
#: filter-in-Python approach (see module docstring / build discussion):
#: VectorStore.search() has no candidate-ID parameter, so restricting it
#: to a category means asking for more than needed, then discarding
#: anything outside the allowed set. A rare category can still come back
#: short of the target pool size -- that's a known, accepted limit of
#: this approach, not a bug.
_DENSE_OVERFETCH_MULTIPLIER = 5


class DenseRoute(Protocol):
    """Structural contract ``embed.store.VectorStore`` satisfies.

    Deliberately has no ``candidate_ids`` parameter -- unlike
    :meth:`BM25Index.search`, the dense route is never asked to filter
    itself; :meth:`Retriever._search_dense` overfetches and filters in
    Python instead, so this module never needs to touch ``embed/store.py``.
    """

    def search(self, query: str, top_k: int) -> Sequence[Tuple[str, float]]: ...


@dataclass
class RetrievalResult:
    """Task 2's hand-off shape: two slices of one fused computation.

    ``entropy_pool``: up to ``entropy_pool_size`` results, each
    ``{"parent_asin": str, "catalog_index": int | None, "score": float}``
    -- the live candidate pool ``policy.ask_attribute``'s entropy
    calculation runs over.

    ``reranker_pool``: up to ``reranker_pool_size`` ``(catalog_index,
    rrf_score)`` pairs, in ranked order -- catalog indices, not
    ``parent_asin`` strings, per the confirmed reranker contract.
    Index -> ``parent_asin`` conversion happens downstream, outside this
    module (see the load-bearing assumption in the build brief: every
    consumer of an index must resolve it against the same catalog order,
    which is exactly what ``CatalogIndex`` is for).
    """

    entropy_pool: List[Dict[str, Any]] = field(default_factory=list)
    reranker_pool: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class Retriever:
    """Fuses BM25 + dense via mode-weighted RRF, behind a shared category pre-filter."""

    bm25: BM25Index
    dense: DenseRoute
    catalog_index: CatalogIndex
    category_lookup: Optional[CategoryLookup] = None
    rrf_k: int = 60

    def retrieve(
        self,
        query_terms: str,
        mode: str,
        message: Optional[str] = None,
        entropy_pool_size: int = DEFAULT_ENTROPY_POOL_SIZE,
        reranker_pool_size: int = DEFAULT_RERANKER_POOL_SIZE,
    ) -> RetrievalResult:
        """Run the full Task 1 + 2 + 3 pipeline for one turn.

        ``query_terms`` is what both routes search with -- building it
        from the conversation's accumulated state (vs. just this turn's
        raw text) is the caller's responsibility, not this module's.

        ``message`` is the raw customer message, used only for Task 3's
        coarse-category extraction (a template match against message
        *shape*, which ``query_terms`` -- already reduced to keywords --
        can't support). Pass ``None`` to skip category extraction
        entirely (equivalent to a message that doesn't match any
        template): both routes then search unscoped.

        ``mode`` must be ``"buying"`` or ``"browsing"`` -- see
        ``retrieval.rrf.weights_for_mode``.
        """
        pool_size = max(entropy_pool_size, reranker_pool_size)
        candidate_ids = self._resolve_candidate_ids(message)

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

    def _resolve_candidate_ids(self, message: Optional[str]) -> Optional[Set[str]]:
        """Task 3: category phrase -> allowed-ID set, or ``None`` (unscoped).

        Returns ``None`` -- not an empty set -- whenever the filter
        shouldn't apply at all: no message, no template match, or no
        ``CategoryLookup`` configured. ``None`` means "both routes search
        the full catalog"; an empty set would incorrectly mean "nothing
        matches anything."
        """
        if message is None or self.category_lookup is None:
            return None
        category = extract_coarse_category(message)
        if category is None:
            return None
        return self.category_lookup.matching_ids(category)

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
