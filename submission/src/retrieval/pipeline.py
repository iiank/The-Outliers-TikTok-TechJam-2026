"""Wires the BM25 and dense routes together through weighted RRF.

``Retriever.retrieve(query_terms, mode, failed_asins=None)`` is the single
entry point. Per the team's Aug 2026 build brief (and the retrieval +
search integration fixes that followed it):

* Task 1 -- fusion is weighted differently for ``"buying"`` vs
  ``"browsing"`` (see ``retrieval.rrf.weights_for_mode``).
* Task 2 -- one fused computation produces two slices: a top-500 pool for
  the entropy/ask-attribute policy, and a top-100 ``(catalog_index,
  rrf_score)`` pool for the reranker. Both are prefixes of the same fused
  list, not two separate retrieval passes.
* Hard filtering (category + budget) is owned entirely by
  ``search.py``'s ``get_failed_hard_filter_asins()``, which reads
  ``state["session_profile"]`` and returns ``parent_asin``s to exclude.
  This module's only job is to turn that exclusion list into the same
  kind of ``candidate_ids`` set both routes already know how to search
  within -- "catalog minus excluded" instead of a positive match set.
  (The turn-1-message category pre-filter that used to live here, in
  ``retrieval.category_filter``, has been removed as redundant with
  ``search.py``'s session-state-based filter.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from .bm25 import BM25Index
from .catalog_ids import CatalogIndex
from .rrf import ROUTE_ORDER, reciprocal_rank_fusion, weights_for_mode

__all__ = ["Retriever", "RetrievalResult", "DenseRoute", "DEFAULT_ENTROPY_POOL_SIZE", "DEFAULT_RERANKER_POOL_SIZE"]

#: Named config, not magic numbers -- per the build brief, 500/100 are
#: adjustable and only worth revisiting if there's spare time.
DEFAULT_ENTROPY_POOL_SIZE = 500
DEFAULT_RERANKER_POOL_SIZE = 100

#: How much deeper to query the dense route than the target pool size
#: whenever a hard filter has excluded any candidates, to compensate for
#: the overfetch+filter-in-Python approach (see module docstring / build
#: discussion): VectorStore.search() has no candidate-ID parameter, so
#: restricting it means asking for more than needed, then discarding
#: anything outside the allowed set. A heavily-filtered turn can still
#: come back short of the target pool size -- that's a known, accepted
#: limit of this approach, not a bug.
_DENSE_OVERFETCH_MULTIPLIER = 5


class DenseRoute(Protocol):
    """Structural contract ``embed.store.VectorStore`` satisfies.

    Deliberately has no ``candidate_ids`` parameter -- unlike
    :meth:`BM25Index.search`, the dense route is never asked to filter
    itself; :meth:`Retriever._search_dense` overfetches and filters in
    Python instead, so this module never needs to touch ``embed/store.py``.
    Works the same way regardless of whether the candidate set came from
    a positive category match or (now) a hard-filter exclusion list.
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
    """Fuses BM25 + dense via mode-weighted RRF, behind search.py's hard-filter exclusions."""

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
        """Run the full retrieval + fusion pipeline for one turn.

        ``query_terms`` is what both routes search with -- building it
        from the conversation's accumulated state (vs. just this turn's
        raw text) is the caller's responsibility, not this module's.

        ``failed_asins``, if given, is a list of ``parent_asin``s to
        exclude before either route searches -- typically
        ``search.get_failed_hard_filter_asins()``'s output (category +
        budget hard constraints). Pass ``None`` or ``[]`` for no
        exclusions: both routes then search the full catalog.

        ``mode`` must be ``"buying"`` or ``"browsing"`` -- see
        ``retrieval.rrf.weights_for_mode``.
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
        """``failed_asins`` -> allowed-ID set (catalog minus excluded), or ``None`` (unscoped).

        ``None`` -- not an empty set -- whenever there's nothing to
        exclude, so both routes take the cheap unscoped path instead of
        needlessly restricting to "everything."
        """
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
