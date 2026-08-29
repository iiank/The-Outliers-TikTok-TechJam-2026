"""Wires the dense and BM25 routes together through RRF into one retrieval call.

``Retriever.retrieve(query, top_k=100)`` (or ``top_k=500``) is the single
entry point downstream code (routing, reranking) should call. It owns the
one non-obvious tuning knob in this pipeline: each route is queried
*deeper* than ``top_k`` before fusing, because RRF only sees rank
position -- a document ranked 150th on the dense route but 3rd on BM25
should still be able to surface into the fused top 100, which can't
happen if both routes are only ever asked for exactly 100 candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence, Tuple

from .catalog_ids import CatalogIndex
from .rrf import reciprocal_rank_fusion

__all__ = ["Retriever", "RetrievalRoute"]


class RetrievalRoute(Protocol):
    """Structural contract both ``VectorStore`` and ``BM25Index`` satisfy."""

    def search(self, query: str, top_k: int) -> Sequence[Tuple[str, float]]: ...


@dataclass
class Retriever:
    """Fuses one or more retrieval routes via RRF, with catalog-index lookup.

    ``routes`` takes any number of ``RetrievalRoute``-shaped objects, not
    just exactly two -- add a third route later (e.g. a category-filtered
    pass) without changing this class.
    """

    routes: Sequence[RetrievalRoute]
    catalog_index: CatalogIndex
    pool_multiplier: int = 4
    rrf_k: int = 60

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` fused results, best first.

        Each result is ``{"parent_asin": str, "catalog_index": int | None,
        "score": float}``. ``catalog_index`` is ``None`` only if a route
        returned an id that isn't in the loaded catalog, which shouldn't
        happen against the real catalog but is tolerated rather than
        raised, consistent with every other route in this pipeline.
        """
        if top_k <= 0:
            return []

        pool_size = top_k * self.pool_multiplier
        rankings = [route.search(query, pool_size) for route in self.routes]
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)

        results: List[Dict[str, Any]] = []
        for parent_asin, score in fused[:top_k]:
            results.append(
                {
                    "parent_asin": parent_asin,
                    "catalog_index": self.catalog_index.index_of(parent_asin),
                    "score": score,
                }
            )
        return results
