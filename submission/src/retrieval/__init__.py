"""Retrieval: keyword (BM25) + dense routes, fused via mode-weighted RRF, behind search.py's hard-filter exclusions.

    from retrieval import BM25Index, CatalogIndex, Retriever, RetrievalResult
"""

from __future__ import annotations

from .bm25 import BM25Index
from .catalog_ids import CatalogIndex
from .pipeline import RetrievalResult, Retriever
from .rrf import (
    BROWSING_WEIGHTS,
    BUYING_WEIGHTS,
    ROUTE_ORDER,
    reciprocal_rank_fusion,
    weights_for_mode,
)

__all__ = [
    "BM25Index",
    "BROWSING_WEIGHTS",
    "BUYING_WEIGHTS",
    "CatalogIndex",
    "ROUTE_ORDER",
    "RetrievalResult",
    "Retriever",
    "reciprocal_rank_fusion",
    "weights_for_mode",
]
