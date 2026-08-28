"""Retrieval: keyword (BM25) + dense routes, fused via RRF.

    from retrieval import BM25Index, CatalogIndex, Retriever, reciprocal_rank_fusion
"""

from __future__ import annotations

from .bm25 import BM25Index
from .catalog_ids import CatalogIndex
from .pipeline import Retriever
from .rrf import reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "CatalogIndex",
    "Retriever",
    "reciprocal_rank_fusion",
]
