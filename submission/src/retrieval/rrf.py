"""Reciprocal Rank Fusion: combine ranked lists from the dense route
(``embed.store.VectorStore.search``) and one from the keyword route
(``retrieval.bm25.BM25Index.search``)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "reciprocal_rank_fusion",
    "weights_for_mode",
    "ROUTE_ORDER",
    "BUYING_WEIGHTS",
    "BROWSING_WEIGHTS",
]

# RRF smoothing constant
_DEFAULT_RRF_K = 60

# Fixed order routes are fused in
ROUTE_ORDER = ("bm25", "dense")

# NOTE: Placeholder weights NEED TO TUNE with gridsearch
# Buying leans on BM25, Browsing leans on dense
BUYING_WEIGHTS: Dict[str, float] = {"bm25": 3.0, "dense": 1.0}
BROWSING_WEIGHTS: Dict[str, float] = {"bm25": 1.0, "dense": 2.0}

_MODE_WEIGHTS = {"buying": BUYING_WEIGHTS, "browsing": BROWSING_WEIGHTS}


def weights_for_mode(mode: str, route_order: Sequence[str] = ROUTE_ORDER) -> List[float]:
    """Resolve ``mode`` ("buying"/"browsing") to an ordered weight list.

    ``route_order`` names each route being fused in the same order
    rankings are passed to :func:`reciprocal_rank_fusion`
    """
    table = _MODE_WEIGHTS[mode]
    return [table[route] for route in route_order]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = _DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ``(doc_id, score)`` rankings into one, best first.
    Formula: ``sum(weight / (rrf_k + rank))``

    ``weights``, if given, must have one entry per ranking (same order),
    e.g. from :func:`weights_for_mode`.
    Omit (or pass ``None``) for unweighted behavior.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries but rankings has {len(rankings)}; "
            "each ranking needs exactly one weight, in the same order."
        )

    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0

    for ranking_index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[ranking_index]
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
