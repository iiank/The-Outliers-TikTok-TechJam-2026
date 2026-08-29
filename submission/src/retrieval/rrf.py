"""Reciprocal Rank Fusion: combine ranked lists from independent retrieval routes.

Takes any number of ``list[(doc_id, score)]`` rankings -- e.g. one from
the dense route (``embed.store.VectorStore.search``) and one from the
keyword route (``retrieval.bm25.BM25Index.search``) -- and returns one
fused ranking. RRF deliberately ignores each route's raw score and uses
only rank *position*, which is what makes it safe to combine routes whose
scores aren't on comparable scales (cosine similarity vs. a BM25 value).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = ["reciprocal_rank_fusion"]

#: Standard RRF smoothing constant. Not the same knob as a pipeline's
#: ``top_k`` (how many results you want back) -- this controls how much
#: a route's rank-1 result is favored over its rank-50 result. Higher
#: values flatten the curve; 60 is the conventional default from the
#: original RRF paper.
_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    rrf_k: int = _DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ``(doc_id, score)`` rankings into one, best first.

    Each input ranking should already be sorted best-first (as
    ``VectorStore.search`` and ``BM25Index.search`` return); this function
    reads rank from list position (1-indexed), not from the score values,
    so a route with no signal this turn can simply pass ``[]`` and is
    skipped rather than crashing the fusion.

    A doc's fused score is ``sum(1 / (rrf_k + rank))`` over every ranking
    it appears in -- absent from a ranking contributes 0, not a penalty.
    Ties in fused score are broken by first-appearance order across the
    input rankings, so the result is deterministic.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0

    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
