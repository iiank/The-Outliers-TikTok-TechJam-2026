"""Reciprocal Rank Fusion: combine ranked lists from independent retrieval routes.

Takes any number of ``list[(doc_id, score)]`` rankings -- e.g. one from
the dense route (``embed.store.VectorStore.search``) and one from the
keyword route (``retrieval.bm25.BM25Index.search``) -- and returns one
fused ranking. RRF deliberately ignores each route's raw score and uses
only rank *position*, which is what makes it safe to combine routes whose
scores aren't on comparable scales (cosine similarity vs. a BM25 value).
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

#: Standard RRF smoothing constant. Not the same knob as a pipeline's
#: ``top_k`` (how many results you want back) -- this controls how much
#: a route's rank-1 result is favored over its rank-50 result. Higher
#: values flatten the curve; 60 is the conventional default from the
#: original RRF paper (Cormack, Clarke & Buettcher, SIGIR 2009), which
#: found the exact value "not critical" -- not worth spending tuning
#: time on, unlike the per-route weights below.
_DEFAULT_RRF_K = 60

#: Fixed order routes are fused in, and the key each mode's weight table
#: uses. Category no longer has a weight here -- per the Aug 2026 build
#: brief, its influence is entirely the hard pre-filter in
#: ``retrieval.category_filter``, not a third RRF vote.
ROUTE_ORDER = ("bm25", "dense")

#: Placeholder weights -- NOT tuned yet. Per the build brief, real values
#: should come from a grid search over the 200 public dev sessions,
#: optimizing TechnicalScore per scenario (local_evaluator.py already
#: reports that split; a full run is ~40s, so a coarse grid over
#: {1, 2, 3, 5} normalized is cheap). These are directionally reasonable
#: starting points so the mechanism is usable before that tuning happens:
#: Buying leans on BM25 (and the upstream hard-constraint filter) with
#: dense along only to catch nuance; Browsing leans relatively more on
#: dense since the opening query is vague and exact term overlap is a
#: weaker signal there.
BUYING_WEIGHTS: Dict[str, float] = {"bm25": 3.0, "dense": 1.0}
BROWSING_WEIGHTS: Dict[str, float] = {"bm25": 1.0, "dense": 2.0}

_MODE_WEIGHTS = {"buying": BUYING_WEIGHTS, "browsing": BROWSING_WEIGHTS}


def weights_for_mode(mode: str, route_order: Sequence[str] = ROUTE_ORDER) -> List[float]:
    """Resolve ``mode`` ("buying"/"browsing") to an ordered weight list.

    ``route_order`` must name each route being fused, in the same order
    rankings are passed to :func:`reciprocal_rank_fusion` -- callers
    should use the module-level :data:`ROUTE_ORDER` unless they have a
    specific reason not to. Raises ``KeyError`` for an unrecognized mode
    or route name rather than silently defaulting a route's weight to 1,
    since a silently-wrong weight would be a scoring bug, not a missing
    feature.
    """
    table = _MODE_WEIGHTS[mode]
    return [table[route] for route in route_order]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Tuple[str, float]]],
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = _DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ``(doc_id, score)`` rankings into one, best first.

    Each input ranking should already be sorted best-first (as
    ``VectorStore.search`` and ``BM25Index.search`` return); this function
    reads rank from list position (1-indexed), not from the score values,
    so a route with no signal this turn can simply pass ``[]`` and is
    skipped rather than crashing the fusion.

    ``weights``, if given, must have one entry per ranking (same order),
    e.g. from :func:`weights_for_mode`. Omit it (or pass ``None``) for the
    original unweighted behavior, where every route counts equally.

    A doc's fused score is ``sum(weight / (rrf_k + rank))`` over every
    ranking it appears in -- absent from a ranking contributes 0, not a
    penalty. Ties in fused score are broken by first-appearance order
    across the input rankings, so the result is deterministic.
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
