"""Post-fusion target-price scoring adjustment.

Confirmed design (30 Aug 2026, Joanne thread -- see
target-price-scoring-spec.md): ``target_price`` is a SOFT scoring signal
only -- it must never exclude a candidate. Hard price bounds
(``min_price``/``max_price``, from ``state.dialogue_state.budget_bounds()``)
are ``search.get_failed_hard_filter_asins()``'s job; this module never
touches exclusion, only re-scoring.

Not an RRF/fusion route -- RRF (``retrieval.rrf``) operates on rank
position, not magnitude, so folding price in there would discard the
continuous, distance-sensitive signal this is meant to provide, and would
double- or triple-apply depending on how many routes are active. This is
instead a one-time post-fusion adjustment: multiply whatever ranking
score a candidate already has by a price-closeness factor, then re-sort.

Lives in ``retrieval/`` as a sibling to ``rrf.py`` rather than a new
``policy/`` package (the spec offered both as options) -- there's no
``policy/`` package anywhere else in this codebase, and this stays
decoupled from ``rrf.py``'s internals either way.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["DEFAULT_DECAY_RATE", "price_multiplier", "apply_target_price_scoring"]

#: Placeholder -- NOT tuned. Per the spec, tune the same way the BM25/
#: dense route weights are: grid search against TechnicalScore on the
#: 200 public dev sessions. Controls how steeply the multiplier falls
#: off with distance from target_price -- smaller is more forgiving,
#: larger punishes distance harder.
DEFAULT_DECAY_RATE = 1.0


def price_multiplier(
    price: Optional[float],
    target_price: Optional[float],
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> float:
    """Multiplicative exponential-decay factor for how close ``price`` is to ``target_price``.

    ``exp(-|price - target_price| / (target_price * decay_rate))`` -- 1.0
    exactly at target, decaying symmetrically as price moves away in
    either direction (over- and under-target penalized equally --
    deliberate, see the spec's Open Decision 2, not an oversight: revisit
    only if a later check of the public dev set shows true target prices
    are systematically below the disclosed target_price).

    Neutral (``1.0``) whenever ``price`` is missing -- matches
    Elasticsearch's own documented default for decay functions on a
    missing field. Never defaults missing price to ``0`` or a large
    sentinel: ``0`` silently computes a real (wrong) ratio instead of
    signaling "unknown" (e.g. ``price=0`` vs ``target_price=60`` computes
    a full ratio of 1.0, multiplier ~0.37, not neutral), and a large
    sentinel would collapse the multiplier toward 0 -- a hard exclusion
    smuggled into a formula that must never exclude anything.

    ``target_price=None`` also returns ``1.0``, so a caller that skips
    :func:`apply_target_price_scoring`'s own no-op check still gets a
    safe neutral value rather than a ``ZeroDivisionError``.
    """
    if price is None or target_price is None:
        return 1.0
    return math.exp(-abs(price - target_price) / (target_price * decay_rate))


def apply_target_price_scoring(
    candidates: Sequence[Dict[str, Any]],
    target_price: Optional[float],
    decay_rate: float = DEFAULT_DECAY_RATE,
    score_key: str = "score",
    price_key: str = "price",
) -> List[Dict[str, Any]]:
    """Re-score and re-sort ``candidates`` by closeness to ``target_price``.

    Each candidate is a dict already carrying a base ranking score under
    ``score_key`` and a price under ``price_key`` -- exactly the shape
    ``Reranker.rank_from_state()`` already returns (its catalog rows
    already include ``price``, so no separate price lookup is needed
    here). Returns *new* dicts (the input is never mutated) with
    ``score_key`` replaced by ``base_score * price_multiplier``,
    re-sorted best-first -- so a candidate ranked outside the caller's
    final cutoff can still move up (or a poorly-priced top candidate can
    move down) once price is factored in. Call this on the wider
    candidate pool *before* slicing to the turn's top-10, not after --
    slicing first would make this adjustment unable to change anything.

    ``target_price=None`` -- no target stated this turn -- is a full
    no-op: returns ``candidates`` unchanged (same dicts, same order), not
    run through the formula with a fabricated target.
    """
    if target_price is None:
        return list(candidates)

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        multiplier = price_multiplier(candidate.get(price_key), target_price, decay_rate)
        adjusted = dict(candidate)
        adjusted[score_key] = candidate.get(score_key, 0.0) * multiplier
        scored.append(adjusted)

    scored.sort(key=lambda c: c[score_key], reverse=True)
    return scored
