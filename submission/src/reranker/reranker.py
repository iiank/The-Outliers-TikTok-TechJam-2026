import logging
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from sentence_transformers import CrossEncoder

from state.dialogue_state import budget_bounds

logger = logging.getLogger(__name__)

RERANKER_CATALOG_LEN = 50000

#: Placeholder -- NOT tuned. Per target-price-scoring-spec.md, tune the
#: same way the BM25/dense route weights are: grid search against
#: TechnicalScore on the 200 public dev sessions. Controls how steeply
#: the multiplier falls off with distance from target_price.
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
    deliberate, per target-price-scoring-spec.md Open Decision 2).

    Neutral (``1.0``) whenever ``price`` or ``target_price`` is missing --
    never defaults missing price to ``0`` or a large sentinel, since
    either would smuggle a hard exclusion into a formula that must never
    exclude a candidate (target_price is a soft signal only).
    """
    if price is None or target_price is None:
        return 1.0
    return math.exp(-abs(price - target_price) / (target_price * decay_rate))

def build_reranker_query(state: Dict[str, Any]) -> str:
    """
    Parses DialogueState into a compact, high-density query string for MS MARCO.
    Note Reranker takes the whole state as-is defined in the README_dialogue_state.md
    """
    session_profile = state.get("session_profile", {})
    user_profile = state.get("user_profile", {})

    query_parts = []

    # category
    categories = session_profile.get("category", [])
    if categories:
        query_parts.append(" ".join(categories))

    # descriptors in session_profile
    descriptors = []
    for slot in ["color", "material", "style", "brand"]:
        values = session_profile.get(slot, [])
        if values:
            descriptors.extend(values)
    if descriptors:
        query_parts.append(" ".join(descriptors))

    # features & use case in session_profile
    specs = []
    for slot in ["feature", "use_case", "other"]:
        values = session_profile.get(slot, [])
        if values:
            specs.extend(values)
    if specs:
        query_parts.append("with " + ", ".join(specs))

    # user preferences (Personalization)
    pref_tags = user_profile.get("preference_tags", [])
    if pref_tags:
        query_parts.append(f"preferences: {', '.join(pref_tags)}")

    query_str = " ".join(query_parts).strip()
    return query_str if query_str else "general merchandise"

def load_reranker_catalog(catalog_path: str = "reranker_catalog.jsonl") -> List[Dict[str, Any]]:
    """
    Loads the complete reranker_catalog into memory once during agent startup.
    """
    catalog = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                catalog.append(json.loads(line_str))
    return catalog

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 192,
        batch_size: int = 64,
        use_fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading CrossEncoder ({model_name}) onto {self.device}...")
        self.model = CrossEncoder(
            model_name,
            max_length=self.max_length,
            device=self.device,
        )

        if self.device == "cuda" and use_fp16:
            self.model.model.half()

    @torch.inference_mode()
    def rank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        target_price: Optional[float] = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate documents against a query string.
        Candidates is an array of preformatted documents.

        ``target_price``, if given, applies a soft price-closeness
        adjustment (target-price-scoring-spec.md) to each candidate's
        cross-encoder score *before* the sort-and-cut below -- so a
        well-priced candidate can move into ``top_k`` and a poorly-priced
        one can move out, not just get reordered within an already-fixed
        top_k. ``None`` (no target stated this turn) is a full no-op:
        every ``price_multiplier`` call returns 1.0 and scores are
        unaffected -- this never excludes a candidate, only reorders.
        """
        if not candidates:
            return []

        pairs = [(query, doc.get("document", "")) for doc in candidates]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for doc, score in zip(candidates, scores):
            doc["score"] = float(score)

        if target_price is not None:
            for doc in candidates:
                doc["score"] *= price_multiplier(doc.get("price"), target_price, decay_rate)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def rank_from_state(
        self,
        state: Dict[str, Any],
        candidate_indices: Sequence[int],
        catalog: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        function for agent: parses state, builds pairs from in-memory catalog, and ranks.

        Pulls ``target_price`` out of ``state`` itself (via
        ``budget_bounds()``) and factors it into the final score inside
        ``rank()`` -- callers just pass ``state``, they don't need to
        extract or pass price info separately.
        """
        query = build_reranker_query(state)
        target_price = budget_bounds(state.get("session_profile", {})).get("target_price")

        candidates = []
        for idx in candidate_indices:
            if 0 <= idx < RERANKER_CATALOG_LEN:
                item = catalog[idx].copy()
                item["index"] = idx
                candidates.append(item)

        return self.rank(query=query, candidates=candidates, top_k=top_k, target_price=target_price)

