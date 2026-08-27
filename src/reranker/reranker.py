import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class ProductReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 192,
        batch_size: int = 64,
        use_fp16: bool = True,
    ):
        """
        Persistent reranker wrapper optimized for low-latency batch scoring.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading CrossEncoder ({model_name}) onto {self.device}...")
        self.model = CrossEncoder(
            model_name,
            max_length=self.max_length,
            device=self.device
        )

        # Enable half precision if running on CUDA for ~2x compute speedup
        if self.device == "cuda" and use_fp16:
            self.model.model.half()

    @torch.inference_mode()
    def rank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate documents against a query string using batch inference.
        """
        if not candidates:
            return []

        # Prepare (query, document) pairs
        pairs = [(query, doc.get("document", "")) for doc in candidates]

        # Fast batched prediction
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        for doc, score in zip(candidates, scores):
            doc["score"] = float(score)

        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    def rank_indices(
        self,
        query: str,
        target_indices: Sequence[int],
        catalog: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Tuple[int, str]]:
        """
        Directly fetches candidate records from an in-memory catalog and returns (index, parent_asin).
        """
        # Fast memory indexing
        candidates = []
        for idx in target_indices:
            if 0 <= idx < len(catalog):
                item = catalog[idx].copy()
                item["index"] = idx
                candidates.append(item)

        ranked = self.rank(query=query, candidates=candidates, top_k=top_k)
        return [(doc["index"], doc.get("parent_asin", "")) for doc in ranked]


def load_catalog(catalog_path: str = "reranker_catalog.jsonl") -> List[Dict[str, Any]]:
    """
    Loads the complete catalog into memory once during application startup.
    """
    catalog = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                catalog.append(json.loads(line_str))
    return catalog