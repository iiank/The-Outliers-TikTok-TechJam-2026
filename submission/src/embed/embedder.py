from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

QUERY_PREFIXES = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}
DOC_PREFIXES = {
    "intfloat/e5-base-v2": "passage: ",
}

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")
        self.doc_prefix = DOC_PREFIXES.get(model_name, "")

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_products(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        payload = [self.doc_prefix + t for t in texts] if self.doc_prefix else texts
        vectors = self.model.encode(
            payload,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype("float32").reshape(1, -1)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.model.encode(
            [self.query_prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")