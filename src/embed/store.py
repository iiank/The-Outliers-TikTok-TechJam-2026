from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb

from .embedder import Embedder
from .product_text import product_to_text

DEFAULT_COLLECTION = "products"


def _number(value: object) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    if not cleaned or cleaned.count(".") > 1:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class VectorStore:
    def __init__(self, collection, embedder: Embedder, variant: str = "default") -> None:
        self.collection = collection
        self.embedder = embedder
        self.variant = variant

    def __len__(self) -> int:
        return int(self.collection.count())

    def search(self, query_text: str, k: int = 200, where: dict | None = None) -> list[tuple[str, float]]:
        vector = self.embedder.encode_query(query_text)[0].tolist()
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, max(len(self), 1)),
            where=where or None,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(asin, 1.0 - float(d)) for asin, d in zip(ids, distances)]

    def search_batch(self, query_texts: list[str], k: int = 200) -> list[list[tuple[str, float]]]:
        vectors = self.embedder.encode_queries(query_texts).tolist()
        result = self.collection.query(
            query_embeddings=vectors,
            n_results=min(k, max(len(self), 1)),
            include=["distances"],
        )
        return [
            [(asin, 1.0 - float(d)) for asin, d in zip(ids, distances)]
            for ids, distances in zip(result["ids"], result["distances"])
        ]


def _price(value: object) -> float:
    if value in (None, ""):
        return -1.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    if not cleaned or cleaned.count(".") > 1:
        return -1.0
    try:
        return float(cleaned)
    except ValueError:
        return -1.0


def _rows(catalog_path: str | Path, variant: str, limit: int | None):
    with Path(catalog_path).open(encoding="utf-8") as handle:
        seen = 0
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            yield (
                str(product["parent_asin"]),
                product_to_text(product, variant=variant),
                {
                    "title": product.get("title") or "",
                    "categories": " > ".join(str(c) for c in (product.get("categories") or [])),
                    "store": product.get("store") or "",
                    "price": _price(product.get("price")),
                    "average_rating": _number(product.get("average_rating")),
                    "rating_number": int(_number(product.get("rating_number"))),
                },
            )
            seen += 1
            if limit is not None and seen >= limit:
                break


def build_store(
    catalog_path: str | Path,
    persist_directory: str | Path = "artifacts/chroma",
    embedder: Embedder | None = None,
    variant: str = "default",
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 512,
    limit: int | None = None,
    recreate: bool = True,
) -> VectorStore:
    embedder = embedder or Embedder()
    client = chromadb.PersistentClient(path=str(persist_directory))

    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "model_name": embedder.model_name,
            "variant": variant,
            "dim": embedder.dim,
        },
    )

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    batch_meta: list[dict] = []

    def flush() -> None:
        if not batch_ids:
            return
        vectors = embedder.encode_products(batch_texts, show_progress=False)
        collection.upsert(ids=batch_ids, embeddings=vectors.tolist(), metadatas=batch_meta)
        batch_ids.clear()
        batch_texts.clear()
        batch_meta.clear()

    for asin, text, meta in _rows(catalog_path, variant, limit):
        batch_ids.append(asin)
        batch_texts.append(text)
        batch_meta.append(meta)
        if len(batch_ids) >= batch_size:
            flush()
    flush()

    return VectorStore(collection, embedder, variant)


def load_store(
    persist_directory: str | Path = "artifacts/chroma",
    embedder: Embedder | None = None,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStore:
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_collection(collection_name)
    meta = collection.metadata or {}
    embedder = embedder or Embedder(meta.get("model_name") or Embedder().model_name)
    if meta.get("model_name") and embedder.model_name != meta["model_name"]:
        raise ValueError(
            f"Collection '{collection_name}' was built with {meta['model_name']} but "
            f"{embedder.model_name} was supplied. Vectors from different models are not "
            "comparable; rebuild the store."
        )
    return VectorStore(collection, embedder, meta.get("variant", "default"))