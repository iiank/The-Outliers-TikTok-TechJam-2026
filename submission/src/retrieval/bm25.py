"""Keyword (BM25) retrieval, extracted from ``starter.agent.Agent``.

The starter baseline built its SQLite FTS5 index and BM25 query inline
inside the ``Agent`` class, which meant nothing else could reuse it. This
module is that same logic, factored into a standalone route with the same
``search(query, top_k) -> list[(parent_asin, score)]`` shape the dense
route (``embed.store.VectorStore``) exposes -- so both can feed
``retrieval.rrf.reciprocal_rank_fusion`` interchangeably.

Score convention: SQLite's ``bm25()`` returns *lower-is-better* values.
Every other route in this pipeline (cosine similarity) is
higher-is-better, and RRF itself only cares about rank order, not raw
score magnitude -- but to avoid a footgun if raw scores are ever compared
or logged, :meth:`BM25Index.search` negates the raw value before
returning it, so "higher score = more relevant" holds everywhere.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, Tuple

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

#: Column weights for ``bm25(products, ...)``, in table column order
#: (parent_asin is UNINDEXED so it takes no weight). Unchanged from the
#: starter agent: title and categories carry the most signal.
_BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

__all__ = ["BM25Index", "TOKEN_RE", "STOPWORDS"]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class BM25Index:
    """In-memory SQLite FTS5 keyword index over the product catalog.

    Same role as ``embed.store.VectorStore`` on the dense side: build once
    (``__init__``), then call :meth:`search` per turn.
    """

    def __init__(self, catalog_path: "str | Path" = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: List[Tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def search(
        self,
        query: str,
        top_k: int,
        candidate_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to ``top_k`` ``(parent_asin, score)`` pairs, best first.

        Returns ``[]`` (never raises) when the query has no usable terms
        after stopword removal -- the same "route has nothing to
        contribute this turn" contract the dense route follows. Also
        returns ``[]`` immediately for an empty (but not ``None``)
        ``candidate_ids`` -- that means some upstream filter (e.g. the
        coarse-category pre-filter) matched nothing, not "search
        everything."

        ``candidate_ids``, if given, restricts the search to just those
        ``parent_asin``s (Task 3's coarse-category hard pre-filter). This
        goes through a temp table + JOIN rather than an inline
        ``parent_asin IN (?, ?, ...)`` list, since a category's candidate
        set can run into the thousands and SQLite caps how many bound
        parameters a single statement may have.
        """
        unique_terms = list(dict.fromkeys(_terms(query)))[:40]
        if not unique_terms or top_k <= 0:
            return []
        if candidate_ids is not None and not candidate_ids:
            return []

        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        weighted_bm25 = "bm25(products, {})".format(", ".join(str(w) for w in _BM25_WEIGHTS))

        if candidate_ids is None:
            rows = self.connection.execute(
                f"SELECT parent_asin, {weighted_bm25} FROM products "
                f"WHERE products MATCH ? ORDER BY {weighted_bm25} LIMIT ?",
                (expression, top_k),
            ).fetchall()
        else:
            cursor = self.connection.cursor()
            cursor.execute("CREATE TEMP TABLE IF NOT EXISTS candidate_filter (parent_asin TEXT PRIMARY KEY)")
            cursor.execute("DELETE FROM candidate_filter")
            cursor.executemany(
                "INSERT INTO candidate_filter VALUES (?)",
                [(asin,) for asin in candidate_ids],
            )
            rows = cursor.execute(
                f"SELECT products.parent_asin, {weighted_bm25} FROM products "
                "JOIN candidate_filter ON candidate_filter.parent_asin = products.parent_asin "
                f"WHERE products MATCH ? ORDER BY {weighted_bm25} LIMIT ?",
                (expression, top_k),
            ).fetchall()

        return [(str(parent_asin), -float(raw_score)) for parent_asin, raw_score in rows]
