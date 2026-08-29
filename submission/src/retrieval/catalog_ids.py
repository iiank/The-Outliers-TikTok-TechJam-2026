"""``parent_asin`` <-> row-index lookup for ``data/catalog.jsonl``.

Neither retrieval route exposes a product's position in the catalog file
by itself: the BM25 route only returns ``parent_asin`` (the FTS5 table
doesn't track row order), and the dense route's Chroma store returns
whatever id it was upserted with. Building this index is a single
sequential read done once at startup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["CatalogIndex"]


class CatalogIndex:
    """Ordered ``parent_asin`` list from the catalog, plus the reverse lookup."""

    def __init__(self, ids: List[str]) -> None:
        self.ids = ids
        self._index_of: Dict[str, int] = {asin: i for i, asin in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, parent_asin: str) -> Optional[int]:
        """Row index of ``parent_asin`` in the catalog, or ``None`` if unknown."""
        return self._index_of.get(parent_asin)

    @classmethod
    def load(cls, catalog_path: "str | Path" = "data/catalog.jsonl") -> "CatalogIndex":
        ids: List[str] = []
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                ids.append(str(json.loads(line)["parent_asin"]))
        return cls(ids)
