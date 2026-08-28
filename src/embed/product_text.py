"""Catalog row -> the string that gets embedded.

This module is the single highest-leverage knob in retrieval. The evaluator
builds its query from the target product's own `categories` and `features`
(see local_evaluator.intent_card / initial_message), so the category path and
raw feature bullets carry most of the matching signal. Variants are kept
side by side so they can be compared against turn-1 recall.
"""

from __future__ import annotations

VARIANTS = ("title_cat", "default", "rich")


def _flatten_details(details: object) -> str:
    if not isinstance(details, dict):
        return ""
    return " ".join(f"{k} {v}" for k, v in details.items() if v not in (None, "", []))


def _join(values: object, limit: int | None = None) -> str:
    if not isinstance(values, list):
        return str(values) if values not in (None, "") else ""
    items = [str(v) for v in values if v not in (None, "")]
    if limit is not None:
        items = items[:limit]
    return " ".join(items)


def product_to_text(row: dict, variant: str = "default") -> str:
    """Build the embedding string for one catalog row.

    variant:
        title_cat - title + category path only. Strong, cheap baseline.
        default   - adds feature bullets and a truncated description.
        rich      - adds details dict and store. Most overlap with the
                    evaluator's generated query, but most dilution risk.
    """
    title = str(row.get("title") or "")
    categories = " > ".join(str(c) for c in (row.get("categories") or []))

    if variant == "title_cat":
        parts = [title, categories]
    elif variant == "rich":
        parts = [
            title,
            categories,
            _join(row.get("features")),
            _join(row.get("description"))[:600],
            _flatten_details(row.get("details")),
            str(row.get("store") or ""),
        ]
    else:
        parts = [
            title,
            categories,
            _join(row.get("features"), limit=6),
            _join(row.get("description"))[:400],
        ]

    return " ".join(p.strip() for p in parts if p and p.strip())
