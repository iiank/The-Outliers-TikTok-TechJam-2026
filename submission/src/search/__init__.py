"""
Search package initialization.
Exposes the overarching search API, SearchPipeline, hard filters, and query builders.
"""

from src.search.search import (
    ALLOWED_ATTRIBUTES,
    SearchPipeline,
    get_failed_hard_filter_asins,
    search,
)

__all__ = [
    "search",
    "SearchPipeline",
    "get_failed_hard_filter_asins",
    "ALLOWED_ATTRIBUTES",
]