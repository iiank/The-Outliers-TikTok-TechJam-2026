"""Coarse-category hard pre-filter (Task 3).

Confirmed design (per the team's Aug 2026 build brief): coarse category is
a hard pre-filter shared by BM25 and dense, not a third RRF-fused route.
When a category can be safely extracted from the customer's message, it
narrows the candidate pool *before* either route searches; both routes
then rank purely within that already-scoped pool.

Extraction is template-based and deliberately conservative: it only
recognizes the exact opening-message shapes the simulator itself produces
(see ``evaluator.local_evaluator.initial_message``). A message that
doesn't match one of those templates -- real free text, paraphrasing, an
Intent Override opener -- yields ``None`` rather than a guess. This is a
safe degrade, not a smarter extractor: inferring a category from context
(e.g. "party" -> dresses) is semantic inference, out of scope here (same
class of problem as the KIV'd intent classifier).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Set

__all__ = ["extract_coarse_category", "CategoryLookup"]

# Mirrors the two simulator templates from initial_message() that contain
# an unambiguous category phrase (Buying and Browsing). Intent Override's
# opener ("I'm looking for {category}. {old_value}") is NOT matched here:
# old_value is arbitrary text, so there's no way to tell where the
# category phrase ends and old_value begins without guessing -- exactly
# the kind of ambiguity this extractor is built to refuse rather than
# resolve by guessing. Falling through to None for that shape is correct,
# not a gap.
_BROWSING_RE = re.compile(
    r"^\s*i'?m looking for\s+(?P<category>.+?),\s*but i'?m still exploring\.?\s*$",
    re.IGNORECASE,
)
_BUYING_RE = re.compile(
    r"^\s*i'?m looking for\s+(?P<category>.+?)\.\s*a key requirement is:",
    re.IGNORECASE,
)


def extract_coarse_category(message: str) -> Optional[str]:
    """Return the category phrase if ``message`` matches a known template, else ``None``.

    Example that must return ``None`` (from the build brief): "im looking
    for something to buy since im going to a party. im just browsing.
    help me find something" -- it has a period in the wrong place and
    neither recognized suffix, so neither pattern matches.
    """
    for pattern in (_BROWSING_RE, _BUYING_RE):
        match = pattern.match(message)
        if match:
            category = match.group("category").strip()
            return category or None
    return None


def _tokens(text: str) -> Set[str]:
    return {token.lower() for token in re.findall(r"[a-z0-9]+", text, re.IGNORECASE) if len(token) > 1}


class CategoryLookup:
    """``parent_asin`` -> catalog category tokens, for coarse matching.

    A product matches a category phrase when every word in the phrase
    appears somewhere in that product's ``categories`` breadcrumb -- a
    coarse, word-set match (like ``retrieval.bm25``'s tokenization), not
    an exact string match, since breadcrumbs are formatted inconsistently
    across the catalog.
    """

    def __init__(self, category_tokens: Dict[str, Set[str]]) -> None:
        self._category_tokens = category_tokens

    @classmethod
    def load(cls, catalog_path: "str | Path" = "data/catalog.jsonl") -> "CategoryLookup":
        category_tokens: Dict[str, Set[str]] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                categories = product.get("categories") or []
                category_tokens[parent_asin] = _tokens(" ".join(str(c) for c in categories))
        return cls(category_tokens)

    def matching_ids(self, category_phrase: str) -> Set[str]:
        """``parent_asin``s whose categories contain every word in ``category_phrase``.

        Returns an empty set (not "everything") if the phrase has no
        usable words -- an empty pre-filter pool is the correct signal to
        the caller that this phrase didn't narrow anything real, rather
        than silently falling back to unscoped.
        """
        phrase_tokens = _tokens(category_phrase)
        if not phrase_tokens:
            return set()
        return {
            asin
            for asin, tokens in self._category_tokens.items()
            if phrase_tokens <= tokens
        }
