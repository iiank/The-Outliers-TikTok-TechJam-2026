"""Build the compact catalog used by the reranker.

Parses the full catalog.jsonl into a smaller JSONL representation containing
only the fields needed by the reranker. Price and average rating are retained
for potential future use.

This script can be run from any working directory. Input and output paths are
resolved relative to this file and the repository root.
"""

from __future__ import annotations
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_PATH = REPO_ROOT / "data" / "catalog.jsonl"
OUTPUT_PATH = ( REPO_ROOT / "submission" / "src" / "reranker" / "reranker_catalog.jsonl" )

TEMPLATE = (
    "Title: {title} | "
    "Brand: {store} | "
    "Category: {leaf_categories} | "
    "Features: {top_features}"
)

def truncate_categories(categories: list[str] | None) -> str:
    if not categories:
        return ""

    # Filter generic top-level departments if present.
    filtered = [
        category
        for category in categories
        if category.lower() not in {
            "clothing, shoes & jewelry",
            "all departments",
        }
    ]

    # Retain the last two leaf categories.
    tail = filtered[-2:] if len(filtered) >= 2 else filtered
    return " > ".join(tail)

def truncate_features(
    features: list[str] | None,
    max_bullets: int = 4,
) -> str:
    if not features:
        return ""

    extracted = []

    for bullet in features:
        # Drop marketing commentary following a colon.
        clean_bullet = bullet.split(":", 1)[0].strip()

        # Keep short technical specifications.
        if len(clean_bullet.split()) <= 8:
            extracted.append(clean_bullet)

        if len(extracted) >= max_bullets:
            break

    # Fall back to a truncated version of the first bullet.
    if not extracted:
        extracted.append(" ".join(features[0].split()[:10]))

    return "; ".join(extracted)

def build_reranker_catalog(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Create the compact reranker catalog from the full product catalog."""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Catalog not found: {input_path}\n"
            "Expected catalog.jsonl to be located in the repository root."
        )

    # Ensure the destination directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    product_count = 0

    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue

            product = json.loads(line)

            title = product.get("title") or ""
            store = product.get("store") or ""
            price = product.get("price")
            avg_rating = product.get("average_rating")
            raw_categories = product.get("categories") or []

            leaf_categories = truncate_categories(raw_categories)
            top_features = truncate_features(product.get("features"))

            output = {
                "parent_asin": product.get("parent_asin"),
                "title": title,
                "document": TEMPLATE.format(
                    title=title,
                    store=store,
                    leaf_categories=leaf_categories,
                    top_features=top_features,
                ),
                "price": price,
                "category": raw_categories,
                "average_rating": avg_rating,
            }

            outfile.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )
            product_count += 1

    print(f"Built reranker catalog with {product_count:,} products.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    build_reranker_catalog()
