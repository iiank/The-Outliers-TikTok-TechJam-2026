"""One-time offline job: embed the full catalog into the Chroma vector store.

Run once, and again whenever ``data/catalog.jsonl`` or the embedding
model/variant changes:

    python scripts/build_chroma_store.py

Writes a persistent Chroma collection to ``artifacts/chroma/`` (gitignored
-- every teammate builds their own local copy rather than committing a
binary index). ``embed.store.load_store()`` opens it at agent startup.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission" / "src"))

from embed import DEFAULT_MODEL, Embedder, VARIANTS, build_store  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--persist-directory", default="artifacts/chroma")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="default", choices=VARIANTS)
    parser.add_argument("--collection-name", default="products")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N rows (smoke test).")
    parser.add_argument(
        "--no-recreate",
        action="store_false",
        dest="recreate",
        help="Upsert into the existing collection instead of deleting it first.",
    )
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Catalog not found at {catalog_path}", file=sys.stderr)
        return 1

    print(f"Embedding {catalog_path} with {args.model!r} (variant={args.variant!r}) ...")
    start_time = time.monotonic()
    store = build_store(
        catalog_path=catalog_path,
        persist_directory=args.persist_directory,
        embedder=Embedder(args.model),
        variant=args.variant,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
    )
    elapsed = time.monotonic() - start_time
    print(f"Wrote {len(store)} vectors to {args.persist_directory} in {elapsed:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
