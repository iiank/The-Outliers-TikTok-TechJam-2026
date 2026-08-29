from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

from embed.embedder import DEFAULT_MODEL, Embedder
from embed.product_text import VARIANTS
from embed.store import build_store, load_store


PROFILE_FIELDS = ("none", "summary", "tags", "both")


def profile_suffix(sample: dict, field: str) -> str:
    if field == "none":
        return ""
    profile = sample.get("user_profile") or {}
    parts: list[str] = []
    if field in ("summary", "both"):
        summary = str(profile.get("summary") or "").strip()
        if summary:
            parts.append(summary)
    if field in ("tags", "both"):
        tags = [str(t) for t in (profile.get("preference_tags") or []) if t]
        if tags:
            parts.append("Preferences: " + ", ".join(tags) + ".")
    return " " + " ".join(parts) if parts else ""


def turn_one_queries(
    samples: list[dict], categories: dict, products: dict, profile_field: str = "none"
) -> list[dict]:
<<<<<<< HEAD
    """Return one record per scorable sample: query, target and scenario type."""
=======
>>>>>>> dc60b38acb2782dd7eeaf7a4b87d830b273f794b
    records: list[dict] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        if target not in products:
            continue
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        base = initial_message(effective, coarse_category(categories.get(target, [])), set())
        records.append({
            "query": base + profile_suffix(sample, profile_field),
            "target": target,
            "scenario_type": sample.get("scenario_type", "unknown"),
        })
    return records


def recall_at(results: list[list[tuple[str, float]]], targets: list[str], k: int) -> float:
    hits = sum(1 for res, t in zip(results, targets) if t in [a for a, _ in res[:k]])
    return round(hits / len(targets), 4) if targets else 0.0


def summarise(results: list[list[tuple[str, float]]], targets: list[str]) -> dict:
    return {
        "recall_at_10": recall_at(results, targets, 10),
<<<<<<< HEAD
        "recall_at_100": recall_at(results, targets, 100),
        "recall_at_500": recall_at(results, targets, 500),
=======
        "recall_at_50": recall_at(results, targets, 50),
        "recall_at_100": recall_at(results, targets, 100),
>>>>>>> dc60b38acb2782dd7eeaf7a4b87d830b273f794b
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-1 dense retrieval recall")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="default", choices=list(VARIANTS))
    parser.add_argument("--persist", default="artifacts/chroma")
    parser.add_argument("--collection", default="products")
    parser.add_argument("--reuse", action="store_true", help="open the prebuilt store instead of rebuilding")
    parser.add_argument("--profile", default="none", choices=list(PROFILE_FIELDS),
                        help="append user_profile text to the query: none, summary, tags or both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    embedder = Embedder(args.model)

    if args.reuse:
        store = load_store(args.persist, embedder=embedder, collection_name=args.collection)
    else:
        store = build_store(
            args.catalog, persist_directory=args.persist, embedder=embedder,
            variant=args.variant, collection_name=args.collection, limit=args.limit,
        )

    records = turn_one_queries(samples, categories, products, profile_field=args.profile)
    queries = [r["query"] for r in records]
    targets = [r["target"] for r in records]
<<<<<<< HEAD
    results = store.search_batch(queries, k=500)
=======
    results = store.search_batch(queries, k=100)
>>>>>>> dc60b38acb2782dd7eeaf7a4b87d830b273f794b

    report = {
        "model": embedder.model_name,
        "variant": store.variant,
        "profile": args.profile,
        "catalogue_size": len(store),
        "samples_scored": len(targets),
        "overall": summarise(results, targets),
        "by_scenario": {},
    }
    for scenario in sorted({r["scenario_type"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["scenario_type"] == scenario]
        report["by_scenario"][scenario] = {
            "sample_count": len(idx),
            **summarise([results[i] for i in idx], [targets[i] for i in idx]),
        }

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()