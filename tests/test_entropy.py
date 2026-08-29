from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generation import AskState, AttributeTable, rank_attributes, select_attribute
from generation.weighted_entropy import ASKABLE_ATTRIBUTES, rank_weights

CATALOG = "data/catalog.jsonl"
PERSIST = "artifacts/chroma"
SAMPLE_MESSAGE = "I'm looking for Earrings Hoop. A key requirement is: spandex."
TAGS = ["fit", "comfort", "durability"]

def check_coverage(table: AttributeTable) -> None:
    print("=" * 62)
    print("1. attribute coverage across the catalogue")
    print("=" * 62)
    print(f"{'attribute':<12}{'coverage':>10}{'distinct':>10}  top values")
    for name in table.attributes():
        codes = table.codes[name]
        known = int((codes >= 0).sum())
        vocab = table.vocabs[name]
        share = known / table.n_products if table.n_products else 0.0
        if known:
            counts = np.bincount(codes[codes >= 0], minlength=len(vocab))
            top = ", ".join(f"{vocab[i]} ({counts[i]})" for i in np.argsort(-counts)[:3])
        else:
            top = "-"
        flag = "  <-- too sparse to use" if share < 0.05 else ""
        print(f"{name:<12}{share:>9.1%}{len(vocab):>10}  {top}{flag}")
    print()


def check_real_pool(table: AttributeTable) -> None:
    print("=" * 62)
    print("2. scoring over a real fused pool")
    print("=" * 62)

    if not Path(PERSIST).exists():
        print(f"skipped: no store at {PERSIST}. Build it first.\n")
        return

    try:
        from embed.store import load_store
        from retrieval import BM25Index, CatalogIndex, CategoryLookup, Retriever
    except ImportError as error:
        print(f"skipped: {error}\n")
        return

    store = load_store(PERSIST)
    retriever = Retriever(
        bm25=BM25Index(CATALOG),
        dense=store,
        catalog_index=CatalogIndex.load(CATALOG),
        category_lookup=CategoryLookup.load(CATALOG),
    )
    result = retriever.retrieve(SAMPLE_MESSAGE, "buying", message=SAMPLE_MESSAGE)
    pool = result.entropy_pool
    print(f"message: {SAMPLE_MESSAGE}")
    print(f"pool: {len(pool)} candidates, {len(result.reranker_pool)} for reranking\n")

    if not pool:
        print("empty pool: nothing to score.\n")
        return

    weights = rank_weights(len(pool))
    print(f"rank weights: r1={weights[0]:.3f}  r20={weights[min(19, len(weights) - 1)]:.3f}  "
          f"r{len(pool)}={weights[-1]:.2e}\n")

    for tags in (None, TAGS):
        label = "no tags" if tags is None else f"tags {tags}"
        print(f"-- {label} --")
        for score in rank_attributes(pool, table, preference_tags=tags)[:5]:
            values = ", ".join(f"{v}" for v, _ in score.top_values[:3])
            print(f"  {score.attribute:<10} score={score.score:6.3f}  ent={score.entropy:5.2f}  "
                  f"head={score.head_split:.2f}  cov={score.coverage:.2f}  w={score.tag_weight:.2f}  [{values}]")
        chosen, _ = select_attribute(pool, table, preference_tags=tags)
        print(f"  -> ask: {chosen}\n")


def check_state(table: AttributeTable) -> None:
    print("=" * 62)
    print("3. ask-state transitions")
    print("=" * 62)

    pool = [{"parent_asin": f"X{i}", "catalog_index": i} for i in range(min(500, table.n_products))]
    state = AskState()

    first, _ = select_attribute(pool, table, state, TAGS)
    print(f"initial argmax:              {first}")
    if first is None:
        print("nothing selectable, so the remaining transitions cannot be exercised.\n")
        return

    state.refute(first)
    after_one, ranked = select_attribute(pool, table, state, TAGS)
    penalty = next((s.refusal_penalty for s in ranked if s.attribute == first), 1.0)
    print(f"after refute({first}):        {after_one}   (penalty {penalty:.2f}, still eligible)")

    state.confirm(first)
    after_confirm, ranked = select_attribute(pool, table, state, TAGS)
    blocked = rank_attributes(pool, table, state, TAGS, include_blocked=True)
    score = next((s.score for s in blocked if s.attribute == first), None)
    print(f"after confirm({first}):       {after_confirm}   ({first} now scores {score})")

    for name in ASKABLE_ATTRIBUTES:
        state.ban(name)
    exhausted, _ = select_attribute(pool, table, state, TAGS)
    print(f"after banning everything:    {exhausted}   (None means recommend)")
    print(f"empty pool:                  {select_attribute([], table)[0]}\n")


def main() -> None:
    print(f"loading attribute table from {CATALOG} ...")
    table = AttributeTable.load(CATALOG)
    print(f"{table.n_products} products, attributes: {', '.join(table.attributes())}\n")

    check_coverage(table)
    check_real_pool(table)
    check_state(table)


if __name__ == "__main__":
    main()