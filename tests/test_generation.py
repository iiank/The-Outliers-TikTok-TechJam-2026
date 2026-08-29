from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)

from embed.store import load_store
from generation import AskState, AttributeTable, select_attribute
from generation.weighted_entropy import ASKABLE_ATTRIBUTES, MIN_SCORE
from retrieval import BM25Index, CatalogIndex, CategoryLookup, Retriever
from retrieval.category_filter import extract_coarse_category
from retrieval.rrf import weights_for_mode

MODE_FOR_SCENARIO = {
    "buying": "buying",
    "browsing": "browsing",
    "intent_override": "buying",
    "boundary": "buying",
}

RECALL_CUTOFFS = (10, 100, 500)

_NO_PREFERENCE = re.compile(r"I don't have (?:an additional )?preference for ([a-z_]+)", re.I)


def build_cases(samples: Sequence[dict], categories: dict, products: dict) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        if target not in products:
            continue
        card, behavior = materialize_hidden_fields(sample, products)
        scenario = str(sample.get("scenario_type", "buying"))
        cases.append(
            {
                "sample": {**sample, "intent_card": card, "behavior": behavior},
                "sample_id": sample.get("sample_id"),
                "scenario": scenario,
                "mode": MODE_FOR_SCENARIO.get(scenario, "buying"),
                "target": target,
                "category": coarse_category(categories.get(target, [])),
                "tags": (sample.get("user_profile") or {}).get("preference_tags") or [],
            }
        )
    return cases


def rank_of(ranking: Sequence[Tuple[str, float]], target: str) -> Optional[int]:
    for position, (parent_asin, _score) in enumerate(ranking, start=1):
        if parent_asin == target:
            return position
    return None


def recall_table(ranks: Sequence[Optional[int]]) -> Dict[str, Any]:
    total = len(ranks) or 1
    table: Dict[str, Any] = {
        f"recall_at_{k}": round(sum(1 for r in ranks if r is not None and r <= k) / total, 4)
        for k in RECALL_CUTOFFS
    }
    hits = [r for r in ranks if r is not None]
    table["mrr"] = round(sum(1.0 / r for r in hits) / total, 4)
    table["median_rank"] = sorted(hits)[len(hits) // 2] if hits else None
    return table


def run_session(
    case: Dict[str, Any],
    retriever: Retriever,
    table: AttributeTable,
    catalog_ids: set,
    clarify: bool,
    pool_size: int,
    min_score: float = MIN_SCORE,
) -> Dict[str, Any]:
    sample = case["sample"]
    target = case["target"]
    scenario = case["scenario"]

    disclosed: set = set()
    boundary_used = False
    override_applied = scenario != "intent_override"
    ask_state = AskState()

    user_message = initial_message(sample, case["category"], disclosed)
    category_message = user_message
    transcript: List[str] = [user_message]

    hit_turn: Optional[int] = None
    best_rank: Optional[int] = None
    asked: List[Optional[str]] = []
    first_pool_rank: Optional[int] = None

    for turn in range(1, MAX_TURNS + 1):
        query_terms = " ".join(transcript)
        result = retriever.retrieve(
            query_terms, case["mode"], message=category_message, entropy_pool_size=pool_size
        )
        pool = result.entropy_pool
        fused = [(item["parent_asin"], item["score"]) for item in pool]

        if turn == 1:
            first_pool_rank = rank_of(fused, target)
        ranked = normalize_recommendations(
            [{"parent_asin": asin} for asin, _ in fused[:TOP_K]], catalog_ids
        )

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            break
        if turn == MAX_TURNS:
            break

        ask_attribute: Optional[str] = None
        if clarify:
            ask_attribute, _ = select_attribute(
                pool, table, ask_state, case["tags"], min_score=min_score
            )
        asked.append(ask_attribute)

        override = sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            ask_state = AskState()
        else:
            user_message, boundary_used = customer_reply(
                sample, ask_attribute, disclosed, boundary_used
            )
            if ask_attribute:
                if _NO_PREFERENCE.search(user_message):
                    ask_state.ban(ask_attribute)
                else:
                    ask_state.confirm(ask_attribute)

        transcript.append(user_message)
        if extract_coarse_category(user_message):
            category_message = user_message

    return {
        "sample_id": case["sample_id"],
        "scenario_type": scenario,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "turn_one_pool_rank": first_pool_rank,
        "asked": asked,
    }


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval and question-policy integration check")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--persist", default="artifacts/chroma")
    parser.add_argument("--pool-size", type=int, default=500)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help="ask gate: higher asks less often. Sweep 0.0 0.05 0.15 0.3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-category-filter", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("loading catalogue, store and routes ...")
    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_id_set, categories, products = catalog_index(args.catalog)

    store = load_store(args.persist)
    catalog_ids = CatalogIndex.load(args.catalog)
    retriever = Retriever(
        bm25=BM25Index(args.catalog),
        dense=store,
        catalog_index=catalog_ids,
        category_lookup=None if args.no_category_filter else CategoryLookup.load(args.catalog),
    )
    table = AttributeTable.load(args.catalog)
    cases = build_cases(samples, categories, products)
    print(f"catalogue {len(catalog_ids)} | dense store {len(store)} | "
          f"askable {', '.join(table.attributes())}")
    print(f"scoring {len(cases)} of {len(samples)} samples | min_score={args.min_score}\n")

    section("1. turn-1 recall by route  (where the purchased product ranks)")
    route_ranks: Dict[str, Dict[str, List[Optional[int]]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        message = initial_message(case["sample"], case["category"], set())
        result = retriever.retrieve(message, case["mode"], message=message, entropy_pool_size=args.pool_size)
        fused = [(i["parent_asin"], i["score"]) for i in result.entropy_pool]
        for route, ranking in (
            ("fused", fused),
            ("dense", list(store.search(message, args.pool_size))),
            ("bm25", retriever.bm25.search(message, args.pool_size)),
        ):
            rank = rank_of(ranking, case["target"])
            route_ranks["ALL"][route].append(rank)
            route_ranks[case["scenario"]][route].append(rank)

    print(f"{'scenario':<17}{'n':>4}{'route':>8}" + "".join(f"{'@' + str(k):>9}" for k in RECALL_CUTOFFS)
          + f"{'mrr':>8}{'median':>8}")
    for scenario in ["ALL"] + sorted(s for s in route_ranks if s != "ALL"):
        for route in ("fused", "dense", "bm25"):
            stats = recall_table(route_ranks[scenario][route])
            head = f"{scenario:<17}{len(route_ranks[scenario][route]):>4}" if route == "fused" else " " * 21
            print(head + f"{route:>8}"
                  + "".join(f"{stats['recall_at_' + str(k)]:>9}" for k in RECALL_CUTOFFS)
                  + f"{stats['mrr']:>8}{str(stats['median_rank']):>8}")
        print()

    section("2. does fusing beat the better single route?")
    for mode in ("buying", "browsing"):
        scenarios = [s for s, m in MODE_FOR_SCENARIO.items() if m == mode and s in route_ranks]
        if not scenarios:
            continue
        merged = {r: [x for s in scenarios for x in route_ranks[s][r]] for r in ("fused", "dense", "bm25")}
        stats = {r: recall_table(v)["recall_at_100"] for r, v in merged.items()}
        best_single = max(stats["dense"], stats["bm25"])
        verdict = ("fusion helps" if stats["fused"] > best_single
                   else "fusion neutral" if stats["fused"] == best_single
                   else "FUSION HURTS -- revisit weights")
        print(f"{mode:<10} weights={dict(zip(('bm25', 'dense'), weights_for_mode(mode)))}  "
              f"recall@100  fused {stats['fused']}  dense {stats['dense']}  bm25 {stats['bm25']}"
              f"   -> {verdict}")

    arms: Dict[str, List[Dict[str, Any]]] = {}
    for label, clarify in (("clarify", True), ("silent", False)):
        arms[label] = [
            run_session(case, retriever, table, catalog_id_set, clarify,
                        args.pool_size, args.min_score)
            for case in cases
        ]

    section("3. multi-turn conversation metrics  (evaluator's own simulator)")
    print(f"{'arm':<10}{'scenario':<17}{'n':>4}{'hit@10':>9}{'mrr':>9}{'mttc':>8}{'score':>9}")
    summary: Dict[str, Dict[str, Any]] = {}
    for label, sessions in arms.items():
        summary[label] = {}
        for scenario in ["ALL"] + sorted({s["scenario_type"] for s in sessions}):
            subset = sessions if scenario == "ALL" else [s for s in sessions if s["scenario_type"] == scenario]
            metrics = metric_summary(subset)
            efficiency = max(0.0, min(1.0, (11.0 - float(metrics["mttc"])) / 10.0))
            score = round(0.50 * metrics["hit_rate_at_10"] + 0.30 * metrics["mrr"] + 0.20 * efficiency, 5)
            summary[label][scenario] = {**metrics, "efficiency": round(efficiency, 4), "score": score}
            print(f"{label:<10}{scenario:<17}{metrics['sample_count']:>4}"
                  f"{metrics['hit_rate_at_10']:>9}{metrics['mrr']:>9}{metrics['mttc']:>8}{score:>9}")
        print()

    section("4. what the clarification policy is worth")
    clar, sil = summary["clarify"]["ALL"], summary["silent"]["ALL"]
    delta_score = clar["score"] - sil["score"]
    delta_mttc = clar["mttc"] - sil["mttc"]
    print(f"score  {sil['score']} -> {clar['score']}   ({delta_score:+.5f})")
    print(f"hit@10 {sil['hit_rate_at_10']} -> {clar['hit_rate_at_10']}")
    print(f"mttc   {sil['mttc']} -> {clar['mttc']}   ({delta_mttc:+.3f} turns, lower is better)")
    if delta_score > 0:
        print("\nAsking helps. Sweep min_score and half_life next to find how hard to lean on it.")
    elif delta_score == 0:
        print("\nNo difference. Check section 5: the policy may be asking unanswerable attributes.")
    else:
        print("\nAsking costs more than it returns on this dev set. Raising min_score in")
        print("select_attribute makes the policy ask less often; that is the knob to sweep.")

    section("5. which attributes were asked, and how often")
    asked = Counter(a for s in arms["clarify"] for a in s["asked"] if a)
    none_asked = sum(1 for s in arms["clarify"] for a in s["asked"] if a is None)
    total_asks = sum(asked.values()) + none_asked
    if total_asks:
        for name, count in asked.most_common():
            print(f"  {name:<12}{count:>5}  ({count / total_asks:.0%} of turns)")
        print(f"  {'(no ask)':<12}{none_asked:>5}  ({none_asked / total_asks:.0%} of turns)")
    unused = [a for a in ASKABLE_ATTRIBUTES if a not in asked]
    if unused:
        print(f"  never chosen: {', '.join(unused)}")

    if args.output:
        payload = {
            "catalogue_size": len(catalog_ids),
            "samples_scored": len(cases),
            "min_score": args.min_score,
            "turn_one_recall": {
                s: {r: recall_table(v) for r, v in routes.items()} for s, routes in route_ranks.items()
            },
            "conversation_metrics": summary,
            "ask_distribution": dict(asked),
            "sessions": arms,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()