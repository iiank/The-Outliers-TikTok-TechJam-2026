from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    catalog_index,
    classify_constraint,
    intent_card,
    load_jsonl,
    materialize_hidden_fields,
)

from generation.weighted_entropy import ANSWERABILITY, ASKABLE_ATTRIBUTES


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-attribute answerability")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, _, products = catalog_index(args.catalog)

    hard_only = Counter()
    soft_only = Counter()
    any_bucket = Counter()
    by_scenario = defaultdict(Counter)
    scenario_totals = Counter()
    scored = 0

    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        if target not in products:
            continue
        scored += 1
        scenario = str(sample.get("scenario_type", "unknown"))
        scenario_totals[scenario] += 1

        card, _ = materialize_hidden_fields(sample, products)
        hard = [str(v) for v in card.get("hard_constraints", [])]
        soft = [str(v) for v in card.get("soft_preferences", [])]

        hard_kinds = {classify_constraint(v) for v in hard}
        soft_kinds = {classify_constraint(v) for v in soft}
        both = hard_kinds | soft_kinds

        hard_only.update(hard_kinds)
        soft_only.update(soft_kinds)
        any_bucket.update(both)
        by_scenario[scenario].update(both)

    print(f"scored {scored} of {len(samples)} samples\n")

    print("=" * 70)
    print("share of samples with at least one constraint of each kind")
    print("=" * 70)
    print(f"{'attribute':<12}{'any':>9}{'hard':>9}{'soft':>9}{'prior':>9}{'suggested':>11}")

    suggested = {}
    order = [a for a in ALLOWED_ATTRIBUTES if a in any_bucket or a in ASKABLE_ATTRIBUTES]
    for attribute in order:
        share = any_bucket[attribute] / scored if scored else 0.0
        prior = ANSWERABILITY.get(attribute)
        value = round(share, 2)
        suggested[attribute] = value
        askable = "" if attribute in ASKABLE_ATTRIBUTES else "   (not askable)"
        print(
            f"{attribute:<12}{share:>8.1%}"
            f"{hard_only[attribute] / scored if scored else 0:>8.1%}"
            f"{soft_only[attribute] / scored if scored else 0:>8.1%}"
            f"{('-' if prior is None else f'{prior:.2f}'):>9}"
            f"{value:>11.2f}{askable}"
        )

    unreachable = [a for a in ASKABLE_ATTRIBUTES if any_bucket[a] == 0]
    if unreachable:
        print(f"\nNever produced by any sample: {', '.join(unreachable)}")
        print("Asking about these always returns a non-answer. Remove them from")
        print("ASKABLE_ATTRIBUTES rather than leaving them to win on entropy.")

    print("\n" + "=" * 70)
    print("by scenario")
    print("=" * 70)
    scenarios = sorted(scenario_totals)
    print(f"{'attribute':<12}" + "".join(f"{s[:10]:>12}" for s in scenarios))
    for attribute in ASKABLE_ATTRIBUTES:
        row = "".join(
            f"{by_scenario[s][attribute] / scenario_totals[s]:>11.0%} " for s in scenarios
        )
        print(f"{attribute:<12}{row}")

    print("\nsuggested ANSWERABILITY block:\n")
    print("ANSWERABILITY: Dict[str, float] = {")
    for attribute in ASKABLE_ATTRIBUTES:
        print(f'    "{attribute}": {suggested.get(attribute, 0.0):.2f},')
    print("}")

    if args.output:
        payload = {
            "samples_scored": scored,
            "answerability": {a: suggested.get(a, 0.0) for a in ASKABLE_ATTRIBUTES},
            "any_constraint": dict(any_bucket),
            "hard_constraint": dict(hard_only),
            "soft_constraint": dict(soft_only),
            "by_scenario": {s: dict(c) for s, c in by_scenario.items()},
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()