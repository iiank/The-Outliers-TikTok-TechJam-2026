"""Grid-search BUYING_WEIGHTS / BROWSING_WEIGHTS against the 200 public dev sessions.

Per the Aug 2026 build brief (Task 1): sweep candidate ``{bm25, dense}``
weight pairs independently per mode, running the full local evaluator
against ``data/public_set.jsonl`` for each candidate, and keep whichever
weights score highest on ``TechnicalScore`` for that scenario.

NOT RUNNABLE YET. This script needs a complete multi-turn Agent (dialogue
state, ask_attribute selection, and the reranker all wired together) --
that doesn't exist as of this writing (see the disconnected-pieces
discussion this was built from: only BM25-only ``starter.agent.Agent``
currently implements the required ``reset``/``respond`` interface).
Rather than guess at that agent's internals, this script takes an
*agent factory* as an injected dependency -- whoever builds the full
agent just needs to expose one function matching this contract:

    def build_agent(weights: dict[str, dict[str, float]]) -> Agent:
        '''``weights`` = {"buying": {"bm25": float, "dense": float},
                           "browsing": {"bm25": float, "dense": float}}``
        Returns an object implementing Agent.reset/Agent.respond
        (docs/agent_api_contract.json) that fuses BM25+dense with these
        weights instead of retrieval.rrf.BUYING_WEIGHTS/BROWSING_WEIGHTS.
        '''

Then run:

    python scripts/tune_rrf_weights.py --agent-factory submission.agent:build_agent

Every other function here (score computation, the sweep loop, the CLI)
is real, tested code today -- ``tests/test_tune_rrf_weights.py`` verifies
all of it using a fake agent factory and a fake evaluate function, since
neither the real agent nor a slow 200-session run belongs in unit tests.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

#: (bm25, dense) candidates to sweep per mode. Matches the brief's
#: suggested coarse grid ({1, 2, 3, 5}) -- 16 combinations per mode,
#: ~40s each per the brief's own timing note, so a full sweep of one
#: mode is on the order of ten minutes. Deliberately not normalized
#: (e.g. to sum to 1) -- reciprocal_rank_fusion's weights are relative,
#: not a probability distribution, so absolute scale doesn't matter,
#: only the ratio between them.
DEFAULT_WEIGHT_CANDIDATES: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

EvaluateFn = Callable[..., Dict[str, Any]]
AgentFactory = Callable[[Dict[str, Dict[str, float]]], Any]


def technical_score_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    """Apply evaluate()'s own TechnicalScore formula to any metric_summary()-shaped dict.

    evaluate() only computes TechnicalScore once, over all 200 sessions
    combined (evaluator/local_evaluator.py:279-280) -- it does NOT compute
    it per scenario, even though ``scenario_metrics`` gives per-scenario
    hit_rate/mrr/mttc. This is that same formula, applied to one
    scenario's summary, so buying and browsing can be scored (and tuned)
    independently.

    Returns ``None`` for an empty scenario group (``sample_count == 0``,
    ``mttc is None``) rather than raising or returning a misleading 0.0 --
    a candidate weight setting can't be judged "best" against zero
    sessions.
    """
    if summary.get("sample_count", 0) == 0 or summary.get("mttc") is None:
        return None
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    return 0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency


def import_agent_factory(spec: str) -> AgentFactory:
    """Import an agent factory from a ``'module.path:function_name'`` string."""
    module_path, separator, attr = spec.partition(":")
    if not separator or not module_path or not attr:
        raise ValueError(f"--agent-factory must be 'module.path:function_name', got {spec!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"{module_path!r} has no attribute {attr!r}") from exc


def score_weights(
    weight_table: Dict[str, Dict[str, float]],
    agent_factory: AgentFactory,
    samples: List[dict],
    catalog_ids: set,
    categories: Dict[str, List[str]],
    products: Dict[str, dict],
    evaluate_fn: EvaluateFn = evaluate,
) -> Dict[str, Optional[float]]:
    """Build one agent with ``weight_table``, run it, return TechnicalScore per scenario.

    ``evaluate_fn`` is injected (defaults to the real ``evaluate``) so
    tests can substitute a fake that skips actually running an agent
    through 200 simulated conversations.
    """
    agent = agent_factory(weight_table)
    result = evaluate_fn(agent, samples, catalog_ids, categories, products)
    return {
        scenario: technical_score_from_summary(summary)
        for scenario, summary in result["scenario_metrics"].items()
    }


def grid_search_mode(
    mode: str,
    other_mode_weights: Dict[str, Dict[str, float]],
    agent_factory: AgentFactory,
    samples: List[dict],
    catalog_ids: set,
    categories: Dict[str, List[str]],
    products: Dict[str, dict],
    candidates: Sequence[float] = DEFAULT_WEIGHT_CANDIDATES,
    evaluate_fn: EvaluateFn = evaluate,
) -> List[Dict[str, Any]]:
    """Sweep ``mode``'s (bm25, dense) weights, holding every other mode fixed.

    ``evaluate()`` always scores all 200 sessions in one run (there's no
    way to run "just the buying ones"), so each candidate still requires
    a full run -- this function just reads off the one scenario's score
    from that run's ``scenario_metrics`` and ignores the rest. Returns
    every candidate's result, sorted best-first, so a scenario with no
    sessions (score ``None``) is visible rather than silently skipped.
    """
    results: List[Dict[str, Any]] = []
    for bm25_weight, dense_weight in itertools.product(candidates, candidates):
        weight_table = {**other_mode_weights, mode: {"bm25": bm25_weight, "dense": dense_weight}}
        scores = score_weights(weight_table, agent_factory, samples, catalog_ids, categories, products, evaluate_fn)
        results.append({
            "bm25": bm25_weight,
            "dense": dense_weight,
            "technical_score": scores.get(mode),
            "scenario_metrics": scores,
        })
    results.sort(key=lambda row: (row["technical_score"] is None, -(row["technical_score"] or 0.0)))
    return results


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--agent-factory",
        required=True,
        help="'module.path:function_name' -- see this script's module docstring for the required contract.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["buying", "browsing"], choices=["buying", "browsing"],
        help="Which mode(s) to sweep. Each is tuned independently, with the other held at --default-weights.",
    )
    parser.add_argument(
        "--default-weights", default='{"bm25": 1.0, "dense": 1.0}',
        help="JSON weight dict used for whichever mode isn't currently being swept.",
    )
    parser.add_argument("--candidates", nargs="+", type=float, default=list(DEFAULT_WEIGHT_CANDIDATES))
    parser.add_argument("--output", default="rrf_weight_tuning_results.json")
    args = parser.parse_args(argv)

    try:
        agent_factory = import_agent_factory(args.agent_factory)
    except (ValueError, ImportError, AttributeError) as exc:
        print(f"Could not load --agent-factory {args.agent_factory!r}: {exc}", file=sys.stderr)
        print(
            "This script needs a full multi-turn Agent factory to exist first -- "
            "see the contract described in this script's module docstring.",
            file=sys.stderr,
        )
        return 1

    samples = load_jsonl(args.public_set)
    catalog_ids, categories, products = catalog_index(args.catalog)
    default_weights = json.loads(args.default_weights)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for mode in args.modes:
        other = {"buying": default_weights, "browsing": default_weights}
        other.pop(mode, None)
        print(f"Sweeping {mode} weights ({len(args.candidates)}x{len(args.candidates)} candidates)...")
        results = grid_search_mode(
            mode, other, agent_factory, samples, catalog_ids, categories, products, candidates=args.candidates,
        )
        all_results[mode] = results
        best = results[0]
        print(f"  best for {mode}: bm25={best['bm25']}, dense={best['dense']}, "
              f"technical_score={best['technical_score']}")

    Path(args.output).write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(f"Full results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
