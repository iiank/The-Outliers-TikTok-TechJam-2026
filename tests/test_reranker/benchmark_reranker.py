# tests/benchmark_reranker.py
import json
import os
import tempfile
import time
import numpy as np
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
CATALOG_PATH = ROOT_DIR / "submission" / "src" / "reranker" / "reranker_catalog.jsonl"

from submission.src.reranker.reranker import Reranker, build_reranker_query, load_reranker_catalog


def benchmark_reranker_latency(num_candidates: int = 100, num_runs: int = 50):
    print(f"\n{'='*20} 1. COMPONENT LATENCY BENCHMARK ({num_candidates} Candidates) {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = Reranker(device=device, batch_size=64, use_fp16=True)

    state = {
        "session_profile": {
            "category": ["running shoes"],
            "material": ["mesh"],
            "color": ["black", "white"],
            "feature": ["cushioned sole", "breathable"],
            "use_case": ["marathon training"],
        },
        "user_profile": {"preference_tags": ["durability", "lightweight"]},
    }

    dummy_candidates = [
        {
            "parent_asin": f"B000_{i}",
            "document": f"Title: Performance Running Shoe Model {i} | Category: Shoes > Running | Material: Engineered Mesh | Features: Cushioned midsole, rubber outsole | Brand: Brand_{i % 5}",
        }
        for i in range(num_candidates)
    ]

    query_times = []
    pairing_times = []
    inference_times = []
    sorting_times = []
    total_times = []

    # Warmup pass
    for _ in range(5):
        q = build_reranker_query(state)
        reranker.rank(query=q, candidates=dummy_candidates, top_k=10)

    for _ in range(num_runs):
        t0 = time.perf_counter()

        # Step A: Query construction
        t_query_start = time.perf_counter()
        query = build_reranker_query(state)
        t_query_end = time.perf_counter()

        # Step B: Pair construction
        t_pair_start = time.perf_counter()
        pairs = [(query, doc.get("document", "")) for doc in dummy_candidates]
        t_pair_end = time.perf_counter()

        # Step C: Model forward pass
        t_inf_start = time.perf_counter()
        scores = reranker.model.predict(
            pairs,
            batch_size=reranker.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        t_inf_end = time.perf_counter()

        # Step D: Scoring & Sorting
        t_sort_start = time.perf_counter()
        for doc, score in zip(dummy_candidates, scores):
            doc["score"] = float(score)
        _ = sorted(dummy_candidates, key=lambda x: x["score"], reverse=True)[:10]
        t_sort_end = time.perf_counter()

        t_end = time.perf_counter()

        query_times.append((t_query_end - t_query_start) * 1000)
        pairing_times.append((t_pair_end - t_pair_start) * 1000)
        inference_times.append((t_inf_end - t_inf_start) * 1000)
        sorting_times.append((t_sort_end - t_sort_start) * 1000)
        total_times.append((t_end - t0) * 1000)

    print(f"Device Used:               {device.upper()} (FP16: {device == 'cuda'})")
    print(f"Query Parsing:             {np.mean(query_times):.3f} ms ± {np.std(query_times):.3f} ms")
    print(f"Pair Construction:         {np.mean(pairing_times):.3f} ms ± {np.std(pairing_times):.3f} ms")
    print(f"Cross-Encoder Inference:   {np.mean(inference_times):.3f} ms ± {np.std(inference_times):.3f} ms")
    print(f"Score Attachment & Sort:   {np.mean(sorting_times):.3f} ms ± {np.std(sorting_times):.3f} ms")
    print(f"Total Turn Rerank Latency: {np.mean(total_times):.3f} ms (p95: {np.percentile(total_times, 95):.3f} ms)")


def benchmark_catalog_cold_start(catalog_path: Path = CATALOG_PATH, catalog_size: int = 50000):
    print(f"\n{'='*20} 2. COLD START CATALOG LOADING BENCHMARK {'='*20}")
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found at: {catalog_path}")
    
    tmp_path = "submission/src/reranker/reranker_catalog.jsonl"

    t0 = time.perf_counter()
    catalog = load_reranker_catalog(str(catalog_path))
    load_duration = (time.perf_counter() - t0) * 1000

    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"Catalog Records:          {len(catalog):,} items")
    print(f"File Size on Disk:        {file_size_mb:.2f} MB")
    print(f"Cold Start Ingestion:     {load_duration:.2f} ms ({load_duration / 1000:.3f} s)")
    
    return catalog

def benchmark_reranker_pipeline():
    print(f"\n{'='*20} 3. FULL RERANKER COMPONENT BENCHMARK {'='*20}")

    # 1. Target product definition
    target_product = {
        "parent_asin": "B01HGHPJFU",
        "document": "Title: Hotouch Men's Floral All Over Print Button Down Short Sleeve Shirt Red Hibiscus S | Brand: HOTOUCH | Category: Shirts > Casual Button-Down Shirts | Features: Button closure",
        "price": None,
        "average_rating": 4.6,
    }
    
    # sample state definition
    sample_state = {
        "session_id": "demo-session",
        "turn": 1,
        "session_profile": {
            "category": ["Casual Button-Down Shirts"],
            "material": [],
            "color": ["red"],
            "size": [],
            "style": [],
            "brand": [],
            "budget": [],
            "feature": [],
            "use_case": [],
            "other": [],
            "rejected": []
        },
        "user_profile": {
            "purchase_frequency": "3-4 prior purchases",
            "average_prior_rating": 5.0,
            "rating_style": "usually positive",
            "preference_tags": ["fit", "comfort", "durability"],
            "summary": "Prior purchases emphasize fit, comfort, durability; ratings are usually positive."
        },
        "previous_top_10": [],
        "conflicts_with_previous": False
    }
    
    # 2. Load catalog
    catalog = load_reranker_catalog(str(CATALOG_PATH))
    catalog_items = list(catalog.items()) if isinstance(catalog, dict) else list(catalog)

    # 3. Extract top 100 candidate items (indices: 27640 < i <= 27740 -> slice [27641:27741])
    start_idx, end_idx = 27641, 27741
    candidate_indices = [i for i in range(27640, 27740)]
    
    # 4. Initialize and run Reranker
    t_init_0 = time.perf_counter()
    reranker = Reranker()
    init_duration = (time.perf_counter() - t_init_0) * 1000

    t_rerank_0 = time.perf_counter()
    ranked_results = reranker.rank_from_state(
            state=sample_state,
            candidate_indices=candidate_indices,
            catalog=catalog_items,
            top_k=10
        )
    rerank_duration = (time.perf_counter() - t_rerank_0) * 1000
    
    # 5. Benchmark Metrics Output
    print(f"Model Init Duration:      {init_duration:.2f} ms")
    print(f"Rerank Duration (100 it): {rerank_duration:.2f} ms")
    print(f"Throughput:               {len(candidate_indices) / (rerank_duration / 1000):.1f} items/sec")
    
    # 6. Print Top 10 Ranked Results
    print(f"\n{'-'*15} TOP 10 RERANKED RESULTS {'-'*15}")
    for rank, item in enumerate(ranked_results[:10], 1):
        asin = item.get("parent_asin", "N/A")
        score = item.get("score", "N/A")
        doc_snippet = item.get("document", "")[:80] + "..." if len(item.get("document", "")) > 80 else item.get("document", "")
        print(f"Rank {rank:02d} | ASIN: {asin} | Score: {score} | {doc_snippet}")

if __name__ == "__main__":
    benchmark_reranker_latency(num_candidates=100, num_runs=30)
    benchmark_catalog_cold_start()
    benchmark_reranker_pipeline()