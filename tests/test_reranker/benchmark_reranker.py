# tests/benchmark_reranker.py
import json
import os
import tempfile
import time
import numpy as np
import torch

from submission.src.reranker.reranker import ProductReranker, build_reranker_query, load_catalog


def benchmark_reranker_latency(num_candidates: int = 100, num_runs: int = 50):
    print(f"\n{'='*20} 1. COMPONENT LATENCY BENCHMARK ({num_candidates} Candidates) {'='*20}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = ProductReranker(device=device, batch_size=64, use_fp16=True)

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

    print(f"Device Used:              {device.upper()} (FP16: {device == 'cuda'})")
    print(f"Query Parsing:            {np.mean(query_times):.3f} ms ± {np.std(query_times):.3f} ms")
    print(f"Pair Construction:        {np.mean(pairing_times):.3f} ms ± {np.std(pairing_times):.3f} ms")
    print(f"Cross-Encoder Inference:  {np.mean(inference_times):.3f} ms ± {np.std(inference_times):.3f} ms")
    print(f"Score Attachment & Sort:  {np.mean(sorting_times):.3f} ms ± {np.std(sorting_times):.3f} ms")
    print(f"Total Turn Rerank Latency: {np.mean(total_times):.3f} ms (p95: {np.percentile(total_times, 95):.3f} ms)")


def benchmark_catalog_cold_start(catalog_size: int = 50000):
    print(f"\n{'='*20} 2. COLD START CATALOG LOADING BENCHMARK {'='*20}")
    
    # Generate synthetic 50k backlog jsonl
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        sample_doc = {
            "parent_asin": "B08TEST000",
            "document": "Title: Classic Leather Running Shoes | Category: Shoes > Running | Material: Leather, Mesh | Features: Padded collar, EVA midsole | Brand: SportCo",
        }
        for i in range(catalog_size):
            sample_doc["parent_asin"] = f"B08_{i:06d}"
            tmp.write(json.dumps(sample_doc) + "\n")
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        catalog = load_catalog(tmp_path)
        load_duration = (time.perf_counter() - t0) * 1000

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"Catalog Records:          {len(catalog):,} items")
        print(f"File Size on Disk:        {file_size_mb:.2f} MB")
        print(f"Cold Start Ingestion:     {load_duration:.2f} ms ({load_duration / 1000:.3f} s)")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    benchmark_reranker_latency(num_candidates=100, num_runs=30)
    benchmark_catalog_cold_start(catalog_size=50000)