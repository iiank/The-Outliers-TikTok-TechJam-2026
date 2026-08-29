# Output dictionary passed from agent/orchestrator to Streamlit
'''
Chat UI + Internal State Inspector
'''

'''
########## Pipeline turn contract ##########
turn_output = {
    "message": response["message"],                     # String
    "ask_attribute": response["ask_attribute"],         # String[cite: 1]
    "recommendations": response["recommendations"],     # Top 10 ASINs + metadata
    "diagnostics": {
        "dynamic_state": state_dict,                    # Parsed session_profile & constraints
        "hard_filters_dropped": len(failed_asins),      # Count / sample of dropped ASINs[cite: 2, 3]
        "retrieval_counts": {
            "pre_filtered_pool": 50000 - len(failed_asins), #[cite: 1, 3]
            "bm25_top": 50,                             #[cite: 1, 2]
            "dense_top": 50,                            #[cite: 1, 2]
            "rrf_pool": 100                             #[cite: 1, 2]
        },
        "top_candidates_ce": [                          # Top 5 cross-encoder items with scores[cite: 1, 2]
            {"asin": doc["parent_asin"], "score": doc["score"], "title": doc.get("title", "")}
            for doc in reranked_docs[:5]
        ],
        "entropy_scores": entropy_attr_distribution     # Key-value map of calculated entropies
    }
}
'''

