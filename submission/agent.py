# submission/agent.py
from src.reranker.reranker import ProductReranker, load_catalog

class SearchAgent:
    def __init__(self, catalog_path: str = "reranker_catalog.jsonl"):
        # 1. Warm load data into RAM once
        self.catalog = load_catalog(catalog_path)
        
        # 2. Warm load model once (persists across queries)
        self.reranker = ProductReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=192,
            batch_size=64
        )

    def process_query(self, session_state: str, retrieved_top_indices: list[int]):
        # Rerank directly against warm memory
        top_results = self.reranker.rank_indices(
            query=session_state,
            target_indices=retrieved_top_indices,
            catalog=self.catalog,
            top_k=10
        )
        return top_results