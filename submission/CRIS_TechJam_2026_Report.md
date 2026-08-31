# Conversational Recommender & Intelligent Search (CRIS)

## TechJam 2026 Report

**Team Name:** The Outliers  
**Problem Statement 4:** Shopping Copilot  
**Deployment:** [CRIS Live Demo](https://f7omkxpgn4oarytaul29be.streamlit.app/)

---

## What CRIS Does

CRIS holds a conversation with a customer across up to 10 turns to find one hidden target product in a 50,000-item Amazon clothing, shoes, and jewelry catalog.

Each turn can:

- Ask a clarification question about one of ten fixed attributes: category, material, color, size, style, brand, budget, feature, use case, or none.
- Return up to 10 ranked recommendations.
- Do both.

On every turn, CRIS routes the user's intent as either **buying** or **browsing**, which may change during the conversation. It tracks when a customer changes their mind and does not re-ask for information that the customer has already declined to specify.

## How We Address the Problem Statement

Problem Statement 4 requires a multi-turn agent to find a hidden target product within 10 turns across a session mix of **Buying, Browsing, Intent Override, and Boundary** scenarios. The agent is never told which scenario it is in, so routing must be inferred live from the conversation.

CRIS addresses each part of that requirement directly.

### Intent Routing and Hybrid Retrieval

One LLM call per turn jointly classifies user input into **buying** or **browsing** intent and extracts constraint slots. This drives a category hard filter, followed by BM25 and dense retrieval fused with **Reciprocal Rank Fusion (RRF)**. Route-specific weights are then applied by mode, followed by a cross-encoder reranker.

### Dialogue Strategy

A dialogue-state tracker maintains ten attribute slots plus a rejected list across turns. It distinguishes incremental accumulation from intent override and tracks unanswered-attribute counts so that the agent stops re-asking questions. This supports the Boundary case.

### Personalisation & Clarification

The customer's long-term `preference_tags` bias which attribute the entropy-based question policy asks next, allowing the clarification strategy to adapt to each customer.

In addition, entropy calculations based on information gain per attribute enable CRIS to ask the most effective question for narrowing down the target product.

## Development Tools Used

- Python 3.10
- Visual Studio Code
- Git/GitHub
- Streamlit (demo dashboard)
- pytest/unittest

## APIs Used

- **Google Gemini** — OpenAI-compatible model `gemini-3.5-flash-lite`. Used for joint intent and slot extraction on every turn. It is called through a schema-constrained client (`state/llm_client.py`, `urllib` only, no SDK). The provider is swappable to OpenRouter, Gemini, or local Ollama via environment variables.
- **Anthropic Messages API** — model `claude-haiku-4-5`, used for optional message phrasing. This is active only when `ANTHROPIC_API_KEY` is set. The shipped `.env` sets only `LLM_API_KEY`, so this path is **disabled by default**; a template builder handles phrasing instead. Future development may opt for LLM message building for more natural language.

No model was fine-tuned or trained from scratch.

## Libraries and Frameworks Used

- `numpy`
- `pandas`
- `scipy` (`scipy.stats.entropy`)
- `sentence-transformers` (`all-MiniLM-L6-v2` embedder)
- `sentence-transformers.CrossEncoder` (`ms-marco-MiniLM-L-6-v2`)
- `torch`
- `chromadb` (local, not hosted)
- `sqlite3` FTS5
- `streamlit`
- `anthropic` SDK
- `pytest/unittest`

## Datasets and Assets Used

- **Amazon Reviews 2023** (McAuley Lab, UCSD), Clothing_Shoes_and_Jewelry category, joined on `parent_asin`. Text and structured metadata only.
- **Frozen 50,000-product catalog** and **200 labeled public development sessions**:
  - 80 buying
  - 80 browsing
  - 30 intent-override
  - 10 boundary
- **800 private holdout sessions** (organizer-only).
- Two derived local assets:
  - Condensed `reranker_catalog.jsonl` (`scripts/reranker_catalog_parser.py`)
  - Persistent Chroma embedding store (`scripts/build_chroma_store.py`)

## Latency & Token Usage

- Each HTTP extraction call enforces a 20-second timeout with up to two attempts, bounding worst-case extraction failure latency to approximately 40 seconds before falling back gracefully.
- CRIS operates at an average workload of ~36 prompt tokens and ~10 completion tokens per dialogue turn.
- Offline benchmarking via `local_evaluator` executed 200 evaluation sessions in approximately 25 minutes (approximately 7.5 seconds per session on average).
- Session 1 exhibited higher initial latency due to lazy model loading and object initialization before reaching steady-state throughput.

## Estimated Model Cost

- **Gemini (`gemini-3.5-flash-lite`)**: Free tier during development.
- **Anthropic (`claude-haiku-4-5`)**: incurs cost only if enabled; disabled by default.

## Limitations & Challenges

- **Symmetric budget scoring:** Under-budget items score as poorly as over-budget items. `price_multiplier()` uses `abs(price - target_price)`, which is symmetric by construction and is a deliberate but unvalidated choice. An item $30 under the target scores identically to one $30 over the target, which may not reflect the user's true intention if they value affordability.
- **No budget hard filter:** Only category has a hard filter. `budget_bounds()` computes `min_price`/`max_price`, but nothing in the current pipeline reads them. Only `target_price` is used, and only as a post-rerank soft adjustment. If the user intended a hard budget cutoff, it is not currently implemented.
- **Regex extraction limitations:** Regex extraction is currently tuned to the local simulator's exact wording. Future development should incorporate more robust intent classification and category extraction from user input to ensure that conversational context is not omitted.
- **Category/brand blind spot:** The simulator's constraint classifier never labels a value as category or brand, so asking about either cannot surface new information. Both attributes are therefore deprioritised during question generation.
- **External LLM dependency:** Intent and slot extraction depends on an external LLM call on every turn. If the call fails, the system degrades to "no new information."
- **Untuned ranking parameters:** RRF fusion weights and the reranker's price-scoring decay rate are currently untuned placeholders. With more time, we would grid-search these parameters for optimal weights.
