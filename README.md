# GraphRAG for Scientific Literature

Evidence-backed question answering over biomedical research papers. The system chunks scientific abstracts, stores them with embeddings and extracted entities in a Neo4j knowledge graph, retrieves relevant passages with a hybrid vector + graph pipeline, and generates cited answers with an LLM. A Streamlit demo exposes retrieval, evidence, graph visualization, and evaluation results.

Our work extends the required baseline with three identifiable additions: **improved graph/entity handling**, **graph-aware reranking with configurable PageRank**, and **adaptive retrieval**. Every extension is measured through a baseline-vs-modified evaluation — no improvement is claimed without numbers.

## 1. Dataset

- **Corpus:** 5,000 PubMed abstracts (Hugging Face `ccdv/pubmed-summarization`, cached to `data/raw/abstracts_sample.jsonl`). The sample is composed deterministically (seed 42) from papers textually matching PubMedQA contexts plus random fill, so evaluation questions have source articles in-corpus (`python -m src.chunking.load_data --eval-ready`).
- **Evaluation questions:** PubMedQA labeled questions with corpus-verified matches only (`src/evaluation/filter_questions.py` → `data/processed/eval_questions.jsonl`, gold `pubmed_id` = corpus article ID). Measured overlap of the dataset pairing is small (6 verified questions); only verified matches are ever written, never fabricated.

## 2. Architecture

```text
User Query
     |
     v
Query Embedding (all-MiniLM-L6-v2, 384D — or configured model)
     |
     v
Vector Search (Neo4j vector index, cosine)
     |
     v
Adaptive Retrieval
   /         \
  /           \
Vector Only   Graph Expansion (Chunk → Entity → Chunk, depth ≤ 2)
                    |
                    v
           Graph-aware Ranking (α·vector + β·entity + γ·proximity)
                    |
                    v
              Optional PageRank (GDS, retrieved subgraph only)
                    |
                    v
              Final Context (top-K)
                    |
                    v
              LLM Generation (Groq / OpenAI / Ollama)
                    |
                    v
           Evidence-Based Answer (+ citations, scores, graph)
```

Pipeline code: `src/retrieval/retrieve.py` (orchestration), `src/retrieval/adaptive.py` (decision), `src/retrieval/graph_expand.py`, `src/retrieval/fusion.py`, `src/retrieval/gds_rerank.py`, `src/retrieval/retrieve_with_pagerank.py`.

## 3. Semantic chunking

Three strategies in `src/chunking/` (run with `python -m src.chunking.chunker --strategy <name|all>`):

- **fixed_token** — sliding token windows (`CHUNK_TOKEN_LIMIT`, default 100). Used by the legacy in-memory baseline retriever.
- **sentence_boundary** — spaCy sentence splits packed to the token limit.
- **semantic_cluster** (required) — sentences embedded with the PDF-required `all-MiniLM-L6-v2` (`SEMANTIC_CLUSTER_MODEL_NAME`, pinned) and clustered with HDBSCAN (`min_cluster_size=2`); clusters become chunks, noise points become singletons, abstracts with < 3 sentences stay whole.

The graph is built from `data/processed/semantic_cluster.jsonl`.

Submission note: the assignment deliverable is ≈5k chunks, while per-cluster semantic chunking yields several chunks per abstract. The submitted graph therefore uses `python -m src.chunking.chunker --strategy semantic_cluster --merge-per-article`, which runs the full semantic pipeline (embeddings + HDBSCAN) per abstract and then merges each article's chunks into one (`<article_id>_chunk_0`), giving ≈1 chunk/article. Default chunking behavior is unchanged without the flag.

## 4. Knowledge graph (Neo4j)

Required schema, preserved exactly:

```text
(:Article)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)
(:Chunk)-[:SEMANTIC_SIMILAR]->(:Chunk)   (opt-in at build time)
```

- `Chunk` carries `text` plus its embedding (`LIST<FLOAT>`, 384D baseline) under the single `chunk_embedding` vector index (cosine).
- Entities (`{name, type}`) come from SciSpacy `en_core_sci_sm`, normalized (whitespace collapse, possessive stripping, noise filtering) with deterministic first-seen type resolution; duplicates merge by normalized name.
- `SEMANTIC_SIMILAR` edges are computed on demand (`--compute-similarity`, configurable `--similarity-threshold`, default `SEMANTIC_SIMILARITY_THRESHOLD=0.85`) because the pairwise scan is expensive — the relationship type and its MERGE-based insertion are always present.
- Chunk/Article inserts use `MERGE … ON MATCH SET`, so rebuilds refresh stored text and embeddings instead of silently keeping stale vectors.

## 5. Embedding architecture

PDF baseline `all-MiniLM-L6-v2` → 384D, defined once in `src/config.py` (`BASELINE_EMBEDDING_MODEL`) and referenced everywhere through `src/embeddings.py` (cached loader, programmatic dimension detection, validation). Retrieval embeddings follow `EMBEDDING_MODEL_NAME` (Experiment B) without pipeline rewrites; semantic clustering stays pinned to the baseline. Changing the model requires re-embedding plus explicit index recreation (`python -m src.graph.build_graph --recreate-vector-index`); the index is never dropped automatically.

## 6. Retrieval

1. The query is embedded with the configured retrieval model (dimension validated).
2. `vector_search` pulls a 3× candidate pool from the Neo4j vector index.
3. The adaptive decision (Section 8) either keeps the vector path or runs graph expansion through shared entities (depth ≤ 2, per-candidate `shared_entities`, `depth`, `path_count`, query-similarity filtering, configurable caps).
4. Candidates dedupe by `chunk_id` and rank with the fused score (Section 7); each result carries `vector_score`, `shared_entities`, `depth`, `combined_score`, `score_breakdown`, `score_explanation`, and `retrieval_strategy`.
5. Optionally, PageRank over the retrieved subgraph reranks (Section 9).

System presets (`src/evaluation/systems.py`): **A. vector-only**, **B. baseline** (required expansion + legacy fusion), **C. +improved fusion**, **D. +PageRank**, **E. adaptive (ours)**.

## 7. Our retrieval improvements (fusion)

Normalized, explainable 3-term blend (weights rescaled to sum to 1, so no component dominates by scale):

```text
final = α·vector_score + β·entity_score + γ·proximity_score
```

`entity_score` is pool-normalized shared-entity overlap; `proximity_score = 1/(1+depth)`. Weights come from `ALPHA_FUSION_WEIGHT` (0.7), `FUSION_BETA_WEIGHT` (0.3), `FUSION_GAMMA_WEIGHT` (0.0) with per-call overrides. Defaults reproduce the previous ranking exactly; raising gamma makes graph distance count. Every candidate explains itself via `score_explanation`.

## 8. Adaptive retrieval

Instead of always expanding, the pipeline assesses the initial vector hits with deterministic rules (`src/retrieval/adaptive.py`): dominant top-1 with a clear gap → vector-focused path; multi-concept cues ("and", "vs", "relationship between", …) or low confidence → graph expansion; no seeds → nothing to expand. Controlled by `ADAPTIVE_RETRIEVAL_ENABLED`, `ADAPTIVE_TOP1_THRESHOLD` (0.90), `ADAPTIVE_GAP_THRESHOLD` (0.05); the decision, confidence, and reasons are logged and shown in the UI. `expand_graph=False` still forces vector-only.

## 9. PageRank extension

`rerank_with_pagerank` projects **only** the retrieved subgraph (candidate chunks + their entities) into a temporary GDS graph, runs PageRank (`PAGERANK_MAX_ITERATIONS`, `PAGERANK_DAMPING_FACTOR`), min-max normalizes to [0,1], blends as `final = (1−w)·combined + w·pagerank` (`PAGERANK_WEIGHT`, default 0.15), and always drops the projection and temp labels. GDS absence or failure degrades gracefully to the non-PageRank path; `ENABLE_PAGERANK`/`use_pagerank=False` preserves plain retrieval.

## 10. LLM generation

Top-K chunks are formatted by `build_rag_prompt` (numbered context, citation instructions, explicit insufficient-context fallback) and answered by `LLMClient` (Groq with key rotation, OpenAI, or Ollama). Generation code is unchanged apart from serving both evaluation arms.

## 11. Evaluation methodology

Retrieval (article-level, per question, then averaged): **Recall@5**, **Recall@10**, **MRR** (`src/evaluation/retrieval_metrics.py`, preserved). Generation vs PubMedQA long answers: **ROUGE-L F1** and **BERTScore F1** (`distilbert-base-uncased`; `src/evaluation/generation_metrics.py`, preserved).

Runner: `python -m src.evaluation.compare` evaluates every system on the **same** questions (`--systems`, `--num-questions`, `--top-k`) and generates for baseline + modified (`--gen-systems`, `--num-gen-questions`, `--skip-generation`). Outputs in `data/processed/`: `eval_systems.csv` (all systems), `eval_comparison.csv` (Baseline vs Modified + deltas, sidebar-compatible), `eval_generation.csv`, `retrieval_comparison.png`. The legacy `run_full_eval` entry point is untouched. The notebook `notebooks/evaluation_report.ipynb` renders these artifacts with an ablation reading guide and a limitations section — it reports only measured tables.

## 12. Baseline vs modified

- **System A (PDF baseline):** required semantic chunking, `all-MiniLM-L6-v2`, 384D, vector retrieval + required graph expansion, legacy fusion (α=0.7/β=0.3/γ=0.0), no PageRank, no adaptive skipping.
- **System B (ours):** same data and chunking plus configurable embeddings, improved entity handling, 3-term fusion, PageRank, adaptive retrieval.
- Ablation A→E isolates each addition. Results are reported as measured; small deltas on ~200 questions are inconclusive by default.

## 13. Streamlit application

`streamlit run app/main.py`: question input, **System selector (Modified GraphRAG / Baseline)**, toggles for graph expansion / adaptive / PageRank / decomposition, top-K slider, cited answer card, evidence panel (vector, entity, depth/proximity, combined, PageRank, final scores plus per-chunk explanation and strategy banner), interactive knowledge-graph view (Chunk/Entity nodes, MENTIONS + SEMANTIC_SIMILAR edges), and a sidebar with the measured evaluation comparison.

## 14. Installation

Prerequisites: Python 3.10+, Docker, Git. Then:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# SciSpacy weights (match your scispacy version):
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
cp .env.example .env   # fill in Neo4j + LLM credentials
```

## 15. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `password` | Graph connection |
| `LLM_BACKEND` / `LLM_MODEL` / `GROQ_API_KEY1..3` | `groq` / `llama-3.3-70b-versatile` | Answer generation |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Retrieval embeddings (Exp. B to switch) |
| `SEMANTIC_CLUSTER_MODEL_NAME` | `all-MiniLM-L6-v2` | Pinned clustering model |
| `TOP_K_DEFAULT` | `5` | Default retrieval depth |
| `ALPHA_FUSION_WEIGHT` / `FUSION_BETA_WEIGHT` / `FUSION_GAMMA_WEIGHT` | `0.7` / `0.3` / `0.0` | Fusion weights |
| `GRAPH_EXPANSION_ENABLED` / `GRAPH_MAX_DEPTH` / `GRAPH_MAX_EXPANDED_CHUNKS` / `GRAPH_MIN_SHARED_ENTITIES` / `GRAPH_MIN_SIMILARITY` / `GRAPH_MAX_ENTITY_DEGREE` | `true` / `1` / `500` / `2` / `0.3` / `0` (0 = hub guard off) | Expansion behavior |
| `ENABLE_PAGERANK` / `PAGERANK_WEIGHT` / `PAGERANK_MAX_ITERATIONS` / `PAGERANK_DAMPING_FACTOR` | `true` / `0.15` / `20` / `0.85` | PageRank reranking |
| `ADAPTIVE_RETRIEVAL_ENABLED` / `ADAPTIVE_TOP1_THRESHOLD` / `ADAPTIVE_GAP_THRESHOLD` | `true` / `0.90` / `0.05` | Adaptive decision |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.85` | `SEMANTIC_SIMILAR` cutoff |

## 16. Run the pipeline

```bash
docker compose up -d   # or your Neo4j container
python -m src.chunking.chunker --strategy semantic_cluster
python -m src.graph.build_graph --limit 100   # smoke test
python -m src.graph.build_graph               # full load
# optional: --compute-similarity [--similarity-threshold 0.85]
python execute_pipeline.py                    # verify-only checks
```

## 17. Run evaluation

```bash
python -m src.evaluation.filter_questions      # writes eval_questions.jsonl (needs datasets + network)
python -m src.evaluation.compare --num-questions 200 --num-gen-questions 20
python -m src.evaluation.compare --num-questions 20 --num-gen-questions 2 --skip-generation  # quick retrieval-only check
```

## 18. Launch the demo

```bash
streamlit run app/main.py   # http://localhost:8501
```

Unit tests (no services needed): `python tests/test_phase1_embeddings.py`, `python tests/test_phase2_graph.py`, `python tests/test_phase3_eval.py`.

## 19. Limitations

- Article-level binary recall rewards finding the gold article, not the best chunk.
- Generation quality depends on LLM backend/credentials as well as retrieval.
- Without the GDS plugin, PageRank degrades gracefully (system D ≈ C).
- Graph expansion can add noise on narrowly focused queries — the motivation for adaptive retrieval.
- No superiority claim is made beyond measured deltas.

## 20. References

PubMedQA · Neo4j Graph Data Science · Sentence Transformers · scispaCy · Streamlit · PyVis.

## License

Educational and research purposes.
