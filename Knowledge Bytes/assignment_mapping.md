# Assignment → Implementation Map

| Assignment requirement | Current project area |
|---|---|
| 5,000 PubMed abstracts | `data/raw/abstracts_sample.jsonl` + data loading |
| Fixed-token baseline | `src/chunking/fixed_token.py` |
| Sentence-boundary chunking | `src/chunking/sentence_boundary.py` |
| Semantic clustering | `src/chunking/semantic_cluster.py` |
| Chunk orchestration | `src/chunking/chunker.py` |
| MiniLM embeddings | Sentence Transformers usage |
| SciSpacy entities | `en_core_sci_sm` + graph insertion |
| Article / Chunk / Entity graph | `src/graph/*` |
| Neo4j batch insertion | `src/graph/batch_insert.py` |
| Graph construction | `src/graph/build_graph.py` |
| Vector index | Neo4j `chunk_embedding` |
| Graph-enhanced retrieval | retrieval/RAG pipeline |
| LLM generation | configured Groq backend |
| Phase 3 comparison | evaluation/test workflow |
| Recall@5 / Recall@10 / MRR | `src/evaluation/retrieval_metrics.py` |
| ROUGE-L / BERTScore | `src/evaluation/generation_metrics.py` |
| PubMedQA filtering | `src/evaluation/filter_questions.py` |
| Evaluation runner | `src/evaluation/run_eval.py` |
| Streamlit UI | `app.py` + `app/components/*` |
| FastAPI backend | `api/main.py` |
| PageRank / graph reranking | Phase 5 retrieval/UI settings |
| Public demo | ngrok → Streamlit 8502 |

## Important

This mapping is a learning guide, not a claim that every requirement is complete merely because a file exists. Final submission should be checked against actual test outputs and evaluation metrics.
