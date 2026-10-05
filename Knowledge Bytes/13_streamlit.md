# Byte 14: Streamlit Interactive UI

**Builds on:** Byte 13

## In plain terms

Streamlit is the presentation layer. It lets a user enter a scientific question, adjust retrieval settings, start a search, and inspect the answer, evidence and knowledge graph.

## The code

```text
python -m streamlit run app.py --server.port 8502

# Main UI flow
question
  → Search Literature
  → API request
  → answer / evidence / graph
```

## What's happening

The current interface exposes Graph Expansion, Adaptive Retrieval, PageRank Reranking, Query Decomposition and Top-K controls. Results include the generated answer and a Knowledge Graph view.

## Why it matters

The assignment requires an interactive demo. Streamlit turns the backend pipeline into something an evaluator can actually use instead of only reading source code.
