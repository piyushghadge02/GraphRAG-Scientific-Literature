# Byte 1: Project Architecture

**Builds on:** None — starting point

## In plain terms

The project is a GraphRAG system for answering questions over scientific literature. The key idea is to combine two kinds of retrieval: semantic similarity from embeddings and structural relationships from a Neo4j knowledge graph.

## The code

```text
Question
  ↓
Sentence Transformer embedding
  ↓
Neo4j vector retrieval
  ↓
Top relevant chunks
  ↓
Graph expansion: Chunk → Entity ← Chunk
  ↓
Reranking
  ↓
Retrieved evidence
  ↓
LLM
  ↓
Answer
  ↓
Streamlit UI
```

## What's happening

The Python pipeline prepares the corpus, chunking and graph data. At query time the question is embedded, Neo4j returns relevant chunks, graph expansion can add connected evidence, and the LLM receives the selected context. FastAPI exposes backend functionality and Streamlit provides the interactive demo.

## Why it matters

This is the main mental model for the whole project. If an evaluator asks how the pieces fit together, start with this flow and then zoom into the requested component.
