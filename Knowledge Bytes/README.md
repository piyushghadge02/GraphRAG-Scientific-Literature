# Knowledge Bytes — GraphRAG for Scientific Literature

## Purpose

This folder explains the **current GraphRAG for Scientific Literature project** in small, progressive Knowledge Bytes.

The structure follows the supplied Knowledge Bytes prompt:
- one concept per byte
- plain-language explanation first
- a small relevant code/config snippet
- what the code is doing
- why it matters
- progressively ordered from project context to implementation, evaluation, UI, runtime, and viva points

## Project in one line

> **PubMed abstracts → semantic chunks → embeddings + biomedical entities → Neo4j graph/vector index → vector retrieval + graph expansion → reranking → LLM answer → evaluation → Streamlit demo**

## Byte order

1. Project purpose and architecture
2. Dataset and 5,000-abstract corpus
3. Chunking strategies
4. Sentence embeddings
5. Semantic clustering and validation
6. Neo4j graph schema
7. Biomedical entity extraction
8. Neo4j vector index
9. Query retrieval
10. Graph expansion
11. Reranking and Phase 5 PageRank
12. LLM generation
13. FastAPI backend
14. Streamlit interface
15. Evaluation with PubMedQA
16. Phase 3 graph ON/OFF comparison
17. Phase 5 query decomposition
18. Runtime, environment variables, and services
19. ngrok hosting
20. Git/GitHub workflow
21. Project limitations and viva answers

## Important implementation note

The explanations describe the current project behavior established during implementation. Small code examples are **representative excerpts** of the relevant module responsibility rather than claims that every line is copied verbatim from the repository.

## Putting the project together

The project first turns scientific abstracts into retrievable units. Those units are represented both semantically (embeddings) and structurally (entities and relationships in Neo4j). A question is answered by combining vector similarity with graph-based context expansion, then giving the selected evidence to an LLM. Evaluation measures whether retrieval and generated answers are good, while Streamlit exposes the pipeline as a usable demo.
