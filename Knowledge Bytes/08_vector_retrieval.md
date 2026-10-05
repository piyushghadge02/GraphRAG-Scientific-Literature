# Byte 9: Vector Retrieval

**Builds on:** Byte 8

## In plain terms

When a user asks a question, the question is embedded and the Neo4j vector index is queried for the top-k most similar chunks. The current UI defaults to top-k = 5 in the demonstrated setup.

## The code

```text
def retrieve(query, top_k=5, expand_graph=True):
    query_embedding = embed(query)
    seeds = vector_search(query_embedding, top_k)
    ...
```

## What's happening

The initial result set is the seed set. Each seed carries a similarity score and chunk information. If graph expansion is enabled, those seeds become the starting points for graph traversal.

## Why it matters

Separating seed retrieval from graph expansion makes it possible to compare GraphRAG with graph expansion ON versus OFF, which is required for the Phase 3 manual comparison.
