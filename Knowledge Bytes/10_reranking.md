# Byte 11: Reranking and PageRank

**Builds on:** Byte 10

## In plain terms

After retrieval and graph expansion, there may be more candidates than the final context can use. The system therefore reranks them using retrieval relevance and graph information. Phase 5 adds graph centrality such as PageRank or Betweenness as an additional signal.

## The code

```text
# Conceptual score
final_score =
    alpha * vector_similarity   + beta  * graph_score   + gamma * centrality_score
```

## What's happening

The exact weights can be exposed as retrieval settings. In the demonstrated UI, fusion weights are shown to the user. PageRank can be calculated on the retrieved subgraph using Neo4j GDS and then used to prioritize structurally important chunks.

## Why it matters

Reranking prevents graph expansion from blindly returning every connected chunk. It turns a larger candidate set into a focused context for the LLM.
