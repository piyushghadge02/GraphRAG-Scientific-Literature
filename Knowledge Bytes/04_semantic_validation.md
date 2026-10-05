# Byte 5: Semantic Clustering and Validation

**Builds on:** Byte 3

## In plain terms

Semantic chunking first represents sentences as vectors, then groups similar vectors with a clustering algorithm such as HDBSCAN or Agglomerative Clustering. t-SNE or UMAP can reduce embeddings to two dimensions for visual inspection.

## The code

```text
sentences
  ↓
MiniLM embeddings
  ↓
HDBSCAN / Agglomerative
  ↓
semantic clusters
  ↓
t-SNE / UMAP
  ↓
2-D validation plot
```

## What's happening

The cluster labels are used to assemble semantically related sentences into chunks. A 2-D plot is a validation aid: nearby points should show meaningful grouping rather than arbitrary mixing.

## Why it matters

The assignment is not only asking for code that runs; it asks for evidence that semantic chunks are meaningful. Visualization gives a human-readable sanity check.
