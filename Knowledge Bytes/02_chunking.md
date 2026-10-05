# Byte 3: Chunking Strategies

**Builds on:** Byte 2

## In plain terms

A scientific abstract is too large to treat as one retrieval unit. The project therefore creates chunks. The assignment asks for three approaches: fixed-token baseline, sentence-boundary chunking, and semantic clustering.

## The code

```text
# Conceptual dispatcher
fixed_token(text)
sentence_boundary(text)
semantic_cluster(sentences)

# Stored chunk shape
{
  "article_id": "...",
  "chunk_id": "...",
  "text": "...",
  "strategy": "..."
}
```

## What's happening

The fixed-token method provides a simple baseline. Sentence-boundary chunking keeps sentence boundaries intact while respecting a size limit. Semantic clustering uses sentence embeddings and groups semantically related sentences into chunks.

## Why it matters

Chunking controls what the retriever can find. Poor chunks can mix unrelated topics or split important context, which can reduce retrieval quality even if the vector model is good.
