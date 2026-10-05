# Byte 8: Neo4j Vector Index

**Builds on:** Byte 6

## In plain terms

Neo4j stores the 384-dimensional chunk embeddings and exposes a vector index so a query embedding can retrieve the nearest chunks by cosine similarity.

## The code

```text
CREATE VECTOR INDEX chunk_embedding
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

## What's happening

The index is built over Chunk.embedding. A query is encoded into the same 384-dimensional space and the index returns the most similar chunks.

## Why it matters

This is the first retrieval stage. It gives GraphRAG a strong semantic starting point before graph traversal adds structurally related evidence.
