# Byte 10: Graph Expansion

**Builds on:** Byte 9

## In plain terms

Graph expansion is the defining GraphRAG step. Starting from retrieved chunks, the system follows shared entities to find additional chunks: Chunk → Entity ← Chunk, with a limited traversal depth.

## The code

```text
MATCH (seed:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Chunk)
WHERE seed.chunk_id IN $seed_ids
RETURN related
```

## What's happening

The seed chunk points to an entity. Other chunks that mention that entity become candidate evidence. The implementation deduplicates candidates and limits traversal depth so the graph does not explode into an unrelated portion of the database.

## Why it matters

Vector similarity finds semantically close text; graph expansion finds connected evidence. The combination is the reason this project is GraphRAG rather than ordinary RAG.
