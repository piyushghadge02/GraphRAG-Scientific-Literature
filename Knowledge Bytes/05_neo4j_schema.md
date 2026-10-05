# Byte 6: Neo4j Graph Schema

**Builds on:** Byte 5

## In plain terms

Neo4j stores the literature as nodes and relationships. The core node types are Article, Chunk and Entity. Relationships connect an article to its chunks and chunks to the scientific entities they mention.

## The code

```text
(:Article {article_id, title, pub_date})
(:Chunk {chunk_id, text, embedding})
(:Entity {name, type})

(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Chunk)-[:SEMANTIC_SIMILAR]->(:Chunk)
```

## What's happening

The graph builder inserts the article and chunk nodes, then creates HAS_CHUNK relationships. Extracted entities become Entity nodes and MENTIONS relationships connect chunks to them. The current rebuild verified 5,000 Article nodes, 5,000 Chunk nodes, 5,000 HAS_CHUNK relationships, 73,072 Entity nodes and 227,151 MENTIONS relationships.

## Why it matters

The graph gives the system structure that a plain vector store does not have. It lets the retriever move from a relevant chunk to shared scientific concepts and then to related chunks.
