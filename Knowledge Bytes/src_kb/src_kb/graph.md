# Knowledge Bytes — `src/graph/`

---

### Byte 1: The Graph Represents Articles, Chunks, and Entities
**Builds on:** `chunking.md`

**In plain terms:**
The graph turns the chunked corpus into connected data. Articles contain chunks, chunks mention scientific entities, and chunks can optionally connect through semantic similarity.

**The code:**
```cypher
(:Article {article_id, title, pub_date})
(:Chunk {chunk_id, text, embedding})
(:Entity {name, type})

(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
```

This structure is what later enables graph expansion during retrieval.

---

### Byte 2: Constraints Make Core IDs Unique
**Builds on:** Byte 1

**In plain terms:**
The schema creates uniqueness constraints for article IDs, chunk IDs, and entity names. These identifiers become stable graph keys.

**The code:**
```cypher
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.chunk_id IS UNIQUE
```

This allows later `MERGE` operations to update existing nodes instead of creating duplicates.

---

### Byte 3: The Vector Index Lives on Chunks
**Builds on:** Byte 1

**In plain terms:**
Each `Chunk` stores an embedding, and Neo4j indexes that property for vector similarity search.

**The code:**
```cypher
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

The index dimension must match the embedding provider's contract.

---

### Byte 4: Entity Names Are Normalized
**Builds on:** Byte 1

**In plain terms:**
Scientific entity extraction can produce inconsistent whitespace, punctuation, and casing. Normalization creates a consistent graph key.

**The code:**
```python
text = name.strip().lower()
text = re.sub(r"\s+", " ", text)
text = re.sub(r"'s\b", "", text).strip()
```

The code intentionally avoids aggressive synonym merging; different spellings are not automatically assumed to mean the same entity.

---

### Byte 5: Noise Entities Are Rejected
**Builds on:** Byte 4

**In plain terms:**
Very short, very long, or purely numeric spans are not useful graph entities. They are filtered before insertion.

**The code:**
```python
if not name or len(name) < 2:
    return True
if len(name) > MAX_ENTITY_CHARS:
    return True
if not any(ch.isalpha() for ch in name):
    return True
```

This reduces graph clutter from extraction artifacts.

---

### Byte 6: SciSpacy Converts Text into Entity Nodes
**Builds on:** Bytes 4–5

**In plain terms:**
`entity_extraction.py` runs the scientific NLP model over chunk text and returns normalized `{name, type}` records.

**The code:**
```python
def extract_entities(text: str) -> List[dict]:
    doc = nlp(text)
    return [
        {"name": name, "type": etype}
        for name, etype in _collect_from_doc(doc).items()
    ]
```

This is the bridge from unstructured scientific text to graph structure.

---

### Byte 7: Batch Entity Extraction Uses `nlp.pipe`
**Builds on:** Byte 6

**In plain terms:**
For many chunks, the module uses spaCy's pipeline batching rather than loading each text through the model separately.

**The code:**
```python
for doc, chunk_id in zip(
    nlp.pipe(texts, batch_size=batch_size),
    chunk_ids
):
    ...
```

The result is keyed by `chunk_id`, making it easy to create `MENTIONS` edges.

---

### Byte 8: Chunk Embeddings Are Prepared Before Neo4j Insertion
**Builds on:** `embeddings.md`

**In plain terms:**
`compute_chunk_embeddings()` encodes chunk text using the shared provider and checks the expected dimension.

**The code:**
```python
embeddings = encode_texts(
    texts,
    model_name=model_name,
    batch_size=256
)

if embeddings.shape[1] != dim:
    raise ValueError("Embedding dimension mismatch")
```

The graph layer receives validated vectors rather than raw model output.

---

### Byte 9: Semantic-Similar Edges Are Optional
**Builds on:** Byte 8

**In plain terms:**
The similarity module can create `SEMANTIC_SIMILAR` edges when cosine similarity crosses the configured threshold. It is optional because pairwise comparison can become expensive.

**The code:**
```python
if sim_matrix[i, j] >= threshold:
    triples.append(...)
```

For larger collections the implementation can use FAISS or batched similarity instead of one full matrix.

---

### Byte 10: Batch Insertion Uses `UNWIND`
**Builds on:** Bytes 1–9

**In plain terms:**
The graph is populated in batches rather than issuing one Cypher query per node. `UNWIND` turns a Python list of records into rows inside Cypher.

**The code:**
```cypher
UNWIND $batch AS chk
MERGE (c:Chunk {chunk_id: chk.chunk_id})
ON CREATE SET c.text = chk.text,
              c.embedding = chk.embedding
ON MATCH SET c.text = chk.text,
             c.embedding = chk.embedding
```

This is both faster and easier to rerun.

---

### Byte 11: Writes Retry Transient Failures
**Builds on:** Byte 10

**In plain terms:**
Neo4j/network operations can fail temporarily. The shared retry decorator waits with exponential backoff and retries before giving up.

**The code:**
```python
sleep_time = backoff_in_seconds * (2 ** attempt)
time.sleep(sleep_time)
attempt += 1
```

This is resilience around database operations, not a change to graph semantics.

---

### Byte 12: `build_graph.py` Orchestrates the Whole Build
**Builds on:** Bytes 1–11

**In plain terms:**
The build script coordinates schema creation, chunk loading, embeddings, entity extraction, optional similarity edges, and batch insertion.

**The code:**
```python
with Neo4jConnection() as driver:
    create_constraints_and_indexes(driver)

chunks = load_chunks(args.limit)
articles = derive_articles(chunks)

embeddings = compute_chunk_embeddings(texts)
...
insert_articles(driver, articles)
insert_chunks(driver, chunks)
```

Optional flags control destructive cleanup, dry runs, entity extraction, similarity computation, and vector-index recreation.

---

### PUTTING IT TOGETHER

The graph package takes processed chunks and turns them into a Neo4j knowledge graph. Each chunk carries text and an embedding, entities provide semantic connections between chunks, and optional similarity edges add another graph relationship. `build_graph.py` assembles these pieces into the populated database that retrieval depends on.
