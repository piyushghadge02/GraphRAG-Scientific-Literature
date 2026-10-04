# `check_graph_scores.py` — Knowledge Bytes

This module is a small diagnostic script for inspecting how GraphRAG candidates are ranked after combining vector similarity with graph-based entity expansion.

---

### Byte 1: Connect the diagnostic script to Neo4j and the embedding model
**Builds on:** None — starting point

**In plain terms:**
The script starts by loading the project configuration and creating two core dependencies: a Neo4j driver for graph access and the configured SentenceTransformer model for query embeddings. This keeps the diagnostic aligned with the same settings used by the main GraphRAG system.

**The code:**
```python
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from src.config import settings

uri = settings.NEO4J_URI
user = settings.NEO4J_USER
password = settings.NEO4J_PASSWORD
driver = GraphDatabase.driver(uri, auth=(user, password))

embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
```

---

### Byte 2: Define the test question
**Builds on:** Byte 1

**In plain terms:**
A fixed PubMed-style question is used as the diagnostic input. Using one known question makes it possible to inspect exactly how vector retrieval, graph expansion, and score fusion behave.

**The code:**
```python
query = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
```

---

### Byte 3: Reuse the project's retrieval and fusion functions
**Builds on:** Byte 2

**In plain terms:**
Instead of implementing retrieval logic again, the script imports the project's existing vector search, graph expansion, and score-combination functions. This makes the experiment a check of the real pipeline rather than a separate approximation.

**The code:**
```python
from src.retrieval.vector_search import vector_search
from src.retrieval.graph_expand import expand_via_entities
from src.retrieval.fusion import combined_score
import numpy as np
```

---

### Byte 4: Convert the question into an embedding
**Builds on:** Byte 3

**In plain terms:**
The question is converted into a numeric embedding using the configured SentenceTransformer model. That vector is what the vector-search stage uses to measure semantic similarity against stored chunk embeddings.

**The code:**
```python
query_embedding = embedding_model.encode(query).tolist()
```

---

### Byte 5: Retrieve the initial vector candidates
**Builds on:** Byte 4

**In plain terms:**
The script asks the existing vector-search implementation for the top 15 chunks. These results become the initial retrieval seeds: chunks that are directly similar to the question before graph information is added.

**The code:**
```python
vector_results = vector_search(driver, query_embedding, top_k=15)
vector_chunk_ids = [r["chunk_id"] for r in vector_results]
```

---

### Byte 6: Expand candidates through graph entities
**Builds on:** Byte 5

**In plain terms:**
The seed chunks are expanded through their linked entities to a graph depth of one. This introduces chunks that may not have been among the strongest vector matches but are connected through shared graph entities.

**The code:**
```python
expanded_results = expand_via_entities(
    driver,
    vector_chunk_ids,
    max_depth=1
)
```

---

### Byte 7: Create a unified candidate table
**Builds on:** Byte 6

**In plain terms:**
Vector results and graph-expanded results need one common representation before they can be ranked together. The `candidates` dictionary uses `chunk_id` as the unique key and stores whether a chunk was a vector seed, its vector score, and its number of shared entities.

**The code:**
```python
candidates = {}
for r in vector_results:
    cid = r["chunk_id"]
    candidates[cid] = {
        "chunk_id": cid,
        "is_vector_seed": True,
        "vector_score": r["score"],
        "shared_entities": 0
    }
```

---

### Byte 8: Normalize the query vector
**Builds on:** Byte 7

**In plain terms:**
For graph-expanded chunks that need a vector score calculated locally, the script normalizes the query embedding first. Normalization makes the later dot product behave like cosine similarity when the candidate vector is normalized as well.

**The code:**
```python
query_arr = np.array(query_embedding, dtype=np.float32)
norm_q = np.linalg.norm(query_arr)
if norm_q > 0:
    query_arr = query_arr / norm_q
```

---

### Byte 9: Merge graph information into existing candidates
**Builds on:** Byte 8

**In plain terms:**
If graph expansion finds a chunk that was already retrieved by vector search, the script keeps the existing candidate and adds its `shared_entities` count. This avoids creating duplicate candidates while preserving both retrieval signals.

**The code:**
```python
for r in expanded_results:
    cid = r["chunk_id"]
    shared = r["shared_entities"]

    if cid in candidates:
        candidates[cid]["shared_entities"] = shared
```

---

### Byte 10: Score newly discovered graph candidates
**Builds on:** Byte 9

**In plain terms:**
A graph-expanded chunk that was not an original vector seed still needs a vector-similarity value for fusion. If its embedding is available, the script normalizes it and computes a dot product with the normalized query; otherwise its vector score remains `0.0`.

**The code:**
```python
cand_emb = r.get("embedding")
vector_score = 0.0
if cand_emb:
    cand_arr = np.array(cand_emb, dtype=np.float32)
    norm_c = np.linalg.norm(cand_arr)
    if norm_c > 0:
        cand_arr = cand_arr / norm_c
    vector_score = float(np.dot(query_arr, cand_arr))
```

---

### Byte 11: Store newly expanded candidates
**Builds on:** Byte 10

**In plain terms:**
After its optional vector score is computed, a newly discovered chunk is inserted into the same candidate dictionary. It is marked as not being a vector seed, which later helps the diagnostic output show where each result originated.

**The code:**
```python
candidates[cid] = {
    "chunk_id": cid,
    "is_vector_seed": False,
    "vector_score": vector_score,
    "shared_entities": shared
}
```

---

### Byte 12: Find the graph-signal scale
**Builds on:** Byte 11

**In plain terms:**
The fusion function needs the largest shared-entity count so that the graph signal can be normalized relative to the candidates in this run. The `default=0` protects the script when there are no candidates.

**The code:**
```python
max_shared_entities = max(
    (c["shared_entities"] for c in candidates.values()),
    default=0
)
```

---

### Byte 13: Test multiple fusion weights
**Builds on:** Byte 12

**In plain terms:**
The script deliberately evaluates three alpha values instead of assuming one weight is correct. In the combined score, alpha controls the balance between vector similarity and the graph-based shared-entity signal.

**The code:**
```python
for alpha in [0.6, 0.5, 0.4]:
    print(f"\n=== RANKING WITH ALPHA = {alpha} ===")
```

---

### Byte 14: Calculate the combined ranking score
**Builds on:** Byte 13

**In plain terms:**
Each candidate receives a fused score from its vector score and shared-entity count. The `max(..., 1)` safeguard prevents the normalization argument from becoming zero when no graph signal is present.

**The code:**
```python
c["final_combined_score"] = combined_score(
    vector_score=c["vector_score"],
    shared_entities=c["shared_entities"],
    max_shared_entities=max(max_shared_entities, 1),
    alpha=alpha
)
```

---

### Byte 15: Rank and inspect the top five candidates
**Builds on:** Byte 14

**In plain terms:**
Candidates are sorted from highest to lowest combined score, and only the top five are printed. The diagnostic output includes the final score, original vector score, shared-entity count, and whether the chunk was one of the original vector seeds.

**The code:**
```python
ranked = sorted(
    candidates.values(),
    key=lambda x: x["final_combined_score"],
    reverse=True
)

for idx, r in enumerate(ranked[:5]):
    is_seed = r["chunk_id"] in vector_chunk_ids
    print(
        f"  {idx+1}. {r['chunk_id']} "
        f"(is_seed={is_seed}, "
        f"combined={r['final_combined_score']:.4f}, "
        f"vector={r['vector_score']:.4f}, "
        f"shared_ent={r['shared_entities']})"
    )
```

---

### Byte 16: Close the Neo4j connection
**Builds on:** Byte 15

**In plain terms:**
The script explicitly closes the Neo4j driver when the diagnostic work is finished. This releases the database resources instead of leaving the connection open after the script exits.

**The code:**
```python
driver.close()
```

---

## PUTTING IT TOGETHER

This scratch script is a focused diagnostic for the GraphRAG retrieval-ranking path. It starts with vector retrieval, expands those seeds through graph entities, merges both sources into one candidate set, and computes missing vector scores for newly discovered chunks. It then varies the fusion weight (`alpha`) to show how the ranking changes when the balance between semantic similarity and graph connectivity changes. Finally, it prints the top candidates with enough detail to inspect whether graph expansion is affecting the ranking as intended and closes the database connection.
