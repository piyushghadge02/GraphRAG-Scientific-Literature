# Knowledge Bytes — `src/retrieval/`

---

### Byte 1: Retrieval Starts with Vector Search
**Builds on:** `graph.md` Byte 3 and `embeddings.md`

**In plain terms:**
The query is embedded and sent to Neo4j's vector index. The result is a ranked list of chunks that are semantically close to the query.

**The code:**
```cypher
CALL db.index.vector.queryNodes(
    'chunk_embedding',
    $top_k,
    $query_embedding
)
YIELD node, score
RETURN node.chunk_id AS chunk_id,
       node.text AS text,
       score
```

These are the seed chunks for the GraphRAG stage.

---

### Byte 2: The Initial Vector Pool Is Wider Than Final `top_k`
**Builds on:** Byte 1

**In plain terms:**
`retrieve()` initially asks for roughly three times the requested number of chunks. This gives graph expansion and reranking room to add useful candidates.

**The code:**
```python
wider_pool_size = top_k * 3

vector_results = vector_search(
    driver,
    query_embedding,
    top_k=wider_pool_size,
)
```

The final cutoff happens only after graph-aware ranking.

---

### Byte 3: Graph Expansion Follows Shared Entities
**Builds on:** `graph.md` Bytes 1 and 6

**In plain terms:**
A seed chunk's entities become bridges to other chunks. The basic traversal is `Chunk → Entity ← Chunk`.

**The code:**
```cypher
MATCH (seed:Chunk)-[:MENTIONS]->(ent:Entity)
      <-[:MENTIONS]-(candidate:Chunk)
WHERE seed.chunk_id IN $seed_chunk_ids
```

This is the central GraphRAG mechanism: semantic similarity finds seeds, graph structure finds related evidence.

---

### Byte 4: Expansion Can Traverse Up to Depth Two
**Builds on:** Byte 3

**In plain terms:**
The retrieval layer can follow a second chunk/entity hop. This helps answer questions where relevant evidence is connected indirectly.

**The code:**
```cypher
MATCH (seed:Chunk)-[:MENTIONS]->(ent1:Entity)
      <-[:MENTIONS]-(intermediate:Chunk)
MATCH (intermediate)-[:MENTIONS]->(ent2:Entity)
      <-[:MENTIONS]-(candidate:Chunk)
```

The implementation clamps the configured depth to the assignment's maximum.

---

### Byte 5: Graph Candidates Can Be Filtered
**Builds on:** Byte 4

**In plain terms:**
A graph connection is not automatically considered useful. Candidates can require shared entities and can be filtered by similarity to the original query.

**The code:**
```python
similarity = float(np.dot(
    query_arr,
    cand_arr
))

if similarity < similarity_floor:
    continue
```

This limits graph expansion from overwhelming the retrieval pool.

---

### Byte 6: Fusion Combines Multiple Signals
**Builds on:** Bytes 2–5

**In plain terms:**
Fusion turns vector similarity and graph evidence into one score. The individual signals are normalized first so their weights are meaningful.

**The code:**
```python
final = (
    alpha * vector_score
    + beta * entity_score
    + gamma * proximity_score
)
```

The score is designed to reward both semantic relevance and structural support.

---

### Byte 7: Graph Proximity Adds a Depth Signal
**Builds on:** Byte 6

**In plain terms:**
Seed chunks receive the strongest proximity score, while deeper graph candidates receive progressively smaller values.

**The code:**
```python
def proximity_score(depth: int = 0):
    d = max(int(depth), 0)
    return 1.0 / (1.0 + d)
```

So depth 0 is `1.0`, depth 1 is `0.5`, and depth 2 is about `0.33`.

---

### Byte 8: Ranking Keeps a Score Breakdown
**Builds on:** Byte 6

**In plain terms:**
The reranker stores not only the combined score but also the component scores and explanation. This makes ranking decisions inspectable.

**The code:**
```python
candidate["combined_score"] = breakdown["combined"]
candidate["score_breakdown"] = breakdown
candidate["score_explanation"] = breakdown["explanation"]
```

This is useful when debugging why a graph-derived chunk moved up or down.

---

### Byte 9: Adaptive Retrieval Decides Whether Expansion Is Needed
**Builds on:** Bytes 1–8

**In plain terms:**
Adaptive retrieval uses simple signals rather than a learned classifier. A decisive top vector result can skip graph expansion, while multi-concept wording or weaker confidence can trigger it.

**The code:**
```python
if cues:
    return {
        "expand": True,
        "confidence": "multi-concept",
    }

if top1 >= t1 and gap >= gt:
    return {
        "expand": False,
        "confidence": "high",
    }
```

This makes graph work conditional instead of mandatory for every query.

---

### Byte 10: `retrieve()` Is the Main Retrieval Coordinator
**Builds on:** Bytes 1–9

**In plain terms:**
This function ties the retrieval stages together: embed the query, perform vector search, optionally expand the graph, deduplicate candidates, rank them, and return the final top-k.

**The code:**
```python
query_embedding = embedding_model.encode(
    query
).tolist()

vector_results = vector_search(...)
expanded_results = expand_via_entities(...)
ranked = rank_candidates_fused(...)

return ranked[:top_k]
```

Most higher-level retrieval and evaluation code reaches the system through this function.

---

### Byte 11: Query Decomposition Splits Complex Questions
**Builds on:** Byte 10

**In plain terms:**
For complex questions, `decompose_query()` asks the LLM for smaller sub-questions. Each sub-question can then have its own retrieval path.

**The code:**
```python
response = llm_client.generate(
    prompt,
    system_prompt=system_prompt
)
sub_queries = json.loads(response)
```

If parsing fails, the original question is returned as a safe fallback.

---

### Byte 12: Decomposed Results Are Merged by `chunk_id`
**Builds on:** Byte 11

**In plain terms:**
Multiple sub-questions may retrieve the same chunk. The merge step deduplicates by `chunk_id` and retains the higher score.

**The code:**
```python
if chunk_id not in all_results:
    all_results[chunk_id] = chunk
elif chunk["combined_score"] > all_results[chunk_id]["combined_score"]:
    all_results[chunk_id] = chunk
```

This keeps the final context compact.

---

### Byte 13: PageRank Reranking Uses Only the Retrieved Subgraph
**Builds on:** Bytes 10–12

**In plain terms:**
The GDS reranker temporarily projects the retrieved chunks and their entity relationships rather than running PageRank across the whole database.

**The code:**
```cypher
CALL gds.graph.project(
    $projectionName,
    '__PageRankTemp__',
    {
        MENTIONS: {
            orientation: 'UNDIRECTED'
        }
    }
)
```

The score therefore measures structural importance within the current evidence set.

---

### Byte 14: Temporary Graph State Is Always Cleaned Up
**Builds on:** Byte 13

**In plain terms:**
The PageRank code drops the temporary GDS projection and removes temporary labels in a `finally` block. Cleanup happens even when the ranking step fails.

**The code:**
```python
finally:
    session.run(
        "CALL gds.graph.drop($projectionName)"
    )
    session.run(
        "MATCH (n:__PageRankTemp__) "
        "REMOVE n:__PageRankTemp__"
    )
```

This prevents stale temporary graph state from accumulating between requests.

---

### Byte 15: PageRank Is an Additional Reranking Signal
**Builds on:** Byte 13

**In plain terms:**
`retrieve_with_pagerank()` first performs ordinary GraphRAG retrieval, then blends PageRank into the final score. PageRank does not replace semantic retrieval.

**The code:**
```python
final = (
    (1.0 - weight) * combined
    + weight * pr_score
)
```

With PageRank disabled, the final score remains the normal combined score.

---

### Byte 16: Expansion Comparison Is a Debugging Tool
**Builds on:** Byte 10

**In plain terms:**
`compare_expansion.py` runs the same question with graph expansion off and on. It records which chunks changed and can generate answers from both contexts.

**The code:**
```python
chunks_no_expand = retrieve(
    driver, embedding_model,
    query=question,
    top_k=5,
    expand_graph=False
)

chunks_with_expand = retrieve(
    driver, embedding_model,
    query=question,
    top_k=5,
    expand_graph=True
)
```

This directly exposes what graph expansion contributes instead of hiding everything inside one final metric.

---

### PUTTING IT TOGETHER

Retrieval is a funnel: vector search finds semantic seeds, graph traversal adds structurally related chunks, fusion ranks the combined candidate pool, and optional adaptive logic/PageRank changes how aggressively the system uses graph information. Query decomposition adds multiple retrieval paths for complex questions. The final output is a ranked set of evidence chunks that the generation layer can safely turn into an answer.
