# Knowledge Bytes — `src/chunking/`

---

### Byte 1: Chunking Produces the Units RAG Retrieves
**Builds on:** None — starting point

**In plain terms:**
The chunking package turns PubMed abstracts into smaller text units. Those chunks later become Neo4j `Chunk` nodes and the basic retrieval units.

**The code:**
```python
{
    "article_id": article_id,
    "chunk_id": chunk_id,
    "text": text,
    "strategy": strategy,
}
```

The `chunk_id` is especially important because it follows a chunk through graph insertion, retrieval, and evaluation.

---

### Byte 2: The Corpus Loader Creates Stable Article IDs
**Builds on:** Byte 1

**In plain terms:**
`load_data.py` loads the PubMed abstract dataset, samples the working corpus, and writes a local JSONL cache. It creates deterministic IDs from the row and text.

**The code:**
```python
digest = hashlib.sha256(
    f"{index}:{text[:200]}".encode()
).hexdigest()[:12]

return f"pubmed_{index}_{digest}"
```

The stable ID prevents later stages from depending on a temporary dataframe index.

---

### Byte 3: Fixed-Token Chunking Is the Baseline
**Builds on:** Byte 2

**In plain terms:**
The fixed strategy cuts an abstract into 100-token windows. It intentionally ignores semantic boundaries so it can serve as a simple comparison baseline.

**The code:**
```python
tokens = ENCODING.encode(text)

for i in range(0, len(tokens), TOKEN_LIMIT):
    window = tokens[i:i + TOKEN_LIMIT]
```

This can split a sentence or idea between two chunks.

---

### Byte 4: Sentence-Boundary Chunking Preserves Complete Sentences
**Builds on:** Byte 3

**In plain terms:**
The sentence strategy detects sentences and packs complete sentences until the token limit would be exceeded. It avoids the arbitrary cuts of the fixed-token baseline.

**The code:**
```python
if current_sentences and (
    current_tokens + sent_tokens > TOKEN_LIMIT
):
    chunks.append(...)

current_sentences.append(sent)
current_tokens += sent_tokens
```

The result is still size-controlled, but the boundaries are more readable.

---

### Byte 5: Semantic Chunking Embeds Sentences
**Builds on:** Bytes 2 and 4

**In plain terms:**
The semantic strategy first splits an abstract into sentences and turns each sentence into an embedding. Similar sentences can then be grouped by meaning rather than by position.

**The code:**
```python
embeddings = embedder.encode(
    sentences,
    show_progress_bar=False
)
```

The embedding model used for this stage is configured separately from the retrieval embedding provider.

---

### Byte 6: HDBSCAN Groups Related Sentences
**Builds on:** Byte 5

**In plain terms:**
HDBSCAN clusters sentence embeddings into semantic groups. Each group becomes a chunk; noise sentences are retained instead of being discarded.

**The code:**
```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    metric="euclidean"
)
labels = clusterer.fit_predict(embeddings)
```

The emitted chunks preserve the original sentence order.

---

### Byte 7: Short Abstracts Use a Safe Fallback
**Builds on:** Byte 6

**In plain terms:**
Clustering is not useful when an abstract has too few sentences. For fewer than three sentences, the semantic strategy simply returns one chunk.

**The code:**
```python
if len(sentences) < 3:
    return [{
        "article_id": article_id,
        "chunk_id": f"{article_id}_chunk_0",
        "text": " ".join(sentences),
        "strategy": "semantic_cluster",
    }]
```

This prevents the clustering algorithm from making an artificial decision on tiny inputs.

---

### Byte 8: The Chunker Orchestrates Strategies
**Builds on:** Bytes 3–7

**In plain terms:**
`chunker.py` maps strategy names to functions and runs the selected strategy over the corpus. The orchestration layer does not need to know the internals of each algorithm.

**The code:**
```python
STRATEGY_MAP = {
    "fixed_token": fixed_token.chunk,
    "sentence_boundary": sentence_boundary.chunk,
    "semantic_cluster": semantic_cluster.chunk,
}
```

This makes adding another chunking strategy relatively isolated.

---

### Byte 9: Merge-Per-Article Can Collapse the Output
**Builds on:** Byte 8

**In plain terms:**
The optional merge mode combines all chunks produced for an article into one stored chunk. This is useful when the required stored representation is approximately one chunk per abstract.

**The code:**
```python
first["chunk_id"] = f"{article_id}_chunk_0"
first["text"] = " ".join(
    c.get("text", "") for c in chunks
)
return [first]
```

Semantic processing can still happen before this representation-level merge.

---

### Byte 10: Semantic Validation Visualizes the Chunk Embeddings
**Builds on:** Bytes 5–6

**In plain terms:**
`validate_semantic.py` embeds the produced chunks and projects them into two dimensions with UMAP. This is a diagnostic step for seeing whether chunks form meaningful regions.

**The code:**
```python
coords = UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(embeddings)
```

The visualization is validation, not part of runtime retrieval.

---

### PUTTING IT TOGETHER

The chunking package provides three levels of sophistication: arbitrary fixed-token chunks, sentence-aware chunks, and semantic clusters. `chunker.py` chooses the strategy, while the validation script provides a visual sanity check. The resulting JSONL records become the input to the graph-building stage.
