# Knowledge Bytes — `src/embeddings.py`

---

### Byte 1: Embeddings Are Centralized
**Builds on:** `config.md` Byte 3

**In plain terms:**
This module is the single embedding provider for documents and queries. It keeps chunk indexing and query retrieval on the same model/configuration.

**The code:**
```python
_MODEL_CACHE: Dict[str, object] = {}

def get_embedder(model_name=None):
    name = model_name or get_retrieval_model_name()
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]
```

The rest of the system does not need to know how the SentenceTransformer is loaded.

---

### Byte 2: Model Instances Are Cached
**Builds on:** Byte 1

**In plain terms:**
Loading a transformer model repeatedly is expensive. The cache keeps one loaded instance per model name and reuses it.

**The code:**
```python
if name not in _MODEL_CACHE:
    _MODEL_CACHE[name] = SentenceTransformer(name)

return _MODEL_CACHE[name]
```

`clear_embedder_cache()` exists for controlled resets and tests.

---

### Byte 3: Text and Query Encoding Share One Path
**Builds on:** Byte 1

**In plain terms:**
`encode_texts()` handles batches of texts, while `encode_query()` handles one query. Both ultimately use the same model-loading mechanism.

**The code:**
```python
embeddings = model.encode(texts, ...)
result = np.array(
    embeddings,
    dtype=np.float32
)
```

This prevents indexing and retrieval from accidentally using different embedding behavior.

---

### Byte 4: Dimensions Are Validated
**Builds on:** Bytes 1–3

**In plain terms:**
The module checks the vector length before the vector reaches Neo4j. A mismatch becomes an explicit error instead of a confusing database/index failure later.

**The code:**
```python
observed = len(embedding)

if observed != expected:
    raise ValueError(
        "Embedding dimension mismatch ..."
    )
```

For this project, the baseline contract is 384 dimensions.

---

### PUTTING IT TOGETHER

The embedding module is the bridge between text and vector search. It loads the configured model once, encodes text consistently, and validates the expected dimensionality. Graph indexing and query retrieval therefore share one controlled embedding implementation.
