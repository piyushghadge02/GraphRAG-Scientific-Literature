# Knowledge Bytes — `src/config.py`

---

### Byte 1: Configuration Is the Shared Control Panel
**Builds on:** None — starting point

**In plain terms:**
`config.py` centralizes environment settings and project-wide constants. Other modules import these values instead of hard-coding Neo4j connection details, model names, retrieval limits, and scoring settings.

**The code:**
```python
BASELINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@dataclass(frozen=True)
class Config:
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
```

The important idea is one source of truth. Changing configuration should not require editing many modules.

---

### Byte 2: Environment Variables Override Defaults
**Builds on:** Byte 1

**In plain terms:**
Every important runtime setting has a safe local default but can be overridden through environment variables or `.env`.

**The code:**
```python
NEO4J_PASSWORD: str = os.getenv(
    "NEO4J_PASSWORD", "password"
)
TOP_K_DEFAULT: int = int(
    os.getenv("TOP_K_DEFAULT", "5")
)
```

This lets the same code run locally, in CI, or against another Neo4j/LLM setup without changing source code.

---

### Byte 3: The Embedding Model and Dimension Are a Contract
**Builds on:** Byte 1

**In plain terms:**
The config defines the baseline embedding model and its expected 384-dimensional output. This matters because Neo4j's vector index must use the same dimensionality as the stored embeddings.

**The code:**
```python
BASELINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BASELINE_EMBEDDING_DIM = 384
VECTOR_INDEX_NAME = "chunk_embedding"
```

If the embedding model changes, stored vectors and the vector index need to be reconsidered together.

---

### Byte 4: Retrieval Behavior Is Configurable
**Builds on:** Bytes 1–3

**In plain terms:**
Graph depth, similarity thresholds, fusion weights, adaptive retrieval, and PageRank settings live in configuration. This makes experiments possible without rewriting retrieval algorithms.

**The code:**
```python
GRAPH_EXPANSION_DEPTH = int(os.getenv(
    "GRAPH_EXPANSION_DEPTH", "2"
))
FUSION_ALPHA = float(os.getenv(
    "FUSION_ALPHA", "0.7"
))
FUSION_BETA = float(os.getenv(
    "FUSION_BETA", "0.3"
))
```

The retrieval code consumes these values; it does not decide them independently.

---

### PUTTING IT TOGETHER

`config.py` defines the runtime contract used by the rest of `src`. The embedding model/dimension, Neo4j connection, retrieval defaults, graph limits, and ranking parameters are all centralized here. This makes the pipeline easier to reproduce and easier to tune.
