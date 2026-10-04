---
### Byte 1: Phase 3 test scope
**Builds on:** None — starting point

**In plain terms:**
This module tests the evaluation and product-facing layer of GraphRAG. It covers retrieval metrics, generation evaluation, system comparison, preset resolution, UI metadata, and dispatch between ordinary and PageRank retrieval.

**The code:**
```python
from src.evaluation import retrieval_metrics as rm
from src.evaluation import systems
from src.evaluation import compare
```

These imports show the three responsibilities being tested: measuring results, defining systems, and orchestrating comparisons.

---
### Byte 2: Fake embeddings keep evaluation tests deterministic
**Builds on:** Byte 1

**In plain terms:**
Evaluation tests should measure evaluation logic, not the behavior of a real embedding model. A fixed fake embedder provides predictable 384-dimensional vectors without downloading a model.

**The code:**
```python
class FakeEmbedder:
    def __init__(self, dim=DIM):
        self._dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)
```

This lets the tests construct controlled retrieval scenarios and focus assertions on the evaluation layer.

---
### Byte 3: Fake Neo4j sessions return controlled retrieval data
**Builds on:** Byte 2

**In plain terms:**
The evaluation pipeline expects database sessions that return vector hits and graph-expansion results. The fake session routes different Cypher shapes to predefined records instead of contacting a real database.

**The code:**
```python
def run(self, cypher, **kwargs):
    if "db.index.vector" in cypher:
        return list(self.driver.vector_records)
    if "COUNT(DISTINCT e)" in cypher:
        return _Single({"count": 2})
    return list(self.driver.expansion_records)
```

This makes it possible to build repeatable retrieval cases for baseline and modified systems.

---
### Byte 4: Retrieval candidates have a minimal test shape
**Builds on:** Byte 3

**In plain terms:**
The tests represent a retrieved chunk with an identifier, text, and score. Keeping this fixture small makes ranking and metric tests easy to understand.

**The code:**
```python
def _vec(score, cid):
    return {"chunk_id": cid, "text": f"text {cid}", "score": score}
```

The helper creates just enough data for the evaluation pipeline to rank candidates and identify the expected article/chunk.

---
### Byte 5: Retrieval metrics are tested independently
**Builds on:** Byte 4

**In plain terms:**
The module directly tests retrieval metrics such as Recall@K and Mean Reciprocal Rank (MRR). Recall asks whether the expected article appears in the top K; MRR also cares about how high that first correct result appears.

**The code:**
```python
# Metric calculations
# Recall@K, MRR, ROUGE-L
```

Testing the metric functions separately prevents a retrieval bug from being confused with a metric-calculation bug.

---
### Byte 6: Generation quality uses ROUGE-L
**Builds on:** Byte 5

**In plain terms:**
For generated answers, the tests include ROUGE-L, a text-overlap metric based on the longest common subsequence. It provides a separate signal from retrieval recall.

**The code:**
```python
# Evaluation metric calculations (Recall@K, MRR, ROUGE-L).
```

The test suite therefore keeps retrieval quality and answer-generation quality as distinct measurements.

---
### Byte 7: Baseline and modified retrieval paths are both evaluated
**Builds on:** Bytes 5–6

**In plain terms:**
Phase 3 compares different retrieval configurations rather than testing only one pipeline. The tests cover both the baseline retrieval path and the modified graph-aware path.

**The code:**
```python
# Baseline retrieval evaluation path.
# Modified retrieval evaluation path.
```

This enables controlled comparisons where the same evaluation machinery can inspect multiple system configurations.

---
### Byte 8: Comparison builders produce structured evaluation output
**Builds on:** Byte 7

**In plain terms:**
After individual systems are evaluated, the comparison layer assembles their results into structured outputs. The tests protect those output-building functions so downstream reports can rely on a stable format.

**The code:**
```python
from src.evaluation import compare
```

The important boundary is between running a system and turning its measurements into comparable evaluation records.

---
### Byte 9: System presets provide named evaluation configurations
**Builds on:** Byte 8

**In plain terms:**
A system preset is a named configuration describing how retrieval should run. The tests verify preset resolution, including the mapping used by the UI.

**The code:**
```python
from src.evaluation import systems
```

This keeps the UI and evaluation runner aligned on what a label such as a retrieval mode actually means.

---
### Byte 10: Retrieval metadata is exposed to the UI
**Builds on:** Byte 9

**In plain terms:**
The product layer needs more than the final answer: it can also receive a strategy label and retrieval breakdowns explaining how the result was produced. The tests verify that this metadata survives the retrieval path.

**The code:**
```python
# Retrieval metadata exposed to the UI
# (strategy label, breakdowns).
```

This creates a contract between backend retrieval behavior and the interface that displays it.

---
### Byte 11: System dispatch selects the correct retriever
**Builds on:** Bytes 7–10

**In plain terms:**
The evaluation runner must route a requested system to the correct implementation. The tests specifically distinguish PageRank retrieval from ordinary retrieval.

**The code:**
```python
# run_system dispatch (pagerank vs plain retrieve).
```

This protects the integration point where a configuration choice becomes an actual execution path.

---
### Byte 12: Phase 3 stays independent of live infrastructure
**Builds on:** Bytes 2–3

**In plain terms:**
The module explicitly avoids requiring live Neo4j, an LLM, or GDS for its unit tests. Fake drivers, fake embedders, and a stub BERT scorer keep the evaluation mechanics isolated.

**The code:**
```python
# No live Neo4j/LLM/GDS required:
# fake drivers, fake embedders and a stub BERT scorer
# are used throughout.
```

This makes the tests suitable for repeatable local or automated execution.

PUTTING IT TOGETHER

Phase 3 turns retrieval behavior into measurable, comparable evidence. Fake infrastructure creates controlled candidate sets, metric functions measure retrieval and generation separately, and comparison helpers organize those measurements across system presets. The UI metadata and dispatch tests then verify that evaluation-oriented retrieval choices remain consistent with product behavior. This gives the project a protected path from retrieval configuration to measurable output.
