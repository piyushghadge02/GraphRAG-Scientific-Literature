---
### Byte 1: Phase 2 test scope
**Builds on:** None — starting point

**In plain terms:**
This module tests the graph-aware retrieval layer introduced after the embedding foundation. Its scope covers entity handling, graph expansion, score fusion, PageRank reranking, adaptive retrieval, and compatibility with vector-only retrieval.

**The code:**
```python
fusion = importlib.import_module("src.retrieval.fusion")
adaptive = importlib.import_module("src.retrieval.adaptive")
graph_expand = importlib.import_module("src.retrieval.graph_expand")
gds_rerank = importlib.import_module("src.retrieval.gds_rerank")
retrieve_mod = importlib.import_module("src.retrieval.retrieve")
retrieve_pr = importlib.import_module("src.retrieval.retrieve_with_pagerank")
```

The imports mirror the retrieval pipeline being protected: each major responsibility gets its own test surface.

---
### Byte 2: External graph and NLP dependencies are stubbed
**Builds on:** Byte 1

**In plain terms:**
The graph tests must exercise retrieval behavior without requiring Neo4j, spaCy, or GDS to be installed and running. The module inserts lightweight stubs so importing the production modules does not pull in those live dependencies.

**The code:**
```python
if "neo4j" not in sys.modules:
    stub = types.ModuleType("neo4j")

    class _GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            raise RuntimeError("no live neo4j in tests")

    stub.__dict__["GraphDatabase"] = _GraphDatabase
    sys.modules["neo4j"] = stub
```

The deliberate exception is a guardrail: if a test accidentally tries to use a real Neo4j connection, it fails instead of silently leaving the unit-test boundary.

---
### Byte 3: Fake embeddings provide deterministic retrieval input
**Builds on:** Byte 2

**In plain terms:**
The graph tests also use fake embedders so vector retrieval has predictable inputs. The dimension is fixed at 384, matching the test contract established for the configured baseline.

**The code:**
```python
DIM = 384

class FakeEmbedder:
    def __init__(self, dim=DIM):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim
```

This keeps failures focused on graph/retrieval behavior rather than model output variability.

---
### Byte 4: Entity normalization creates comparable graph labels
**Builds on:** Bytes 1–3

**In plain terms:**
Graph retrieval depends on matching entities consistently. The tests therefore verify that equivalent entity spellings are normalized into a common representation before they participate in graph operations.

**The code:**
```python
# Entity normalization is covered by the phase test suite.
```

Normalization matters because small differences in text representation can otherwise split what should be the same graph entity into separate candidates.

---
### Byte 5: Entity deduplication removes repeated graph evidence
**Builds on:** Byte 4

**In plain terms:**
The graph can surface the same entity more than once. The tests verify that duplicates are removed and that the resulting entity type is resolved deterministically when multiple observations disagree.

**The code:**
```python
# Entity deduplication (+ deterministic type resolution).
```

This makes later graph expansion and scoring stable: the same input should not receive extra weight merely because an entity appeared repeatedly.

---
### Byte 6: Expansion depth is observable
**Builds on:** Byte 5

**In plain terms:**
Graph expansion means starting from retrieved chunks and following entity relationships to discover additional candidates. The tests check that the configured expansion depth is actually reflected in the evidence returned by the expansion layer.

**The code:**
```python
# Graph expansion depth evidence + config defaults.
```

The goal is to test the retrieval contract, not the database engine: expansion should respect the application's configured search boundary.

---
### Byte 7: Duplicate candidates are removed before scoring
**Builds on:** Byte 6

**In plain terms:**
Vector retrieval and graph expansion can produce the same chunk independently. The tests verify that duplicate candidates are collapsed so a chunk does not receive accidental double representation in the ranking pipeline.

**The code:**
```python
# Duplicate candidate removal.
```

This is important because scoring assumes each candidate represents one retrievable chunk, not one copy per route that happened to find it.

---
### Byte 8: Graph candidates receive a three-term fusion score
**Builds on:** Bytes 6–7

**In plain terms:**
The graph-aware ranker combines multiple signals rather than trusting one score. Phase 2 tests the calculation of the three-term fusion used to balance vector relevance with graph-derived evidence.

**The code:**
```python
# Graph/entity score calculation (3-term fusion).
```

Keeping this calculation under test is important because changing a weight or normalization rule can alter the entire ranking order.

---
### Byte 9: Scores are normalized before they are fused
**Builds on:** Byte 8

**In plain terms:**
Different retrieval signals may live on different numeric scales. The tests cover the normalization helpers used by fusion and PageRank so one signal does not dominate simply because its raw numbers are larger.

**The code:**
```python
# Score normalization (fusion + PageRank helper).
```

Normalization makes the combined score interpretable as a deliberate combination of signals rather than an accidental consequence of scale.

---
### Byte 10: Fusion weights are configurable
**Builds on:** Bytes 8–9

**In plain terms:**
The relative importance of fusion signals is configurable, and the tests verify both custom weights and defaults that preserve the legacy behavior. This gives the project room to tune retrieval without rewriting the ranking algorithm.

**The code:**
```python
# Configurable fusion weights (incl. legacy-equivalent defaults).
```

The legacy-equivalent default is especially useful during refactoring: introducing configurability should not silently change the old ranking behavior.

---
### Byte 11: PageRank is normalized before reranking
**Builds on:** Bytes 9–10

**In plain terms:**
PageRank provides graph-centrality information: nodes that are more structurally important within the evaluated graph can receive stronger evidence. The tests verify that PageRank scores are normalized before they participate in ranking.

**The code:**
```python
# PageRank score normalization.
```

The test treats PageRank as one retrieval signal, not as an independent replacement for semantic relevance.

---
### Byte 12: The system has a PageRank-disabled path
**Builds on:** Byte 11

**In plain terms:**
PageRank is an optional enhancement, so retrieval must still work when it is disabled or unavailable. The tests explicitly cover that fallback path.

**The code:**
```python
# PageRank-disabled retrieval path.
```

This protects graceful degradation: missing graph analytics should not make ordinary retrieval unusable.

---
### Byte 13: Adaptive retrieval chooses a strategy
**Builds on:** Bytes 8–12

**In plain terms:**
Adaptive retrieval means the system can decide how much graph work a query needs instead of applying the same retrieval strategy to every query. The tests verify the decision logic and its integration with the retrieval pipeline.

**The code:**
```python
adaptive = importlib.import_module("src.retrieval.adaptive")
```

The important boundary is that the adaptive decision must affect the actual pipeline, not exist as an isolated helper that production retrieval ignores.

---
### Byte 14: Vector-only retrieval remains a valid baseline
**Builds on:** Byte 13

**In plain terms:**
Graph features are additions, not a replacement for the basic vector retriever. The tests keep the vector-only path working so it can serve as a baseline and fallback.

**The code:**
```python
# Vector-only retrieval still works.
```

This also supports later evaluation, where vector-only behavior is compared with progressively richer retrieval systems.

---
### Byte 15: Vector-plus-graph retrieval remains supported
**Builds on:** Bytes 7–14

**In plain terms:**
The final Phase 2 integration check makes sure vector retrieval and graph expansion can work together as one retrieval path. This is the bridge from individual helpers to the complete graph-aware pipeline.

**The code:**
```python
# Vector + graph retrieval still works.
```

A component can pass its unit tests and still be wired incorrectly; this integration coverage catches that class of failure.

PUTTING IT TOGETHER

Phase 2 starts with deterministic test doubles, then protects the graph pipeline from the inside out. Entities are normalized and deduplicated, graph expansion produces controlled candidates, and duplicate candidates are prepared for scoring. Fusion and PageRank add graph signals while normalization and configurable weights keep those signals controlled. Finally, adaptive, vector-only, and vector-plus-graph paths are checked together so the retrieval system remains usable across different configurations.
