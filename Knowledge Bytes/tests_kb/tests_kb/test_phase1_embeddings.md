---
### Byte 1: Phase 1 test scope
**Builds on:** None — starting point

**In plain terms:**
This module protects the embedding and retrieval foundation. It checks that the configured embedding model and dimensions remain consistent while ensuring retrieval and fusion still work when the embedding configuration changes.

**The code:**
```python
BASELINE_MODEL = BASELINE_EMBEDDING_MODEL
BASELINE_DIM = 384
CUSTOM_DIM = 768
```

The constants make the intended baseline explicit and give the tests a second dimension to verify configurability rather than assuming one fixed vector size everywhere.

---
### Byte 2: Fake embeddings replace real model downloads
**Builds on:** Byte 1

**In plain terms:**
The tests use a tiny fake SentenceTransformer-like object instead of downloading or loading a real model. This isolates application logic from model availability and makes the suite fast and deterministic.

**The code:**
```python
class FakeEmbedder:
    def __init__(self, dim: int):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)
```

The fake only needs the two behaviors the production code depends on: reporting its dimension and turning text into vectors of that dimension.

---
### Byte 3: A mismatched embedder is an explicit test case
**Builds on:** Byte 2

**In plain terms:**
A second fake embedder can intentionally use a different vector dimension. This lets the tests verify that the system notices configuration changes instead of silently accepting incompatible vectors.

**The code:**
```python
class MismatchedEmbedder(FakeEmbedder):
    ...
```

The important idea is not the subclass itself; it is the ability to manufacture a controlled mismatch and test the application's dimension checks.

---
### Byte 4: Imports target modules without triggering package-name collisions
**Builds on:** Byte 1

**In plain terms:**
The module imports retrieval submodules with `importlib.import_module`. The test explains that this avoids a collision caused by `src.retrieval.__init__` re-exporting function names that can shadow submodule objects.

**The code:**
```python
fusion = importlib.import_module("src.retrieval.fusion")
vector_search = importlib.import_module("src.retrieval.vector_search")
retrieve_mod = importlib.import_module("src.retrieval.retrieve")
```

This is a testing-specific import pattern: the tests need the actual module objects so they can patch or call internal behavior precisely.

---
### Byte 5: Retrieval tests stay independent of Neo4j
**Builds on:** Byte 2

**In plain terms:**
The phase tests are designed to run without a live Neo4j instance. Fake drivers and sessions stand in for database behavior, allowing retrieval logic to be tested without infrastructure.

**The code:**
```python
# No model download, no Neo4j.
class FakeEmbedder:
    ...
```

Together with the fake database objects used by the retrieval tests, this keeps the test boundary around application logic rather than external services.

---
### Byte 6: The embedding model is treated as configuration
**Builds on:** Bytes 1–5

**In plain terms:**
The tests verify that the embedding model is not hard-wired throughout the retrieval stack. They check both the baseline model and the ability to change the configured model while preserving the expected retrieval contract.

**The code:**
```python
from src.config import BASELINE_EMBEDDING_MODEL, settings

BASELINE_MODEL = BASELINE_EMBEDDING_MODEL
```

The configuration module becomes the source of truth, while the tests make sure downstream code respects it.

---
### Byte 7: Vector dimensions are part of the retrieval contract
**Builds on:** Bytes 3 and 6

**In plain terms:**
An embedding is only useful to the vector index if its dimensionality matches what the system expects. The tests therefore cover generated chunk embeddings, query embeddings, and custom model dimensions.

**The code:**
```python
self.assertEqual(
    embeddings.get_embedding_dimension(),
    expected_dim,
)
```

The exact assertions vary across the file, but the protected invariant is the same: vectors produced by the configured model must have the dimension the retrieval layer is prepared to consume.

---
### Byte 8: Fusion remains backward compatible
**Builds on:** Bytes 6–7

**In plain terms:**
Changing embedding configuration should not accidentally break the score-fusion and ranking layer. The tests keep fusion covered so the embedding refactor remains compatible with the existing retrieval behavior.

**The code:**
```python
fusion = importlib.import_module("src.retrieval.fusion")
```

This test boundary connects the new embedding configuration to an existing retrieval component rather than testing embeddings in isolation only.

---
### Byte 9: Phase 1 protects against stale hard-coded assumptions
**Builds on:** Bytes 6–8

**In plain terms:**
The final concern is architectural: old assumptions about the original vector dimension should not remain hidden in unrelated code. The test suite is intended to catch those stale assumptions while preserving the normal retrieval path.

**The code:**
```python
# No hard-coded old dimension assumptions remain where they should not.
```

This is less about one function and more about preventing a configuration change from creating subtle failures elsewhere.

PUTTING IT TOGETHER

Phase 1 establishes a testable contract around embeddings: the model is configurable, dimensions are checked, and retrieval continues to operate with the configured vectors. Fake embedders and drivers keep the suite independent of external services. The module also protects compatibility between the embedding layer and score-fusion/retrieval code. That foundation is what later graph-aware retrieval tests build on.
