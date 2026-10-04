# Knowledge Bytes — `__init__.py` Files

---

### Byte 1: Package Initializers Define Public Imports
**Builds on:** None — starting point

**In plain terms:**
The `__init__.py` files make package-level imports cleaner by exposing commonly used functions from deeper modules. They are organizational glue rather than part of the retrieval algorithm.

**The code:**
```python
from src.retrieval.retrieve import retrieve
from src.retrieval.vector_search import vector_search
from src.retrieval.graph_expand import expand_via_entities
```

This lets callers use stable package imports instead of knowing every internal file path.

---

### Byte 2: Retrieval Exports the Main Entry Points
**Builds on:** Byte 1

**In plain terms:**
The retrieval package exposes the functions most other parts of the project need: vector search, graph expansion, and the overall `retrieve()` coordinator.

**The code:**
```python
__all__ = [
    "vector_search",
    "expand_via_entities",
    "retrieve",
]
```

This communicates which names are intended as the package's public surface.

---

### Byte 3: Other Packages Use the Same Pattern
**Builds on:** Bytes 1–2

**In plain terms:**
The graph, generation, evaluation, and chunking packages also have initializers. Their purpose is to keep package boundaries clear and provide convenient imports.

**The code:**
```python
# Example pattern
from .some_module import some_public_function

__all__ = ["some_public_function"]
```

The root `src/__init__.py` is similarly lightweight.

---

### PUTTING IT TOGETHER

The initializer files do not introduce new algorithms. Their responsibility is package organization: they expose selected functionality and hide unnecessary implementation paths. When navigating the codebase, treat them as the map of each package's intended public interface.
