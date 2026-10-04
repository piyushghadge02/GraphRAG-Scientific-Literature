---
### Byte 1: Compliance tests protect deliverable requirements
**Builds on:** None — starting point

**In plain terms:**
This module checks behaviors called out by the project's submission/compliance specification. Instead of testing the entire application, it focuses on a few concrete requirements around sample construction, article merging, graph cleanup, and graph-expansion configuration.

**The code:**
```python
import importlib

load_data = importlib.import_module("src.chunking.load_data")
chunker = importlib.import_module("src.chunking.chunker")
graph_expand = importlib.import_module("src.retrieval.graph_expand")
```

The selected modules correspond directly to the required behaviors being verified.

---
### Byte 2: The compliance suite uses a fake Neo4j driver
**Builds on:** Byte 1

**In plain terms:**
Graph cleanup and expansion need database interactions, but the compliance tests must not depend on a live Neo4j server. The fake driver records what Cypher is requested so the test can inspect the interaction.

**The code:**
```python
class FakeDriver:
    def __init__(self):
        self.seen = []

    def session(self):
        return FakeSession(self)
```

The `seen` list turns a database side effect into testable in-memory evidence.

---
### Byte 3: Database calls are recorded rather than executed
**Builds on:** Byte 2

**In plain terms:**
The fake session captures Cypher statements and parameters. This lets tests verify that the production code issues the required database operation without actually modifying a graph.

**The code:**
```python
def run(self, cypher, **kwargs):
    self.driver.seen.append((cypher, kwargs))
    return []
```

The fake transaction path does the same thing for write operations, keeping database behavior observable and harmless.

---
### Byte 4: Sample construction prioritizes matched papers
**Builds on:** Byte 1

**In plain terms:**
The compliance rules require sample creation to prefer matched papers, remain deterministic, and respect a cap. The test checks that ordering behavior rather than relying on an incidental input order.

**The code:**
```python
def test_matched_prioritized_and_deterministic(self):
    a = load_data.compose_sample([90, 10, 50]
```

The test name captures the contract: matching status affects selection, repeated runs are stable, and the sample stays bounded.

---
### Byte 5: Article chunks are merged into one representative chunk
**Builds on:** Byte 4

**In plain terms:**
The compliance test expects `merge_article_chunks()` to collapse multiple chunks belonging to one article into a single article-level record. It also verifies that the first encountered ID is preserved.

**The code:**
```python
# merge_article_chunks(): one chunk per article, first ID kept.
```

This requirement matters for evaluation and downstream graph construction when the unit of comparison is the article rather than an individual chunk.

---
### Byte 6: Graph cleanup uses one detach-delete operation
**Builds on:** Byte 3

**In plain terms:**
The compliance requirement for `clear_graph()` is a single `DETACH DELETE` operation. The fake driver makes the issued Cypher observable so the test can enforce that exact cleanup behavior.

**The code:**
```python
# clear_graph(): issues a single DETACH DELETE (fake driver).
```

`DETACH DELETE` removes nodes together with their relationships, which makes it suitable for resetting the graph in a controlled operation.

---
### Byte 7: Graph expansion receives the degree-cap configuration
**Builds on:** Byte 6

**In plain terms:**
Graph expansion can be constrained by a degree cap to prevent excessive branching. The compliance test checks that the configured `deg_cap` reaches the expansion function and that the default value of zero means the guard is disabled.

**The code:**
```python
# expand_via_entities(): passes deg_cap
# (0 = guard disabled by default).
```

The test protects configuration plumbing, which is easy to break when function signatures or retrieval settings evolve.

---
### Byte 8: The schema module is loaded without a real Neo4j package
**Builds on:** Bytes 2–3

**In plain terms:**
The compliance suite also imports the graph schema module dynamically after installing a Neo4j stub. This allows schema-related behavior to be tested even in an environment where the real Neo4j Python package is absent.

**The code:**
```python
path = PROJECT_ROOT / "src" / "graph" / "schema.py"
spec = importlib.util.spec_from_file_location(
    "compliance_schema_test", path
)
```

Dynamic loading keeps the test isolated and avoids requiring the normal application runtime environment.

PUTTING IT TOGETHER

The compliance suite is a focused guardrail around requirements that are easy to regress during implementation changes. It uses fake database infrastructure so the tests can inspect graph operations without modifying a real database. Data-sampling and article-merging checks protect the shape of the evaluation corpus, while graph cleanup and expansion checks protect operational requirements. The result is a small set of tests that directly encode important submission constraints.
