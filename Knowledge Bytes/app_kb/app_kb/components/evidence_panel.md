# Knowledge Bytes — `app/components/evidence_panel.py`

---

### Byte 1: The Evidence Panel Displays Retrieved Chunks
**Builds on:** `main.md` Byte 17

**In plain terms:**
`render_evidence_panel()` shows the chunks that the retrieval system supplied to generation. It is a transparency component: the user can inspect the evidence behind the answer.

**The code:**
```python
def render_evidence_panel(chunks: List[Dict]):
    st.markdown("### Evidence")
```

The component expects retrieval dictionaries rather than raw text alone.

---

### Byte 2: Each Chunk Has Retrieval Metadata
**Builds on:** Byte 1

**In plain terms:**
For every chunk, the component reads vector similarity, shared entities, graph depth, fused score, PageRank, final score, and optional explanation/strategy metadata.

**The code:**
```python
vector_score = chunk.get("vector_score", 0.0)
shared_entities = chunk.get("shared_entities", 0)
depth = chunk.get("depth", 0)
combined_score = chunk.get("combined_score", 0.0)
pagerank_score = chunk.get("pagerank_score", 0.0)
final_score = chunk.get(
    "final_score", combined_score
)
```

The use of `.get()` makes the UI tolerant of retrieval results that do not contain every optional field.

---

### Byte 3: Proximity Is Derived When It Is Missing
**Builds on:** Byte 2

**In plain terms:**
The panel can display graph proximity even when the reranker did not explicitly store it in the chunk dictionary. It derives the value from graph depth.

**The code:**
```python
proximity = breakdown.get(
    "proximity_score",
    1.0 / (1 + max(int(depth), 0))
)
```

This keeps the display consistent with the retrieval scoring model.

---

### Byte 4: Chunk Text Is Truncated for UI Readability
**Builds on:** Byte 1

**In plain terms:**
Long evidence chunks are shortened to 300 characters in the initial display. The component is showing evidence for inspection, not attempting to reproduce the entire source document.

**The code:**
```python
display_text = (
    text[:300] + "..."
    if len(text) > 300
    else text
)
```

The retrieval data itself is not changed.

---

### Byte 5: Each Chunk Gets Its Own Expandable Card
**Builds on:** Bytes 2–4

**In plain terms:**
The panel uses a nested Streamlit expander for each chunk. The first chunk is expanded by default, while later chunks remain collapsed.

**The code:**
```python
with st.expander(
    f"[{i}] {chunk_id} — final {final_score:.3f}",
    expanded=(i == 1)
):
    ...
```

This prevents many evidence chunks from making the page excessively long.

---

### Byte 6: The Panel Explains Why a Chunk Was Selected
**Builds on:** Byte 2

**In plain terms:**
If retrieval supplied a score explanation or strategy, the panel displays them underneath the evidence. This connects the UI to the retrieval system's ranking diagnostics.

**The code:**
```python
explanation = chunk.get(
    "score_explanation", ""
)
strategy = chunk.get(
    "retrieval_strategy", ""
)
```

This is particularly useful when comparing vector, graph, adaptive, and PageRank behavior.

---

### PUTTING IT TOGETHER

The evidence panel turns retrieval dictionaries into an inspectable UI. It shows the actual text plus the signals that influenced ranking, while keeping long content collapsed. This lets a user move from the generated answer back to the evidence and the retrieval scores that supported it.
