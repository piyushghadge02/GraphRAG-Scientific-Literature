# Knowledge Bytes — `app/components/search_bar.py`

---

### Byte 1: The Search Bar Captures the User Question
**Builds on:** `main.md` Byte 10

**In plain terms:**
`render_search_bar()` creates the main text input where the user asks a scientific-literature question.

**The code:**
```python
query = st.text_input(
    "Question",
    placeholder="Ask a question about the scientific literature...",
    key="search_query",
    label_visibility="collapsed"
)
```

The `search_query` key is also used by the example-question callback.

---

### Byte 2: The User Chooses a System Mode
**Builds on:** Byte 1

**In plain terms:**
The UI lets the user choose between `Modified GraphRAG` and `Baseline`. `main.py` uses that selection to choose retrieval behavior.

**The code:**
```python
system_mode = st.radio(
    "System",
    options=[
        "Modified GraphRAG",
        "Baseline"
    ],
    index=0,
    horizontal=True,
)
```

The component does not execute either system; it only captures the choice.

---

### Byte 3: Sidebar Controls Configure Retrieval
**Builds on:** Byte 2

**In plain terms:**
`render_sidebar_settings()` exposes four retrieval switches and one result-count slider: graph expansion, adaptive retrieval, PageRank, query decomposition, and Top K.

**The code:**
```python
expand_graph = st.checkbox(
    "Graph Expansion",
    value=True
)

use_adaptive = st.checkbox(
    "Adaptive",
    value=True
)

use_gds = st.checkbox(
    "PageRank Reranking",
    value=True
)
```

The remaining controls follow the same pattern.

---

### Byte 4: The Settings Function Returns One Configuration Tuple
**Builds on:** Byte 3

**In plain terms:**
Rather than making `main.py` read five individual widget values, the component packages them into one return value.

**The code:**
```python
return (
    expand_graph,
    use_adaptive,
    use_gds,
    use_decomposition,
    top_k
)
```

This gives the application a compact interface between UI controls and retrieval orchestration.

---

### PUTTING IT TOGETHER

The search-bar component owns user-facing retrieval controls. It captures the question, system mode, and advanced retrieval options, then returns those values to `main.py`. It does not contain retrieval algorithms itself.
