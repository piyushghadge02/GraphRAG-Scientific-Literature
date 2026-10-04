# Knowledge Bytes — `app/main.py`

---

### Byte 1: `main.py` Is the Streamlit Application Entry Point
**Builds on:** None — starting point

**In plain terms:**
`main.py` is the top-level UI coordinator. It initializes Streamlit, connects the UI to the existing GraphRAG modules, and controls the complete question → retrieval → generation → display flow.

**The code:**
```python
st.set_page_config(
    page_title="GraphRAG Scientific Literature Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

This must be the first Streamlit command, so page configuration happens before the rest of the UI is built.

---

### Byte 2: The Project Root Is Added to `sys.path`
**Builds on:** Byte 1

**In plain terms:**
The app lives one directory below the project root. `main.py` calculates that root and adds it to Python's import path so it can import `src.*`.

**The code:**
```python
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, project_root)
```

This connects the UI layer to the core GraphRAG implementation without copying that logic into `app`.

---

### Byte 3: `main.py` Connects UI Components to Core GraphRAG Services
**Builds on:** Byte 2

**In plain terms:**
The application imports the central configuration, embedding provider, LLM client, prompt builder, and retrieval variants. The UI therefore orchestrates existing backend functionality rather than implementing retrieval itself.

**The code:**
```python
from src.config import settings
from src.embeddings import get_embedder
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_rag_prompt
from src.retrieval.retrieve import retrieve
from src.retrieval.retrieve_with_pagerank import retrieve_with_pagerank
```

This separation is important: `app` is primarily the presentation/orchestration layer.

---

### Byte 4: Neo4j, Embeddings, and LLM Clients Are Cached
**Builds on:** Byte 3

**In plain terms:**
The app creates three expensive resources: a Neo4j driver, an embedding model, and an LLM client. `st.cache_resource` keeps them alive across Streamlit reruns instead of rebuilding them for every interaction.

**The code:**
```python
@st.cache_resource
def init_neo4j_driver():
    ...

@st.cache_resource
def init_embedding_model():
    ...

@st.cache_resource
def init_llm_client():
    ...
```

This is especially important for transformer models because repeatedly loading the model would be slow and memory-heavy.

---

### Byte 5: Neo4j Initialization Tests the Connection
**Builds on:** Byte 4

**In plain terms:**
The Neo4j driver is created from the centralized settings and immediately tested with a tiny query. A connection failure becomes a user-visible error instead of appearing later during retrieval.

**The code:**
```python
driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)

with driver.session() as session:
    session.run("RETURN 1")
```

The function returns `None` when initialization fails.

---

### Byte 6: The Embedding Model Uses the Central Provider
**Builds on:** Bytes 3–4

**In plain terms:**
The UI does not directly construct a SentenceTransformer. It calls the project's central embedding provider, ensuring the app uses the same configured model as the rest of GraphRAG.

**The code:**
```python
return get_embedder(
    get_retrieval_model_name()
)
```

This prevents the UI from accidentally drifting to a different embedding model.

---

### Byte 7: The LLM Client Is Created Once
**Builds on:** Byte 4

**In plain terms:**
The UI creates `LLMClient` from the configured backend and model. Temperature is explicitly set to zero for the application.

**The code:**
```python
return LLMClient(
    backend=settings.LLM_BACKEND,
    model=settings.LLM_MODEL,
    temperature=0.0
)
```

The UI therefore delegates provider-specific generation behavior to the core generation module.

---

### Byte 8: Example Questions Use Session State
**Builds on:** Byte 1

**In plain terms:**
The example-question buttons do not perform retrieval themselves. Their callback writes the selected question into Streamlit's session state, which populates the search input.

**The code:**
```python
def set_example_question(question):
    st.session_state.search_query = question
```

This keeps example buttons as input shortcuts rather than creating a second search implementation.

---

### Byte 9: Sidebar Controls Define Retrieval Behavior
**Builds on:** Byte 8

**In plain terms:**
The sidebar lets the user choose graph expansion, adaptive retrieval, PageRank, query decomposition, and the number of results. These values are passed into the selected retrieval path.

**The code:**
```python
expand_graph, use_adaptive, use_gds, use_decomposition, top_k = render_sidebar_settings()
```

The UI exposes retrieval configuration without changing the underlying retrieval code.

---

### Byte 10: The Search Input Returns a Query and System Mode
**Builds on:** Byte 9

**In plain terms:**
The search component supplies the actual question and whether the user selected the Modified GraphRAG or Baseline system.

**The code:**
```python
query, system_mode = render_search_bar()
```

`main.py` uses this pair to choose the retrieval path when the Search button is pressed.

---

### Byte 11: Query Decomposition Gets Its Own Retrieval Path
**Builds on:** Bytes 9–10

**In plain terms:**
When decomposition is enabled, the app sends the question to `retrieve_decomposed()`. That function handles splitting the question and combining results; the UI only limits the merged result to `top_k`.

**The code:**
```python
retrieved_chunks = retrieve_decomposed(
    driver=driver,
    embedding_model=embedding_model,
    question=query,
    llm_client=llm_client,
    top_k_per_subq=3,
    expand_graph=expand_graph
)

retrieved_chunks = retrieved_chunks[:top_k]
```

This keeps query decomposition optional.

---

### Byte 12: The Baseline Path Uses Fixed Retrieval Settings
**Builds on:** Byte 10

**In plain terms:**
Selecting Baseline calls the regular `retrieve()` function with adaptive skipping disabled and fixed legacy fusion weights. PageRank is explicitly not used.

**The code:**
```python
retrieved_chunks = retrieve(
    driver=driver,
    embedding_model=embedding_model,
    query=query,
    top_k=top_k,
    expand_graph=expand_graph,
    adaptive_enabled=False,
    alpha=0.7,
    beta=0.3,
    gamma=0.0,
)
```

The app then supplies zero PageRank scores so the displayed result fields stay consistent.

---

### Byte 13: The Modified Path Can Add PageRank and Adaptive Retrieval
**Builds on:** Byte 12

**In plain terms:**
The Modified GraphRAG path calls `retrieve_with_pagerank()`. The sidebar controls determine whether PageRank and adaptive retrieval are active.

**The code:**
```python
retrieved_chunks = retrieve_with_pagerank(
    driver=driver,
    embedding_model=embedding_model,
    query=query,
    top_k=top_k,
    expand_graph=expand_graph,
    use_pagerank=use_gds,
    adaptive_enabled=use_adaptive
)
```

The UI therefore exposes the advanced retrieval pipeline without implementing its algorithms.

---

### Byte 14: Empty Retrieval Stops the Request Early
**Builds on:** Bytes 11–13

**In plain terms:**
If no chunks are retrieved, there is no evidence from which to build the answer. The app shows a warning and stops the current Streamlit execution.

**The code:**
```python
if not retrieved_chunks:
    st.warning(
        "No relevant chunks found. Try a different question."
    )
    st.stop()
```

This prevents the generation stage from running with an empty context.

---

### Byte 15: The UI Shows Which Retrieval Strategy Was Used
**Builds on:** Bytes 11–13

**In plain terms:**
After retrieval, the app reads metadata from the first result to display the selected strategy and fusion weights. This gives the user visibility into how the evidence was ranked.

**The code:**
```python
strategy = retrieved_chunks[0].get(
    "retrieval_strategy", "n/a"
)
breakdown = retrieved_chunks[0].get(
    "score_breakdown", {}
)
```

The display is diagnostic information; it does not alter the retrieved results.

---

### Byte 16: Retrieved Evidence Is Sent to the Prompt Builder
**Builds on:** Byte 14

**In plain terms:**
Once evidence exists, `main.py` builds the RAG prompt using the original question and retrieved chunks, then asks the LLM client to generate the answer.

**The code:**
```python
prompt = build_rag_prompt(
    query,
    retrieved_chunks
)
answer = llm_client.generate(prompt)
```

The UI does not manually construct the evidence text; that responsibility belongs to the generation module.

---

### Byte 17: Results Are Organized into Three Tabs
**Builds on:** Byte 16

**In plain terms:**
The application separates the result into the generated answer, supporting evidence, and the knowledge graph. All three views are derived from the same retrieval result.

**The code:**
```python
answer_tab, evidence_tab, graph_tab = st.tabs(
    ["Answer", "Evidence", "Knowledge Graph"]
)
```

This keeps the main answer readable while still exposing the underlying evidence and graph.

---

### Byte 18: The Knowledge Graph Uses Retrieved Chunk IDs
**Builds on:** Byte 17

**In plain terms:**
The graph view does not visualize the entire Neo4j database. It receives only the chunk IDs that were retrieved for the current question.

**The code:**
```python
chunk_ids = [
    c["chunk_id"]
    for c in retrieved_chunks
]

render_graph_view(driver, chunk_ids)
```

This keeps the visualization focused on the evidence relevant to the current query.

---

### Byte 19: Errors Are Contained at the Search Boundary
**Builds on:** Bytes 11–18

**In plain terms:**
The complete search workflow is inside a `try/except`. If retrieval, generation, or visualization fails, the user gets an error message instead of a raw application crash.

**The code:**
```python
except Exception as e:
    st.error(
        f"An error occurred: {str(e)}"
    )
    st.info(
        "Please try again or contact support if the issue persists."
    )
```

The application intentionally treats graph visualization as part of the same protected request flow.

---

### PUTTING IT TOGETHER

`main.py` is the conductor of the UI, not the implementation of GraphRAG itself. It initializes cached backend resources, collects user settings, chooses among decomposition/baseline/modified retrieval paths, sends the resulting evidence to the LLM, and displays the answer, evidence, and graph. The core work remains in `src`, while `app` turns those services into an interactive Streamlit application.
