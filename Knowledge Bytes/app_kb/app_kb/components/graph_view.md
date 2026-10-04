# Knowledge Bytes — `app/components/graph_view.py`

---

### Byte 1: The Graph View Visualizes Retrieved Evidence
**Builds on:** `main.md` Byte 18

**In plain terms:**
`render_graph_view()` displays the graph neighborhood around the retrieved chunks. It uses Neo4j for data and PyVis for an interactive browser visualization.

**The code:**
```python
def render_graph_view(
    driver,
    chunk_ids: List[str],
    show_similar: bool = True
):
```

The component receives an existing Neo4j driver and the current retrieval result IDs.

---

### Byte 2: Empty Retrieval Produces No Graph
**Builds on:** Byte 1

**In plain terms:**
A graph cannot be meaningfully built without chunk IDs. The function exits early and informs the user when the list is empty.

**The code:**
```python
if not chunk_ids:
    st.info("No chunks to visualize")
    return
```

This prevents unnecessary database work.

---

### Byte 3: Neo4j Supplies Chunk–Entity Relationships
**Builds on:** Byte 1

**In plain terms:**
The first Neo4j query retrieves each requested chunk, its text, and entities connected by `MENTIONS`. The query is limited to 200 records to keep the visualization bounded.

**The code:**
```cypher
MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
WHERE c.chunk_id IN $chunkIds
RETURN c.chunk_id AS chunk_id,
       c.text AS chunk_text,
       e.name AS entity_name,
       'Entity' AS entity_type
LIMIT 200
```

The visualization therefore reflects the graph structure already stored by the GraphRAG backend.

---

### Byte 4: Semantic-Similarity Edges Are Optional
**Builds on:** Byte 3

**In plain terms:**
A second query can find `SEMANTIC_SIMILAR` edges between the retrieved chunks. If that query fails, the visualization simply proceeds without those edges.

**The code:**
```python
if show_similar:
    try:
        sim_result = session.run(
            "... MATCH (a:Chunk)-[:SEMANTIC_SIMILAR]->(b:Chunk) ...",
            chunkIds=chunk_ids,
        )
    except Exception:
        similar_pairs = []
```

This makes semantic-similarity visualization an optional enhancement.

---

### Byte 5: PyVis Creates the Interactive Network
**Builds on:** Bytes 3–4

**In plain terms:**
The component creates a PyVis network with a large canvas and interactive physics. The browser receives generated HTML rather than a static image.

**The code:**
```python
net = Network(
    height="780px",
    width="100%",
    bgcolor="#FFFFFF",
    font_color="#1A1D1E",
    directed=False
)
```

The force-based physics positions connected nodes automatically.

---

### Byte 6: Chunks and Entities Have Different Visual Roles
**Builds on:** Byte 5

**In plain terms:**
Chunk nodes are large boxes; entity nodes are smaller circular nodes. This visual distinction mirrors their different roles in the knowledge graph.

**The code:**
```python
net.add_node(
    chunk_id,
    label=chunk_id,
    color="#0F5257",
    size=34,
    shape="box",
)

net.add_node(
    entity_name,
    label=entity_name,
    color="#FF6B4A",
    size=20,
    shape="dot",
)
```

The component also tracks already-added IDs so the same node is not duplicated.

---

### Byte 7: `MENTIONS` Edges Connect Chunks to Entities
**Builds on:** Byte 6

**In plain terms:**
For every chunk/entity record, the graph view draws an edge representing the `MENTIONS` relationship.

**The code:**
```python
net.add_edge(
    chunk_id,
    entity_name,
    color="#C4C0B5",
    width=2
)
```

This makes shared entities visually apparent across retrieved chunks.

---

### Byte 8: Semantic-Similarity Edges Connect Retrieved Chunks
**Builds on:** Byte 4 and Byte 7

**In plain terms:**
When available, `SEMANTIC_SIMILAR` edges are drawn between retrieved chunk nodes using a different visual style and a title containing the similarity value.

**The code:**
```python
net.add_edge(
    from_id,
    to_id,
    color="#2D9B5C",
    width=3,
    dashes=True,
    title=title
)
```

Only nodes already included in the visualization are connected.

---

### Byte 9: PyVis HTML Is Generated In Memory
**Builds on:** Byte 5

**In plain terms:**
The component generates the PyVis HTML directly as a string instead of writing a temporary HTML file. The source comments explain that this avoids Windows file-locking problems encountered with temporary-file round trips.

**The code:**
```python
html_content = net.generate_html()
```

The HTML is then lightly modified for sizing and rounded corners.

---

### Byte 10: Streamlit Embeds the Interactive HTML
**Builds on:** Byte 9

**In plain terms:**
`components.html()` places the generated PyVis document inside the Streamlit page. The height matches the configured 780-pixel graph canvas.

**The code:**
```python
components.html(
    html_content,
    height=780,
    scrolling=False
)
```

This is what makes the graph interactive inside the Knowledge Graph tab.

---

### Byte 11: The Graph Includes a Legend
**Builds on:** Bytes 6–8

**In plain terms:**
A small legend explains which visual elements represent chunks, entities, `MENTIONS`, and `SEMANTIC_SIMILAR` relationships.

**The code:**
```python
st.markdown(
    "<div class='graph-legend'>"
    "Evidence Chunk · Entity · MENTIONS · SEMANTIC_SIMILAR"
    "</div>",
    unsafe_allow_html=True
)
```

The actual source uses inline HTML for this legend; the snippet above captures its single responsibility.

---

### Byte 12: Visualization Failures Do Not Hide the Answer
**Builds on:** Bytes 1–11

**In plain terms:**
The whole graph-generation block is protected by `try/except`. If Neo4j visualization or PyVis fails, the application reports that graph visualization is optional and the answer/evidence remain available.

**The code:**
```python
except Exception as e:
    st.error(
        f"Failed to generate graph visualization: {str(e)}"
    )
    st.info(
        "Graph visualization is optional. "
        "The answer and evidence are still available above."
    )
```

This keeps visualization from becoming a hard dependency for the core Q&A experience.

---

### PUTTING IT TOGETHER

The graph-view component takes the current retrieved chunk IDs, queries Neo4j for their entity relationships, optionally adds semantic-similarity edges, and converts that subgraph into an interactive PyVis visualization. It deliberately visualizes only the current evidence neighborhood rather than the entire database. Failures are contained so the core answer and evidence UI remain useful.
