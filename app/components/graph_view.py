"""
Phase 5 - Graph Visualization Component
Visualizes the knowledge graph around retrieved chunks using PyVis.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import List
from pyvis.network import Network
import re


def render_graph_view(graph_data: dict, show_similar: bool = True):
    """
    Render an interactive knowledge graph visualization.

    Shows retrieved chunks and their connected entities with MENTIONS
    relationships, plus SEMANTIC_SIMILAR edges between retrieved chunks
    where available. Data comes from GET /graph (no Neo4j access here).

    Args:
        graph_data: Dict with "records" (chunk/entity rows) and
            "similar_pairs" (SEMANTIC_SIMILAR edges).
        show_similar: Whether to include SEMANTIC_SIMILAR edges (default: True)
    """

    records = graph_data.get("records", []) if graph_data else []
    similar_pairs = graph_data.get("similar_pairs", []) if graph_data else []

    if not records:
        st.markdown(
            """
            <div class="card" style="text-align: center; padding: 3rem;">
                <p style="color: var(--color-text-muted); margin: 0;">No chunks to visualize</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    st.markdown('<p class="section-title">Knowledge Graph Context</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">Entity relationships across the retrieved literature</p>',
        unsafe_allow_html=True
    )

    try:
        if not show_similar:
            similar_pairs = []

        if not records:
            st.warning("No graph data found for retrieved chunks")
            return

        # Create PyVis network with dark background
        net = Network(
            height="780px",
            width="100%",
            bgcolor="#0D1117",
            font_color="#E6EDF3",
            directed=False
        )

        # Configure physics
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            },
            "nodes": {
                "font": {
                    "color": "#E6EDF3",
                    "size": 12,
                    "face": "Inter"
                },
                "borderWidth": 2
            },
            "edges": {
                "color": {
                    "color": "#30363D"
                },
                "width": 2
            }
        }
        """)

        # Track added nodes to avoid duplicates
        added_chunks = set()
        added_entities = set()

        # Add nodes and edges
        for record in records:
            chunk_id = record["chunk_id"]
            entity_name = record["entity_name"]
            entity_type = record["entity_type"] or "Unknown"

            # Add chunk node (teal box)
            if chunk_id not in added_chunks:
                chunk_text = record["chunk_text"][:100] + "..." if len(record["chunk_text"]) > 100 else record["chunk_text"]
                net.add_node(
                    chunk_id,
                    label=chunk_id,
                    title=chunk_text,
                    color="#00D4AA",
                    size=34,
                    shape="box",
                    font={"color": "#0D1117", "size": 12, "face": "JetBrains Mono"},
                    borderWidth=0
                )
                added_chunks.add(chunk_id)

            # Add entity node (coral dot)
            if entity_name not in added_entities:
                entity_type = record["entity_type"] if record["entity_type"] else "Entity"
                net.add_node(
                    entity_name,
                    label=entity_name,
                    title=f"{entity_name} ({entity_type})",
                    color="#F85149",
                    size=20,
                    shape="dot",
                    font={"color": "#E6EDF3", "size": 12, "face": "Inter"},
                    borderWidth=2,
                    borderWidthSelected=3
                )
                added_entities.add(entity_name)

            # Add MENTIONS edge (subtle gray)
            net.add_edge(chunk_id, entity_name, color="#30363D", width=2)

        # Add SEMANTIC_SIMILAR edges between retrieved chunk nodes (green)
        for pair in similar_pairs:
            from_id, to_id = pair["from_id"], pair["to_id"]
            if from_id in added_chunks and to_id in added_chunks:
                sim = pair["similarity"]
                title = f"SEMANTIC_SIMILAR ({sim:.3f})" if sim is not None else "SEMANTIC_SIMILAR"
                net.add_edge(from_id, to_id, color="#3FB950", width=3,
                             dashes=True, title=title)

        # Generate HTML directly as a string (no temp file: Windows
        # file locking (WinError 32) made the NamedTemporaryFile
        # round-trip unreliable). Same PyVis output, kept in memory.
        html_content = net.generate_html()

        # Style the emitted #mynetwork CSS rule (pyvis 0.3.2 emits the
        # canvas size as a stylesheet rule, not inline div styles).
        # Rounded corners/overflow only — size comes from Network().
        html_content = html_content.replace(
            "#mynetwork {",
            "#mynetwork {\n border-radius: 8px;\n overflow: hidden;",
            1,
        )

        # Also ensure body and html have proper sizing
        html_content = re.sub(
            r'<body>',
            '<body style="margin: 0; padding: 0; overflow: hidden; height: 780px;">',
            html_content
        )

        html_content = re.sub(
            r'<html>',
            '<html style="height: 780px;">',
            html_content
        )

        # Display graph using st.components.v1.html() for interactive PyVis content
        # Height matches the 780px canvas; width fills the tab column.
        components.html(html_content, height=780, scrolling=False)

        # Legend
        st.markdown(
            """
            <div class="graph-legend">
                <div class="graph-legend-item">
                    <span class="graph-legend-swatch chunk"></span>
                    <span style="color: #E6EDF3;">Evidence Chunk</span>
                </div>
                <div class="graph-legend-item">
                    <span class="graph-legend-swatch entity"></span>
                    <span style="color: #E6EDF3;">Entity</span>
                </div>
                <div class="graph-legend-item">
                    <span class="graph-legend-swatch edge"></span>
                    <span style="color: #E6EDF3;">MENTIONS relationship</span>
                </div>
                <div class="graph-legend-item">
                    <span class="graph-legend-swatch similar"></span>
                    <span style="color: #E6EDF3;">SEMANTIC_SIMILAR relationship</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.markdown(
            f"""
            <div class="card" style="text-align: center; padding: 2rem;">
                <p style="color: var(--color-danger); margin-bottom: 0.5rem;">Failed to generate graph visualization</p>
                <p style="color: var(--color-text-muted); font-size: 0.9rem;">{str(e)}</p>
                <p style="color: var(--color-text-muted); font-size: 0.85rem; margin-top: 0.5rem;">Graph visualization is optional. The answer and evidence are still available above.</p>
            </div>
            """,
            unsafe_allow_html=True
        )