"""
Phase 5 - Search Bar Component
Provides the main search interface with controls for GraphRAG parameters.
"""

import streamlit as st


def render_search_bar():
    """
    Render the search bar and system selector (main content area).

    Returns:
        Tuple of (query, system_mode).
        - system_mode: "Modified GraphRAG" or "Baseline"
    """

    # Search input
    query = st.text_input(
        "Question",
        placeholder="Ask a question about the scientific literature...",
        key="search_query",
        label_visibility="collapsed"
    )

    st.markdown('<div style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)

    # System mode selection (Baseline vs Modified GraphRAG)
    st.markdown('<p class="section-title">System</p>', unsafe_allow_html=True)
    system_mode = st.radio(
        "System",
        options=["Modified GraphRAG", "Baseline"],
        index=0,
        horizontal=True,
        help="Baseline: required expansion + legacy fusion. Modified: adaptive retrieval, graph-aware fusion, PageRank.",
        label_visibility="collapsed"
    )

    st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)

    return query, system_mode


def render_sidebar_settings():
    """
    Render the retrieval settings controls in the sidebar.

    Same controls, labels, defaults, and help texts as before — only the
    render location changed. Returns:
        Tuple of (expand_graph, use_adaptive, use_gds, use_decomposition, top_k)
    """

    # Graph Retrieval section
    st.markdown(
        """
        <p class="settings-section-title">Graph Retrieval</p>
        """,
        unsafe_allow_html=True
    )

    expand_graph = st.checkbox(
        "Graph Expansion",
        value=True,
        help="Expand retrieval via entity relationships (Chunk → Entity ← Chunk traversal)"
    )

    use_adaptive = st.checkbox(
        "Adaptive Retrieval",
        value=True,
        help="Let the system skip expansion when vector hits are confident"
    )

    st.markdown('<div style="margin: 0.75rem 0 0.25rem 0;"></div>', unsafe_allow_html=True)

    # Reranking & Processing section
    st.markdown(
        """
        <p class="settings-section-title">Reranking & Processing</p>
        """,
        unsafe_allow_html=True
    )

    use_gds = st.checkbox(
        "PageRank Reranking",
        value=True,
        help="Use PageRank (Neo4j GDS) to rerank results by graph centrality"
    )

    use_decomposition = st.checkbox(
        "Query Decomposition",
        value=False,
        help="Break complex queries into sub-queries for broader coverage"
    )

    st.markdown('<div style="margin: 0.75rem 0 0.25rem 0;"></div>', unsafe_allow_html=True)

    # Results section
    st.markdown(
        """
        <p class="settings-section-title">Results</p>
        """,
        unsafe_allow_html=True
    )

    top_k = st.slider(
        "Top K Results",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of chunks to retrieve and use for generation"
    )

    return expand_graph, use_adaptive, use_gds, use_decomposition, top_k