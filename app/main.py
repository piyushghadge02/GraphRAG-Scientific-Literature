"""
Phase 5 - Interactive Streamlit Demo for GraphRAG Scientific Literature
Full-featured Q&A interface with query decomposition, PageRank reranking, and graph visualization.
"""

import streamlit as st
import json
import os
import sys
import urllib.request
import urllib.error

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Scientific Literature Q&A",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI backend (Step 3: Streamlit talks to the API, never to Neo4j/LLM
# directly for the query flow). Configurable; defaults to local laptop API.
API_URL = os.getenv("GRAPHRAG_API_URL", "http://localhost:8000").rstrip("/")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.components.search_bar import render_search_bar, render_sidebar_settings
from app.components.answer_card import render_answer_card
from app.components.evidence_panel import render_evidence_panel
from app.components.graph_view import render_graph_view

# Load custom CSS
css_path = os.path.join(project_root, "app", "styles", "theme.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def api_get(path: str, timeout: int = 120) -> dict:
    """GET JSON from the FastAPI backend (stdlib only); raise RuntimeError on failure."""
    try:
        with urllib.request.urlopen(API_URL + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(f"API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach GraphRAG API at {API_URL}: {e.reason}")


def fetch_graph_data(chunk_ids: list) -> dict:
    """Fetch Knowledge Graph viz data for chunk IDs (empty lists if none)."""
    if not chunk_ids:
        return {"records": [], "similar_pairs": []}
    from urllib.parse import quote
    return api_get("/graph?chunk_ids=" + quote(",".join(chunk_ids)))


def build_query_payload(query, system_mode, expand_graph, use_adaptive, use_gds,
                        use_decomposition, top_k) -> dict:
    """Build the POST /query body from the current UI settings (pure)."""
    return {
        "query": query,
        "system_mode": system_mode,
        "expand_graph": expand_graph,
        "use_adaptive": use_adaptive,
        "use_gds": use_gds,
        "use_decomposition": use_decomposition,
        "top_k": top_k,
    }


def api_post(path: str, payload: dict, timeout: int = 600) -> dict:
    """POST JSON to the FastAPI backend (stdlib only); raise RuntimeError on failure."""
    request = urllib.request.Request(
        API_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(f"API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach GraphRAG API at {API_URL}: {e.reason}")


def api_health() -> bool:
    """True when GET /health answers 200 (backend reachable)."""
    try:
        with urllib.request.urlopen(API_URL + "/health", timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def set_example_question(question):
    """Callback to set example question in session state."""
    st.session_state.search_query = question


def main():
    """Main application logic."""

    # No local backends: retrieval/generation/graph data all come from FastAPI.
    # Gate on API health instead.
    if not api_health():
        st.markdown(
            f"""
            <div class="card" style="max-width: 600px; margin: 4rem auto; text-align: center;">
                <h2 style="margin-bottom: 1rem; color: var(--color-text-primary);">API Unavailable</h2>
                <p style="color: var(--color-text-secondary); margin-bottom: 1.5rem;">
                    Scientific Literature Q&A cannot reach the GraphRAG API at <code>{API_URL}</code>.
                </p>
                <p style="color: var(--color-text-muted); font-size: 0.9rem;">
                    Start the API server:
                </p>
                <code style="background: var(--color-bg); padding: 1rem; border-radius: var(--radius-md); display: block; margin: 1rem auto; max-width: 400px;">
                    uvicorn api.main:app --host 127.0.0.1 --port 8000
                </code>
                <p style="color: var(--color-text-muted); font-size: 0.85rem; margin-top: 1rem;">
                    Or set <code>GRAPHRAG_API_URL</code> to your API address.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    # Sidebar: retrieval settings - compact, no branding
    with st.sidebar:
        st.markdown('<p class="section-title">Retrieval Settings</p>', unsafe_allow_html=True)
        expand_graph, use_adaptive, use_gds, use_decomposition, top_k = render_sidebar_settings()
        
        st.markdown('<div style="margin: 1rem 0 0.5rem 0;"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card" style="padding: 0.75rem;">
                <p class="section-title" style="margin-bottom: 0.4rem;">System Status</p>
                <div class="status-strip">
                    <span class="status-badge active"><span class="status-dot"></span>API Connected</span>
                    <span class="status-badge active"><span class="status-dot" style="background: var(--color-primary);"></span>Neo4j</span>
                    <span class="status-badge active"><span class="status-dot" style="background: var(--color-primary);"></span>Vector Index</span>
                    <span class="status-badge active"><span class="status-dot" style="background: var(--color-primary);"></span>LLM</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Header - compact with GRAPHRAG RESEARCH eyebrow
    st.markdown(
        """
        <div style="margin-bottom: 1rem;">
            <p style="color: var(--color-text-muted); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem;">GRAPHRAG RESEARCH</p>
            <h1 style="margin-bottom: 0.4rem; font-size: 1.75rem; font-weight: 700; color: var(--color-text-primary);">Scientific Literature Q&A</h1>
            <p style="color: var(--color-text-secondary); font-size: 0.95rem; line-height: 1.5; max-width: 700px; margin: 0;">
                Explore scientific literature with evidence-backed answers powered by GraphRAG, hybrid retrieval, and knowledge graphs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Search interface - compact container, no extra blank bar
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    query, system_mode = render_search_bar()
    st.markdown('</div>', unsafe_allow_html=True)

    # Example questions - compact inline buttons
    st.markdown('<p class="section-title">Example Questions</p>', unsafe_allow_html=True)
    eq_col1, eq_col2, eq_col3 = st.columns(3, gap="small")
    with eq_col1:
        st.button(
            "Oxidative stress in neurodegeneration",
            on_click=set_example_question,
            args=("What role does oxidative stress play in neurodegenerative diseases?",),
            type="secondary",
            use_container_width=True,
            key="eq_1"
        )
    with eq_col2:
        st.button(
            "Cardiovascular risk factors",
            on_click=set_example_question,
            args=("What are the risk factors associated with cardiovascular disease?",),
            type="secondary",
            use_container_width=True,
            key="eq_2"
        )
    with eq_col3:
        st.button(
            "Mitochondrial dysfunction & cell death",
            on_click=set_example_question,
            args=("How does mitochondrial dysfunction contribute to programmed cell death?",),
            type="secondary",
            use_container_width=True,
            key="eq_3"
        )

    # Search button - centered below examples with proper spacing
    st.markdown('<div style="margin: 0.75rem 0 0.25rem 0;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_clicked = st.button(
            "Search Literature",
            type="primary",
            use_container_width=True,
            key="search_btn"
        )

    if search_clicked:
        if not query or not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Processing your question..."):
                try:
                    # Query flow goes through FastAPI (same branching/payload
                    # the backend implements from the Streamlit settings).
                    payload = build_query_payload(
                        query, system_mode, expand_graph, use_adaptive,
                        use_gds, use_decomposition, top_k,
                    )
                    data = api_post("/query", payload)
                    answer = data["answer"]
                    retrieved_chunks = data["retrieved_chunks"]

                    if not retrieved_chunks:
                        st.warning("No relevant chunks found. Try a different question.")
                        st.stop()

                    # Success status with stats
                    strategy = data.get("retrieval_strategy", "n/a")
                    breakdown = retrieved_chunks[0].get("score_breakdown", {}) if retrieved_chunks else {}
                    weights_line = ""
                    if breakdown:
                        weights_line = (
                            f"Fusion α={breakdown.get('alpha', 0):.2f} "
                            f"β={breakdown.get('beta', 0):.2f} "
                            f"γ={breakdown.get('gamma', 0):.2f}"
                        )

                    st.markdown(
                        f"""
                        <div class="stats-strip">
                            <div class="stat-item">
                                <span class="stat-label">Chunks Retrieved</span>
                                <span class="stat-value">{len(retrieved_chunks)}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Strategy</span>
                                <span class="stat-value" style="color: var(--color-primary);">{strategy}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Mode</span>
                                <span class="stat-value">{system_mode}</span>
                            </div>
                            {f'<div class="stat-item"><span class="stat-label">Weights</span><span class="stat-value" style="font-size: 0.8rem;">{weights_line}</span></div>' if weights_line else ''}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display results in tabs
                    st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

                    answer_tab, evidence_tab, graph_tab = st.tabs(
                        ["Generated Answer", "Evidence", "Knowledge Graph"]
                    )

                    with answer_tab:
                        render_answer_card(answer)

                    with evidence_tab:
                        render_evidence_panel(retrieved_chunks)

                    with graph_tab:
                        render_graph_view(fetch_graph_data(data.get("chunk_ids", [])))

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Please try again or check the API server logs.")

    # Footer - keep existing branding
    st.markdown(
        """
        <div class="app-footer">
            <p>ScholarGraph AI — GraphRAG for Scientific Literature | Built for academic research and demonstration purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()