"""
Phase 5 - Evidence Panel Component
Displays retrieved chunks as evidence with scores.
"""

import streamlit as st
from typing import List, Dict


def render_evidence_panel(chunks: List[Dict]):
    """
    Render the evidence panel showing retrieved chunks with scores.

    Args:
        chunks: List of retrieved chunks with scores
    """

    st.markdown('<p class="section-title">Retrieved Evidence</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color: var(--color-text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">{len(chunks)} chunks used to generate the answer</p>',
        unsafe_allow_html=True
    )

    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "")
        vector_score = chunk.get("vector_score", 0.0)
        shared_entities = chunk.get("shared_entities", 0)
        depth = chunk.get("depth", 0)
        combined_score = chunk.get("combined_score", 0.0)
        pagerank_score = chunk.get("pagerank_score", 0.0)
        final_score = chunk.get("final_score", combined_score)
        breakdown = chunk.get("score_breakdown", {}) or {}
        proximity = breakdown.get("proximity_score", 1.0 / (1 + max(int(depth), 0)))
        explanation = chunk.get("score_explanation", "")
        strategy = chunk.get("retrieval_strategy", "")

        # Display full text in expander
        with st.expander(f"[{i}] {chunk_id}  •  Final Score: {final_score:.3f}", expanded=(i == 1)):
            # Card header with ID and scores
            st.markdown(
                f"""
                <div class="evidence-card-header">
                    <span class="evidence-card-id">[{i}] {chunk_id}</span>
                    <div class="evidence-card-scores">
                        <div class="evidence-score">
                            <span class="evidence-score-label">Vector</span>
                            <span class="evidence-score-value">{vector_score:.3f}</span>
                        </div>
                        <div class="evidence-score">
                            <span class="evidence-score-label">Entities</span>
                            <span class="evidence-score-value">{shared_entities}</span>
                        </div>
                        <div class="evidence-score">
                            <span class="evidence-score-label">Graph Depth</span>
                            <span class="evidence-score-value">{depth} ({proximity:.2f})</span>
                        </div>
                        <div class="evidence-score">
                            <span class="evidence-score-label">Combined</span>
                            <span class="evidence-score-value">{combined_score:.3f}</span>
                        </div>
                        <div class="evidence-score">
                            <span class="evidence-score-label">PageRank</span>
                            <span class="evidence-score-value">{pagerank_score:.3f}</span>
                        </div>
                        <div class="evidence-score">
                            <span class="evidence-score-label">Final</span>
                            <span class="evidence-score-value final">{final_score:.3f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Chunk text
            st.markdown(
                f'<p class="evidence-text">{text}</p>',
                unsafe_allow_html=True
            )

            # Metadata
            meta_parts = []
            if explanation:
                meta_parts.append(f"Why selected: {explanation}")
            if strategy:
                meta_parts.append(f"Strategy: {strategy}")
            
            if meta_parts:
                st.markdown(
                    f'<p class="evidence-meta">{"  |  ".join(meta_parts)}</p>',
                    unsafe_allow_html=True
                )