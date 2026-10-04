"""
Phase 5 - Answer Card Component
Displays the generated answer with styling.
"""

import streamlit as st
import re


def render_answer_card(answer: str):
    """
    Render the generated answer in a styled card.

    Args:
        answer: Generated answer text from the LLM
    """

    st.markdown('<p class="section-title">Generated Answer</p>', unsafe_allow_html=True)

    # Highlight citation numbers [1], [2], etc. with primary color
    answer_html = re.sub(
        r'\[(\d+)\]',
        r'<span style="color: var(--color-primary); font-weight: 600;">[\1]</span>',
        answer
    )

    # Render answer in hero card (presentation only; text unchanged)
    st.markdown(
        f"""
        <div class="answer-hero">
            <p>{answer_html}</p>
        </div>
        """,
        unsafe_allow_html=True
    )