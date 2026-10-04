"""
Phase 5 - Metrics Panel Component
Displays Phase 4 evaluation metrics from the real CSV file.
"""

import streamlit as st
import pandas as pd
import os


def render_metrics_panel():
    """
    Render evaluation metrics from Phase 3 results.

    Reads from data/processed/eval_comparison.csv and displays:
    - Retrieval metrics (Recall@5, Recall@10, MRR): Baseline vs Modified
    - Generation metrics (ROUGE-L, BERTScore)
    """

    csv_path = "data/processed/eval_comparison.csv"

    if not os.path.exists(csv_path):
        st.info("Evaluation data not available yet.")
        return

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            st.info("No evaluation data found.")
            return

        # Parse the CSV structure: Metric, Baseline, Full_System, Delta
        metrics_dict = {}
        for _, row in df.iterrows():
            metric_name = row['Metric']
            metrics_dict[metric_name] = {
                'baseline': row['Baseline'],
                'full': row['Full_System'],
                'delta': row['Delta']
            }

        # Display retrieval metrics
        st.markdown("**Retrieval**")

        for metric_name in ['Recall@5', 'Recall@10', 'MRR']:
            if metric_name in metrics_dict:
                m = metrics_dict[metric_name]
                baseline_val = m['baseline']
                full_val = m['full']

                if baseline_val != 'N/A' and full_val != 'N/A':
                    baseline_val = float(baseline_val)
                    full_val = float(full_val)
                    delta = full_val - baseline_val
                    if delta > 0:
                        arrow, color, cls = '↑', '#2D9B5C', 'metric-delta-up'
                    elif delta < 0:
                        arrow, color, cls = '↓', '#D64545', 'metric-delta-down'
                    else:
                        arrow, color, cls = '=', '#5C6265', 'metric-delta-flat'

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{metric_name}</div>
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                <span class="metric-value">{full_val:.3f}</span>
                                <span style="font-size: 0.75rem;" class="{cls}">
                                    {arrow} {abs(delta):.3f}
                                </span>
                            </div>
                            <div style="font-size: 0.7rem; color: #5C6265; margin-top: 0.2rem;">Modified vs baseline {baseline_val:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        st.markdown("")
        st.markdown("**Generation**")

        for metric_name in ['ROUGE-L F1', 'BERTScore F1']:
            if metric_name in metrics_dict:
                m = metrics_dict[metric_name]
                full_val = m['full']

                if full_val != 'N/A':
                    full_val = float(full_val)

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{metric_name}</div>
                            <div class="metric-value">{full_val:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # Interpretation
        with st.expander("Interpretation"):
            st.markdown("""
            **How to read this panel:**

            Values come from `data/processed/eval_comparison.csv`, written by
            `python -m src.evaluation.compare` (same questions for every system).

            - **Retrieval:** article-level Recall@5 / Recall@10 / MRR.
            - **Generation:** ROUGE-L F1 / BERTScore F1 vs PubMedQA long answers.
            - Deltas are Modified (adaptive GraphRAG) minus PDF baseline.
            - No conclusion is shown here beyond the measured numbers; see the
              evaluation report for interpretation once results exist.
            """)

    except Exception as e:
        st.info("Evaluation data not available yet.")
        # Log error for debugging
        import traceback
        print(f"Error loading metrics: {str(e)}")
        print(traceback.format_exc())
