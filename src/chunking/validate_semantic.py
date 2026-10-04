"""
Task 4 — Validation plot for semantic clustering.
Load semantic_cluster.jsonl → re-embed full chunk text → UMAP 2-D →
k-means (k=15) for visual coloring → interactive Plotly scatter →
save as semantic_validation.html.
"""

import json
from pathlib import Path

import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from tqdm import tqdm
from umap import UMAP

from src.embeddings import encode_texts, get_retrieval_model_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_cluster.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_validation.html"


def main() -> None:
    # ── Load chunks ──────────────────────────────────────────────
    print(f"[validate] Loading chunks from {INPUT_PATH} …")
    chunks = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    print(f"[validate] {len(chunks)} chunks loaded.")

    # ── Embed full chunk text ────────────────────────────────────
    print(f"[validate] Embedding chunks with {get_retrieval_model_name()} …")
    embeddings = encode_texts(texts, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings)

    # ── UMAP reduction to 2-D ───────────────────────────────────
    print("[validate] Running UMAP (n_neighbors=15, min_dist=0.1) …")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)

    # ── K-means for visual grouping ──────────────────────────────
    print("[validate] K-means clustering (k=15) on UMAP coordinates …")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    visual_labels = kmeans.fit_predict(coords)

    # ── Build Plotly scatter ─────────────────────────────────────
    hover_texts = [t[:150] + "…" if len(t) > 150 else t for t in texts]

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=[str(l) for l in visual_labels],
        hover_name=[c["chunk_id"] for c in chunks],
        hover_data={"text_preview": hover_texts},
        labels={"x": "UMAP-1", "y": "UMAP-2", "color": "Visual cluster"},
        title="Semantic Chunks — UMAP projection (colored by k-means visual cluster)",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        template="plotly_dark",
        width=1200,
        height=800,
        legend_title_text="Visual cluster",
    )

    # ── Save ─────────────────────────────────────────────────────
    fig.write_html(str(OUTPUT_PATH))
    print(f"\n[validate] ✓ Saved interactive plot → {OUTPUT_PATH}")
    print(f"[validate]   Open in your browser:  open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
