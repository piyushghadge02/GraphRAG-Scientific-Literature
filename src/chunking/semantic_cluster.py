"""
Task 2c — Semantic-cluster chunking.
1. Split into sentences (spaCy sentencizer).
2. Embed each sentence with the PDF-required all-MiniLM-L6-v2
   (SEMANTIC_CLUSTER_MODEL_NAME; do NOT change in Phase 1).
3. Cluster with HDBSCAN (min_cluster_size=2).
4. Group sentences by cluster label, preserving original order.
5. Noise points (label -1) become singleton chunks.
6. Abstracts with < 3 sentences skip clustering entirely.
"""

from typing import List

import hdbscan
import numpy as np
import spacy

from src.embeddings import get_cluster_model_name, get_embedder

# Lightweight sentencizer (same approach as sentence_boundary.py)
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")

# Embedding model for semantic clustering — PDF-required baseline.
# Loaded once via the central provider, reused across calls.
# NOTE: this intentionally uses SEMANTIC_CLUSTER_MODEL_NAME (not the
# configurable retrieval model) so clustering stays on all-MiniLM-L6-v2.


def _get_embedder():
    """Return the cached PDF-required clustering embedder."""
    return get_embedder(get_cluster_model_name())


def chunk(article_id: str, text: str) -> List[dict]:
    """
    Semantically cluster sentences of *text*, then form chunks per cluster.
    """
    doc = _nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    # Too few sentences to cluster meaningfully — treat whole abstract as one chunk
    if len(sentences) < 3:
        return [{
            "article_id": article_id,
            "chunk_id": f"{article_id}_chunk_0",
            "text": " ".join(sentences),
            "strategy": "semantic_cluster",
        }]

    # Embed every sentence
    embedder = _get_embedder()
    embeddings = embedder.encode(sentences, show_progress_bar=False)

    # Cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    # Group by cluster label, preserving original sentence order.
    # label == -1 means noise → each becomes its own chunk.
    cluster_groups: dict[int, List[int]] = {}
    noise_indices: List[int] = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            cluster_groups.setdefault(label, []).append(idx)

    chunks: List[dict] = []

    # Emit cluster chunks (ordered by first-appearing sentence in each cluster)
    for _label in sorted(cluster_groups, key=lambda l: cluster_groups[l][0]):
        indices = cluster_groups[_label]
        grouped_text = " ".join(sentences[i] for i in indices)
        chunks.append({
            "article_id": article_id,
            "chunk_id": f"{article_id}_chunk_{len(chunks)}",
            "text": grouped_text,
            "strategy": "semantic_cluster",
        })

    # Emit noise sentences as singleton chunks
    for idx in noise_indices:
        chunks.append({
            "article_id": article_id,
            "chunk_id": f"{article_id}_chunk_{len(chunks)}",
            "text": sentences[idx],
            "strategy": "semantic_cluster",
        })

    return chunks
