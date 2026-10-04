"""
Task 4 — Semantic Similarity Computation
Computes semantic similarity edges among chunks.
Uses scikit-learn cosine_similarity for N < 10,000 chunks.
Uses FAISS IndexFlatIP (with L2-normalized embeddings) for N >= 10,000 chunks.

Chunk (retrieval) embeddings use the configured EMBEDDING_MODEL_NAME
(PDF baseline: all-MiniLM-L6-v2, 384D). The embedding dimension is resolved
via src.embeddings so a configured custom model flows through without
rewriting this module. Changing the model requires rebuild/reindex.
"""

from typing import List, Tuple

import numpy as np
from src.config import settings
from src.embeddings import (
    encode_texts,
    get_chunk_embedding_dimension,
    get_embedder,
    get_retrieval_model_name,
)


def _get_embedder():
    """Return the cached retrieval embedder (central provider)."""
    return get_embedder(get_retrieval_model_name())


def compute_chunk_embeddings(texts: List[str]) -> np.ndarray:
    """Generate retrieval-model embeddings for list of chunk texts."""
    model_name = get_retrieval_model_name()
    dim = get_chunk_embedding_dimension()
    print(f"[similarity] Embedding {len(texts)} chunks with {model_name} ({dim}D) ...")
    embeddings = encode_texts(texts, model_name=model_name, show_progress_bar=True, batch_size=256)
    if embeddings.shape[1] != dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
            f"expected {dim} for model '{model_name}'."
        )
    return np.array(embeddings, dtype=np.float32)


def compute_similarities(
    chunks: List[dict],
    embeddings: np.ndarray,
    threshold: "float | None" = None,
) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise cosine similarities above threshold.
    Returns a list of (chunk_id_1, chunk_id_2, similarity_score).

    Each unordered pair is emitted at most once (upper-triangle / seen-set),
    and MERGE on insert keeps the relationship duplicate-free.
    SEMANTIC_SIMILAR stays opt-in at build time because the O(n^2) scan is
    expensive on large corpora and FAISS is unavailable on some platforms.
    """
    num_chunks = len(chunks)
    threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD if threshold is None else float(threshold)
    triples: List[Tuple[str, str, float]] = []

    if num_chunks == 0:
        return []

    chunk_ids = [c["chunk_id"] for c in chunks]

    # Mode A: Full pairwise cosine similarity via scikit-learn (N < 10,000)
    if num_chunks < 10000:
        print(f"[similarity] Chunks count ({num_chunks}) < 10,000. Running O(n^2) sklearn pairwise search ...")
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)
        # Avoid duplicate pairs and self-similarities by keeping only the upper triangle
        for i in range(num_chunks):
            for j in range(i + 1, num_chunks):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    triples.append((chunk_ids[i], chunk_ids[j], sim))

    # Mode B: Large-N path (N >= 10,000)
    # Tries FAISS first (fastest); falls back to batched sklearn if FAISS is
    # unavailable or crashes (e.g. on Apple Silicon without a compatible build).
    else:
        # L2-normalise once — needed for both FAISS and sklearn cosine paths
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        norm_emb = (embeddings / norms).astype(np.float32)

        faiss_ok = False
        try:
            import faiss  # optional; may not be installed on Apple Silicon
            d = norm_emb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(norm_emb)
            k = min(50, num_chunks)
            print(
                f"[similarity] {num_chunks:,} chunks → FAISS IndexFlatIP, "
                f"top-{k} neighbours per chunk …"
            )
            sims_matrix, idx_matrix = index.search(norm_emb, k)
            faiss_ok = True

            seen_pairs: set = set()
            for i in range(num_chunks):
                for rank in range(k):
                    j = int(idx_matrix[i, rank])
                    sim = float(sims_matrix[i, rank])
                    if j == -1 or i == j or sim < threshold:
                        continue
                    pair = (chunk_ids[min(i, j)], chunk_ids[max(i, j)])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        triples.append((pair[0], pair[1], sim))

        except Exception as faiss_exc:
            if faiss_ok:
                raise  # FAISS was fine but search failed — real error
            print(
                f"[similarity] FAISS unavailable ({faiss_exc}); "
                f"falling back to batched sklearn cosine similarity …"
            )
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim

            # Process in row-batches to keep memory bounded (~500 rows × N cols)
            batch_rows = 500
            seen_pairs = set()
            for start in range(0, num_chunks, batch_rows):
                end = min(start + batch_rows, num_chunks)
                block = cos_sim(norm_emb[start:end], norm_emb)  # (batch, N)
                for local_i, global_i in enumerate(range(start, end)):
                    for global_j in range(global_i + 1, num_chunks):
                        sim = float(block[local_i, global_j])
                        if sim >= threshold:
                            triples.append(
                                (chunk_ids[global_i], chunk_ids[global_j], sim)
                            )

    print(f"[similarity] Found {len(triples):,} similarity edges (threshold ≥ {threshold})")
    return triples
