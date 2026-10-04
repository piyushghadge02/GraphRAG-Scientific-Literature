"""
Phase 1 — Central embedding provider.

PDF-required baseline (MUST remain intact):
    model:     all-MiniLM-L6-v2
    dimension: 384
    Neo4j Chunk.embedding: LIST<FLOAT>, 384 dims
    vector index: dimensions=384, similarity=cosine

Extension:
    The retrieval embedding model is configurable via EMBEDDING_MODEL_NAME
    (default: all-MiniLM-L6-v2). The semantic-clustering model is configured
    separately via SEMANTIC_CLUSTER_MODEL_NAME (default: all-MiniLM-L6-v2,
    PDF-required — do not change it in Phase 1).

    Changing EMBEDDING_MODEL_NAME requires rebuilding/reindexing all stored
    Chunk embeddings and recreating the Neo4j vector index with the new
    dimension. There is intentionally only ONE vector index ("chunk_embedding").

Dimension handling:
    - For the baseline model, the dimension is the PDF-specified constant 384
      (returned without loading the model).
    - For any other model, the dimension is determined programmatically from
      the loaded SentenceTransformer model (get_sentence_embedding_dimension).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.config import BASELINE_EMBEDDING_MODEL, settings

# PDF-required baseline dimension. Do not change.
BASELINE_EMBEDDING_DIM = 384

# Single Neo4j vector index name (do NOT create a second index in Phase 1).
VECTOR_INDEX_NAME = "chunk_embedding"

# In-process cache of loaded SentenceTransformer models: name -> model.
_MODEL_CACHE: Dict[str, object] = {}


def get_retrieval_model_name() -> str:
    """Return the configured retrieval embedding model name."""
    return settings.EMBEDDING_MODEL_NAME


def get_cluster_model_name() -> str:
    """Return the PDF-required semantic-clustering model name."""
    return settings.SEMANTIC_CLUSTER_MODEL_NAME


def get_embedder(model_name: Optional[str] = None) -> Any:
    """Load (and cache) the SentenceTransformer for *model_name*.

    Args:
        model_name: Model name; defaults to the configured retrieval model.

    The import is lazy so modules importing this provider do not pay the
    sentence-transformers import cost until an embedding is actually needed.
    """
    from sentence_transformers import SentenceTransformer

    name = model_name or get_retrieval_model_name()
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def clear_embedder_cache() -> None:
    """Clear the in-process model cache (mainly useful for tests)."""
    _MODEL_CACHE.clear()


def get_embedding_dimension(
    model_name: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> int:
    """Return the embedding dimension for *model_name*.

    - If *embedder* is given, the dimension is read programmatically from
      ``embedder.get_sentence_embedding_dimension()``.
    - If *model_name* is the PDF baseline, returns 384 without loading
      the model (fast path; the value is required by the PDF).
    - Otherwise the model is loaded and its dimension is read programmatically.

    When the default model is configured, this MUST return 384.
    """
    if embedder is not None:
        dim = embedder.get_sentence_embedding_dimension()
        return int(dim)
    name = model_name or get_retrieval_model_name()
    if name == BASELINE_EMBEDDING_MODEL:
        return BASELINE_EMBEDDING_DIM
    return int(get_embedder(name).get_sentence_embedding_dimension())


def get_chunk_embedding_dimension() -> int:
    """Dimension to use for Chunk embeddings and the Neo4j vector index."""
    return get_embedding_dimension(get_retrieval_model_name())


def validate_embedding_dim(
    embedding,
    expected_dim: Optional[int] = None,
    model_name: Optional[str] = None,
) -> int:
    """Validate that *embedding* has the expected dimension.

    Returns the observed dimension. Raises ValueError on mismatch with a
    message reminding that switching models requires rebuild/reindex.
    """
    expected = expected_dim if expected_dim is not None else get_chunk_embedding_dimension()
    observed = len(embedding)
    if observed != expected:
        label = model_name or get_retrieval_model_name()
        raise ValueError(
            f"Embedding dimension mismatch: got {observed}, expected {expected} "
            f"for model '{label}'. "
            f"Changing the embedding model requires rebuilding/reindexing "
            f"stored embeddings and the Neo4j vector index."
        )
    return observed


def encode_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    embedder: Optional[Any] = None,
    batch_size: int = 256,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Encode *texts* with the configured (or given) model.

    Returns a float32 numpy array of shape (len(texts), dim).
    Validates that the output dimension matches the model actually used.
    """
    name = model_name or get_retrieval_model_name()
    model = embedder if embedder is not None else get_embedder(name)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar, batch_size=batch_size)
    result = np.array(embeddings, dtype=np.float32)
    if result.size == 0:
        return result
    expected = get_embedding_dimension(name, model)
    if result.shape[1] != expected:
        raise ValueError(
            f"Embedding dimension mismatch: got {result.shape[1]}, expected {expected} "
            f"for model '{name}'. "
            f"Changing the embedding model requires rebuilding/reindexing "
            f"stored embeddings and the Neo4j vector index."
        )
    return result


def encode_query(
    query: str,
    model_name: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> List[float]:
    """Encode a single query string; validates against the model actually used."""
    name = model_name or get_retrieval_model_name()
    model = embedder if embedder is not None else get_embedder(name)
    embedding = model.encode(query).tolist()
    validate_embedding_dim(embedding, get_embedding_dimension(name, model), name)
    return embedding


def get_embedding_config() -> dict:
    """Return metadata describing the active embedding configuration.

    The dimension is the PDF-required 384 when the baseline model is
    configured; otherwise it is resolved programmatically. No model is
    loaded solely to build this dict when the baseline is active.
    """
    retrieval_model = get_retrieval_model_name()
    cluster_model = get_cluster_model_name()
    is_baseline = retrieval_model == BASELINE_EMBEDDING_MODEL
    if is_baseline:
        dimension = BASELINE_EMBEDDING_DIM
    else:
        dimension = get_embedding_dimension(retrieval_model)
    return {
        "retrieval_model": retrieval_model,
        "cluster_model": cluster_model,
        "dimension": dimension,
        "is_baseline": is_baseline,
        "experiment": "A (baseline)" if is_baseline else "B (custom)",
        "baseline_model": BASELINE_EMBEDDING_MODEL,
        "baseline_dim": BASELINE_EMBEDDING_DIM,
        "vector_index": {
            "name": VECTOR_INDEX_NAME,
            "dimensions": dimension,
            "similarity_function": "cosine",
        },
        "rebuild_required_on_model_change": True,
    }
