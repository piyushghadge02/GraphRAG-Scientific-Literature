"""
Task 1 — Vector Search
Queries Neo4j's native vector index to retrieve similar chunks by embedding.
Uses CALL db.index.vector.queryNodes for efficient similarity search.

Phase 1: dimension-agnostic. The query embedding dimension must match the
dimension of the configured retrieval model / Neo4j vector index
(PDF baseline: 384D). Pass expected_dim to enforce it explicitly;
a mismatch means stored embeddings were built with a different model and
require rebuild/reindex.
"""

from typing import List, Optional


def vector_search(
    driver,
    query_embedding: List[float],
    top_k: int = 5,
    expected_dim: Optional[int] = None,
) -> List[dict]:
    """
    Query Neo4j vector index 'chunk_embedding' to retrieve top-k similar chunks.

    Args:
        driver: Neo4j driver connection
        query_embedding: Query embedding as a list of floats. Its dimension
            must match the configured vector index (PDF baseline: 384 for
            all-MiniLM-L6-v2; custom models use their own dimension).
        top_k: Number of top results to return (default: 5)
        expected_dim: Optional dimension to validate against. When given and
            len(query_embedding) != expected_dim, a ValueError is raised with
            a rebuild/reindex hint.

    Returns:
        List of dicts with keys: chunk_id, text, score
        Sorted by score descending (highest similarity first)
    """
    if expected_dim is not None and len(query_embedding) != expected_dim:
        raise ValueError(
            f"Query embedding dimension {len(query_embedding)} does not match "
            f"expected dimension {expected_dim}. The query embedding must come "
            f"from the configured retrieval model; changing the model requires "
            f"rebuilding/reindexing stored embeddings and the vector index."
        )
    query = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $query_embedding)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id, node.text AS text, score
    """

    results = []
    with driver.session() as session:
        result = session.run(query, top_k=top_k, query_embedding=query_embedding)
        for record in result:
            results.append({
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "score": record["score"]
            })

    # Already sorted by score descending from Neo4j, but explicitly ensure it
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
