"""
Phase 5 - Retrieval with PageRank Integration (Phase 2: clean, configurable blend).
Wraps existing retrieve() function with optional PageRank reranking.
"""

from typing import List, Dict, Optional
from src.config import settings
from src.retrieval.retrieve import retrieve
from src.retrieval.gds_rerank import rerank_with_pagerank


def retrieve_with_pagerank(
    driver,
    embedding_model,
    query: str,
    top_k: int = 5,
    expand_graph: bool = True,
    use_pagerank: bool = True,
    pagerank_weight: Optional[float] = None,
    adaptive_enabled: Optional[bool] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> List[Dict]:
    """
    Enhanced retrieval with optional PageRank reranking.

    Pipeline:
    1. Call existing retrieve() to get candidates with combined_score
    2. If use_pagerank=True, compute PageRank scores for retrieved chunks
       (subgraph-scoped GDS projection, normalized to [0, 1])
    3. Compute final_score = (1 - w) * combined_score + w * pagerank,
       with w from PAGERANK_WEIGHT (default 0.15)
    4. Re-sort by final_score and return top_k

    Args:
        driver: Neo4j driver connection
        embedding_model: SentenceTransformer model
        query: User's search query
        top_k: Number of final results to return
        expand_graph: Whether to use graph expansion (default: True)
        use_pagerank: Whether to apply PageRank reranking (default: True).
            When False (or ENABLE_PAGERANK=false), the path is identical to
            plain retrieve() plus final_score = combined_score.
        pagerank_weight: Blend weight for PageRank (default: PAGERANK_WEIGHT).
        adaptive_enabled: Override for ADAPTIVE_RETRIEVAL_ENABLED (default: config).
        alpha/beta/gamma: Fusion weight overrides passed through to retrieve().

    Returns:
        List of chunks with fields: chunk_id, text, vector_score, shared_entities,
        combined_score, pagerank_score (if used), final_score
    """

    # Step 1: Get initial retrieval results using existing pipeline
    # Retrieve more candidates than needed to allow PageRank to reorder
    retrieval_top_k = top_k * 2 if use_pagerank else top_k

    results = retrieve(
        driver=driver,
        embedding_model=embedding_model,
        query=query,
        top_k=retrieval_top_k,
        expand_graph=expand_graph,
        adaptive_enabled=adaptive_enabled,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    if not results:
        return []

    # Step 2: Apply PageRank reranking if enabled
    if use_pagerank:
        chunk_ids = [r["chunk_id"] for r in results]
        pagerank_scores = rerank_with_pagerank(driver, chunk_ids)

        # Final score: clean convex blend of the fused retrieval score and
        # the normalized structural (PageRank) score. Both terms are in
        # [0, 1], so the weight directly controls the structural contribution.
        weight = settings.PAGERANK_WEIGHT if pagerank_weight is None else pagerank_weight
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"pagerank_weight must be in [0, 1], got {weight}")
        for result in results:
            chunk_id = result["chunk_id"]
            combined = result["combined_score"]

            # Get PageRank score (default to 0.0 if not found)
            pr_score = pagerank_scores.get(chunk_id, 0.0)

            final = (1.0 - weight) * combined + weight * pr_score

            result["pagerank_score"] = pr_score
            result["pagerank_weight"] = float(weight)
            result["final_score"] = final

        # Re-sort by final_score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        print(f"[retrieve_with_pagerank] Applied PageRank reranking to {len(results)} chunks")
    else:
        # No PageRank: final_score = combined_score
        for result in results:
            result["pagerank_score"] = 0.0
            result["final_score"] = result["combined_score"]

    # Step 3: Return top_k results with detailed diagnostics
    final_results = results[:top_k]

    # Print detailed diagnostics for each final chunk
    print("\n--- FINAL RETRIEVAL RESULTS ---")
    for i, result in enumerate(final_results, 1):
        source = "vector" if result.get("shared_entities", 0) == 0 else "graph"
        text_sample = result["text"][:150].replace('\n', ' ')
        print(f"\n[{i}] {result['chunk_id']}")
        print(f"    Source: {source}")
        print(f"    Vector similarity: {result['vector_score']:.4f}")
        print(f"    Shared entities: {result.get('shared_entities', 0)}")
        print(f"    PageRank: {result.get('pagerank_score', 0.0):.4f}")
        print(f"    Final score: {result['final_score']:.4f}")
        print(f"    Text: {text_sample}...")
    print("-----------------------------------\n")

    return final_results
