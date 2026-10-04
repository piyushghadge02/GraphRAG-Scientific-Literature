"""
Task 4 — Unified Retrieval (Phase 2: adaptive graph-aware pipeline).
Orchestrates the RAG retrieval pipeline:
1. Embed query
2. Vector search (with wider pool)
3. Adaptive retrieval decision (expand vs vector-focused)
4. Optional graph expansion (seed -> entity -> chunk, depth <= 2)
5. Candidate deduplication
6. 3-term vector + graph + entity scoring
7. Rerank and return top-k
(PageRank remains one layer up in retrieve_with_pagerank.)
"""

from typing import List, Optional

from src.config import settings
from src.embeddings import (
    get_chunk_embedding_dimension,
    get_embedding_dimension,
    get_retrieval_model_name,
    validate_embedding_dim,
)
from src.retrieval.adaptive import expansion_decision
from src.retrieval.fusion import normalize_weights, rank_candidates_fused
from src.retrieval.graph_expand import expand_via_entities
from src.retrieval.vector_search import vector_search


def retrieve(
    driver,
    embedding_model,
    query: str,
    top_k: int = 5,
    expand_graph: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    adaptive_enabled: Optional[bool] = None,
) -> List[dict]:
    """
    Unified retrieve function for RAG pipeline.

    Steps:
    1. Embed the query using the same embedding model as chunks
    2. Call vector_search with top_k * 3 to get a wider candidate pool
    3. Adaptive decision: skip graph expansion when vector retrieval is
       already high-confidence (deterministic, logged)
    4. If expanding, expand via entity mentions and merge candidates
    5. Dedupe by chunk_id (keep higher vector_score if duplicate)
    6. Compute fused scores (alpha*vector + beta*entity + gamma*proximity)
    7. Sort by combined_score descending and return top_k

    Args:
        driver: Neo4j driver connection
        embedding_model: SentenceTransformer model for embedding the query.
            Must produce the configured retrieval-model dimension
            (PDF baseline: 384D for all-MiniLM-L6-v2).
        query: User's search query
        top_k: Number of final results to return (default: 5)
        expand_graph: Whether to allow graph expansion (default: True).
            Hard master switch: False always means vector-only. Also gated by
            GRAPH_EXPANSION_ENABLED.
        alpha/beta/gamma: Fusion weight overrides (defaults: ALPHA/BETA/GAMMA
            settings; rescaled to sum to 1). See src/retrieval/fusion.py.
        adaptive_enabled: Override for ADAPTIVE_RETRIEVAL_ENABLED (default: config).

    Returns:
        List of dicts with keys: chunk_id, text, vector_score, shared_entities,
        depth, path_count, combined_score (plus score_breakdown/score_explanation)
        Sorted by combined_score descending, limited to top_k results
    """

    # Step 1: Embed the query and normalize it for cosine similarity.
    # The externally supplied model must be compatible with the configured
    # retrieval dimension. When it reports its own dimension, compare it via
    # the central provider before encoding; the query embedding itself is
    # validated again below. A mismatch means the index was built with a
    # different model (rebuild/reindex required).
    try:
        supplied_dim = get_embedding_dimension(embedder=embedding_model)
    except Exception:
        supplied_dim = None
    if supplied_dim is not None and supplied_dim != get_chunk_embedding_dimension():
        raise ValueError(
            f"Supplied embedding model dimension ({supplied_dim}) does not match "
            f"the configured retrieval model '{get_retrieval_model_name()}' "
            f"({get_chunk_embedding_dimension()}D). Pass the model for the "
            f"configured EMBEDDING_MODEL_NAME; changing the model requires "
            f"rebuilding/reindexing stored embeddings and the vector index."
        )
    query_embedding = embedding_model.encode(query).tolist()
    validate_embedding_dim(query_embedding, get_chunk_embedding_dimension())

    import numpy as np
    query_arr = np.array(query_embedding, dtype=np.float32)
    norm_q = np.linalg.norm(query_arr)
    if norm_q > 0:
        query_arr = query_arr / norm_q

    # Step 2: Vector search with wider pool (3x multiplier for pre-reranking)
    wider_pool_size = top_k * 3
    vector_results = vector_search(
        driver,
        query_embedding,
        top_k=wider_pool_size,
        expected_dim=get_chunk_embedding_dimension(),
    )

    # Build initial candidate pool from vector search
    # Key: chunk_id, Value: {text, vector_score, shared_entities=0, depth=0}
    # Seeds sit at graph distance 0 (proximity 1.0 in fusion).
    candidates = {}
    vector_chunk_ids = []

    for result in vector_results:
        chunk_id = result["chunk_id"]
        candidates[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "vector_score": result["score"],
            "shared_entities": 0,  # Will be updated if expand_graph=True
            "depth": 0,
            "path_count": 0,
        }
        vector_chunk_ids.append(chunk_id)

    # Step 3: Adaptive decision + expansion (if enabled)
    expanded_results = []
    entity_count = 0
    adaptive_info = None
    do_expand = bool(expand_graph and settings.GRAPH_EXPANSION_ENABLED and vector_chunk_ids)
    if do_expand:
        adaptive_info = expansion_decision(
            vector_results, query, enabled=adaptive_enabled
        )
        print(f"[adaptive] expand={adaptive_info['expand']} "
              f"({adaptive_info['confidence']})")
        for reason in adaptive_info["reasons"]:
            print(f"[adaptive]   - {reason}")
        do_expand = bool(adaptive_info["expand"])

    if do_expand:
        # Get count of entities connected to seed chunks
        with driver.session() as session:
            ent_res = session.run(
                "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.chunk_id IN $seed_ids RETURN COUNT(DISTINCT e) as count",
                seed_ids=vector_chunk_ids
            )
            entity_row = ent_res.single()
            if entity_row:
                entity_count = entity_row["count"]

        # Expand with query-aware filtering (limits from settings)
        expanded_results = expand_via_entities(
            driver,
            vector_chunk_ids,
            query_embedding=query_embedding,
        )

        for result in expanded_results:
            chunk_id = result["chunk_id"]
            shared_entities = result["shared_entities"]
            depth = int(result.get("depth", 1))
            path_count = int(result.get("path_count", 1))

            if chunk_id in candidates:
                # Chunk already in vector search results, keep higher vector_score.
                # Seeds keep depth 0 but record their shared-entity evidence.
                candidates[chunk_id]["shared_entities"] = shared_entities
                candidates[chunk_id]["path_count"] = path_count
            else:
                # Calculate vector similarity score on the fly using dot product of normalized vectors
                cand_emb = result.get("embedding")
                vector_score = 0.0
                if cand_emb is not None and len(cand_emb) > 0:
                    cand_arr = np.array(cand_emb, dtype=np.float32)
                    norm_c = np.linalg.norm(cand_arr)
                    if norm_c > 0:
                        cand_arr = cand_arr / norm_c
                    vector_score = float(np.dot(query_arr, cand_arr))

                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "vector_score": vector_score,
                    "shared_entities": shared_entities,
                    "depth": depth,
                    "path_count": path_count,
                }

    # Step 4: Compute fused scores for all candidates.
    # final = alpha*vector + beta*entity + gamma*proximity (weights rescaled
    # to sum to 1; defaults reproduce the Phase 1 ranking exactly).
    weights = normalize_weights(alpha, beta, gamma)
    ranked = rank_candidates_fused(
        list(candidates.values()), alpha=weights[0], beta=weights[1], gamma=weights[2]
    )

    # Step 6: Return top_k with detailed diagnostics
    final_top_k = ranked[:top_k]

    # Retrieval strategy label (additive metadata for UI/debugging).
    if not expand_graph or not settings.GRAPH_EXPANSION_ENABLED or not vector_chunk_ids:
        strategy = "vector-only"
    elif adaptive_info is not None and adaptive_info["confidence"] != "n/a" and not adaptive_info["expand"]:
        strategy = "adaptive-vector"
    elif adaptive_info is not None and adaptive_info["confidence"] not in ("n/a", "none") and adaptive_info["expand"]:
        strategy = "adaptive-graph"
    else:
        strategy = "graph"
    for c in final_top_k:
        c["retrieval_strategy"] = strategy

    # Print detailed diagnostic output
    if expand_graph:
        expanded_non_seeds = [r["chunk_id"] for r in expanded_results if r["chunk_id"] not in vector_chunk_ids]
        final_seeds = [c["chunk_id"] for c in final_top_k if c["chunk_id"] in vector_chunk_ids]
        final_exp = [c["chunk_id"] for c in final_top_k if c["chunk_id"] not in vector_chunk_ids]
        max_depth_seen = max((c.get("depth", 0) for c in final_top_k), default=0)

        print("\n--- GRAPH RETRIEVAL DIAGNOSTICS ---")
        print(f"Number of vector seed chunks: {len(vector_chunk_ids)}")
        print(f"Seed chunk IDs: {vector_chunk_ids}")
        if adaptive_info is not None:
            print(f"Adaptive decision: expand={adaptive_info['expand']} "
                  f"({adaptive_info['confidence']})")
        print(f"Graph expansion ran: {do_expand}")
        print(f"Fusion weights (alpha/beta/gamma): "
              f"{weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f}")
        print(f"Number of entities connected to seed chunks: {entity_count}")
        print(f"Number of candidate neighboring chunks found through MENTIONS: {len(expanded_results)}")
        print(f"Number of unique expanded chunks after removing seed chunks: {len(expanded_non_seeds)}")
        print(f"Expanded chunk IDs (first 10): {expanded_non_seeds[:10]}")
        print(f"Max graph depth in final results: {max_depth_seen}")
        print(f"Final number of chunks after reranking: {len(final_top_k)}")
        print(f"How many final chunks came from vector seeds: {len(final_seeds)} (IDs: {final_seeds})")
        print(f"How many final chunks came from graph expansion: {len(final_exp)} (IDs: {final_exp})")
        print("-----------------------------------\n")

    return final_top_k
