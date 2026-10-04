"""
Task 2 — Graph Expansion (Phase 2: evidence-rich, configurable expansion).
Expands initial retrieved chunks by traversing through shared entities.
Traverses (:Chunk)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(:Chunk) to find related chunks.
Supports depth-1 (default) and depth-2 (PDF maximum traversal depth).

Phase 2 improvements:
- all limits/thresholds come from settings (GRAPH_MAX_DEPTH,
  GRAPH_MAX_EXPANDED_CHUNKS, GRAPH_MIN_SHARED_ENTITIES, GRAPH_MIN_SIMILARITY)
  with optional per-call overrides; depth is clamped to the PDF limit of 2.
- each candidate carries graph evidence: shared_entities (distinct entities
  linking it to seeds), depth (1 or 2), path_count (seed->...->candidate
  paths supporting it).
- defensive Python-side dedupe by chunk_id (Cypher aggregation already
  yields one row per candidate) and deterministic re-sort after filtering.

Seed chunks may appear in the results with their shared-entity counts; the
retrieval layer merges them into the existing seed candidates.
"""

from typing import List, Optional
import numpy as np

from src.config import settings

# PDF maximum traversal depth for the required Chunk->Entity->Chunk traversal.
PDF_MAX_DEPTH = 2


def _resolve(name: str, value, cast):
    """Per-call override wins; otherwise the configured default."""
    if value is not None:
        return cast(value)
    return cast(getattr(settings, name))


def expand_via_entities(
    driver,
    seed_chunk_ids: List[str],
    query_embedding: Optional[List[float]] = None,
    max_depth: Optional[int] = None,
    max_candidates: Optional[int] = None,
    min_similarity: Optional[float] = None,
    min_shared_entities: Optional[int] = None,
) -> List[dict]:
    """
    Expand initial seed chunks by traversing entity mentions with query-aware filtering.

    Starting from seed_chunk_ids, finds other chunks that mention the same entities.
    Returns candidate chunks with graph evidence used by fusion reranking.

    Args:
        driver: Neo4j driver connection
        seed_chunk_ids: List of initial chunk IDs to expand from
        query_embedding: Normalized query embedding for semantic filtering
        max_depth: Traversal depth, 1 or 2 (default: GRAPH_MAX_DEPTH, max: 2 per PDF).
                   Depth 1: seed chunk -> entity -> other chunk
                   Depth 2: seed chunk -> entity -> chunk -> entity -> chunk
        max_candidates: Maximum candidates to retrieve (default: GRAPH_MAX_EXPANDED_CHUNKS)
        min_similarity: Minimum cosine similarity to query for expanded candidates
            (default: GRAPH_MIN_SIMILARITY)
        min_shared_entities: Minimum distinct shared entities (default:
            GRAPH_MIN_SHARED_ENTITIES). Raises the bar against irrelevant
            single-entity expansion.

    Returns:
        List of dicts with keys: chunk_id, text, embedding, shared_entities,
        depth, path_count. Deduplicated by chunk_id, sorted by
        (shared_entities, path_count) descending.
    """

    if not seed_chunk_ids:
        return []

    depth = _resolve("GRAPH_MAX_DEPTH", max_depth, int)
    limit = _resolve("GRAPH_MAX_EXPANDED_CHUNKS", max_candidates, int)
    similarity_floor = _resolve("GRAPH_MIN_SIMILARITY", min_similarity, float)
    shared_floor = _resolve("GRAPH_MIN_SHARED_ENTITIES", min_shared_entities, int)
    degree_cap = int(getattr(settings, "GRAPH_MAX_ENTITY_DEGREE", 0) or 0)

    if depth not in (1, 2):
        raise ValueError(f"max_depth must be 1 or 2 (PDF limit), got {depth}")
    if shared_floor < 1:
        raise ValueError(f"min_shared_entities must be >= 1, got {shared_floor}")

    # Optional hub-entity guard, wired as the $deg_cap parameter below.
    # Off (0) by default, in which case the clause is always true and
    # traversal is exactly the assignment's Chunk → Entity → Chunk pattern.
    if depth == 1:
        # Depth 1: chunks sharing entities with seed chunks.
        query = """
        MATCH (seed:Chunk)-[:MENTIONS]->(ent:Entity)<-[:MENTIONS]-(candidate:Chunk)
        WHERE seed.chunk_id IN $seed_chunk_ids
          AND NOT candidate.chunk_id = seed.chunk_id
          AND ( $deg_cap <= 0 OR COUNT { (ent)<-[:MENTIONS]-() } <= $deg_cap )
        WITH candidate.chunk_id AS chunk_id, candidate.text AS text,
              candidate.embedding AS embedding,
              COUNT(DISTINCT ent) AS shared_entities,
              COUNT(*) AS path_count
        WHERE shared_entities >= $min_shared
        RETURN chunk_id, text, embedding, shared_entities, path_count, 1 AS depth
        ORDER BY shared_entities DESC, path_count DESC
        LIMIT $max_candidates
        """

    else:
        # Depth 2: seed -> entity -> chunk -> entity -> candidate.
        # Transitive discovery through intermediate chunks.
        query = """
        MATCH (seed:Chunk)-[:MENTIONS]->(ent1:Entity)<-[:MENTIONS]-(intermediate:Chunk)
        WHERE seed.chunk_id IN $seed_chunk_ids
          AND NOT intermediate.chunk_id = seed.chunk_id
          AND ( $deg_cap <= 0 OR COUNT { (ent1)<-[:MENTIONS]-() } <= $deg_cap )
        MATCH (intermediate)-[:MENTIONS]->(ent2:Entity)<-[:MENTIONS]-(candidate:Chunk)
        WHERE NOT candidate.chunk_id = seed.chunk_id
          AND NOT candidate.chunk_id = intermediate.chunk_id
          AND ( $deg_cap <= 0 OR COUNT { (ent2)<-[:MENTIONS]-() } <= $deg_cap )
        WITH candidate.chunk_id AS chunk_id, candidate.text AS text,
              candidate.embedding AS embedding,
              COUNT(DISTINCT ent1) + COUNT(DISTINCT ent2) AS shared_entities,
              COUNT(*) AS path_count
        WHERE shared_entities >= $min_shared
        RETURN chunk_id, text, embedding, shared_entities, path_count, 2 AS depth
        ORDER BY shared_entities DESC, path_count DESC
        LIMIT $max_candidates
        """

    results = []
    with driver.session() as session:
        result = session.run(
            query,
            seed_chunk_ids=seed_chunk_ids,
            max_candidates=limit,
            min_shared=shared_floor,
            deg_cap=degree_cap,
        )

        # Normalize query embedding for cosine similarity
        query_arr = None
        if query_embedding is not None:
            query_arr = np.array(query_embedding, dtype=np.float32)
            norm_q = np.linalg.norm(query_arr)
            if norm_q > 0:
                query_arr = query_arr / norm_q

        seen = set()
        for record in result:
            chunk_id = record["chunk_id"]
            if chunk_id in seen:
                continue  # defensive: one entry per chunk

            # Apply semantic filtering if query embedding provided; this keeps
            # weakly connected expansion from overwhelming vector relevance.
            if query_arr is not None and record["embedding"] is not None:
                cand_arr = np.array(record["embedding"], dtype=np.float32)
                norm_c = np.linalg.norm(cand_arr)
                if norm_c > 0:
                    cand_arr = cand_arr / norm_c
                    similarity = float(np.dot(query_arr, cand_arr))

                    # Filter out candidates below similarity threshold
                    if similarity < similarity_floor:
                        continue

            seen.add(chunk_id)
            results.append({
                "chunk_id": chunk_id,
                "text": record["text"],
                "embedding": record["embedding"],
                "shared_entities": int(record["shared_entities"]),
                "depth": int(record.get("depth", depth)),
                "path_count": int(record.get("path_count", 1)),
            })

    # Re-sort after Python-side filtering (Neo4j order may have gaps now).
    results.sort(key=lambda r: (r["shared_entities"], r["path_count"]), reverse=True)
    return results
