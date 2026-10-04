"""
Phase 5 - GDS PageRank Reranking (Phase 2: configurable, subgraph-scoped).
Uses Neo4j Graph Data Science library to compute PageRank scores
on retrieved chunks and their connected entities for improved ranking.

The projection covers ONLY the retrieved subgraph (candidate chunks +
their MENTIONS entities), never the whole database; the temporary
projection and labels are always cleaned up. Scores are min-max normalized
to [0, 1] via the shared fusion.normalize_scores() helper. GDS iterations
and damping come from settings (PAGERANK_MAX_ITERATIONS,
PAGERANK_DAMPING_FACTOR) with optional per-call overrides. Any failure
degrades gracefully to {} so retrieval continues without PageRank.
"""

import uuid
from typing import List, Dict, Optional

from src.config import settings
from src.retrieval.fusion import normalize_scores


def rerank_with_pagerank(
    driver,
    chunk_ids: List[str],
    max_iterations: Optional[int] = None,
    damping_factor: Optional[float] = None,
) -> Dict[str, float]:
    """
    Rerank retrieved chunks using PageRank from Neo4j GDS.

    Creates a temporary subgraph projection containing:
    - Retrieved chunks
    - Entities mentioned by those chunks
    - MENTIONS relationships between them

    Runs PageRank on this subgraph and returns scores for chunks.

    Args:
        driver: Neo4j driver connection
        chunk_ids: List of chunk IDs to rerank

    Returns:
        Dict mapping chunk_id to normalized PageRank score [0, 1]
        Returns empty dict {} on any error (graceful degradation)
    """

    if not chunk_ids:
        return {}

    iterations = settings.PAGERANK_MAX_ITERATIONS if max_iterations is None else int(max_iterations)
    damping = settings.PAGERANK_DAMPING_FACTOR if damping_factor is None else float(damping_factor)

    # Generate unique projection name to avoid conflicts
    projection_name = f"pagerank_projection_{uuid.uuid4().hex[:8]}"

    try:
        with driver.session() as session:
            # Step 1: Create temporary UNDIRECTED graph projection SCOPED to retrieved chunks
            # Using native projection approach with undirected orientation
            # First, create a temporary subgraph label

            # Tag the nodes we want in projection
            tag_query = """
            MATCH (c:Chunk) WHERE c.chunk_id IN $chunkIds
            SET c:__PageRankTemp__
            WITH collect(c) AS chunks
            UNWIND chunks AS c
            MATCH (c)-[:MENTIONS]->(e:Entity)
            SET e:__PageRankTemp__
            RETURN count(DISTINCT c) + count(DISTINCT e) AS tagged_count
            """

            try:
                tag_result = session.run(tag_query, chunkIds=chunk_ids)
                tag_record = tag_result.single()
                print(f"[gds_rerank] Tagged {tag_record['tagged_count']} nodes for projection")
            except Exception as e:
                print(f"[gds_rerank] WARNING: Failed to tag nodes: {e}")
                return {}

            # Create projection with UNDIRECTED orientation
            projection_query = """
            CALL gds.graph.project(
                $projectionName,
                '__PageRankTemp__',
                {
                    MENTIONS: {
                        orientation: 'UNDIRECTED'
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """

            try:
                result = session.run(projection_query, projectionName=projection_name)
                record = result.single()
                if record:
                    print(f"[gds_rerank] Created UNDIRECTED scoped projection '{projection_name}' for {len(chunk_ids)} requested chunks")
                    print(f"[gds_rerank] Projected {record['nodeCount']} nodes and {record['relationshipCount']} relationships")
            except Exception as e:
                print(f"[gds_rerank] WARNING: Failed to create graph projection: {e}")
                # Clean up tags
                session.run("MATCH (n:__PageRankTemp__) REMOVE n:__PageRankTemp__")
                return {}

            # Step 2: Run PageRank algorithm with diagnostic information
            pagerank_query = """
            CALL gds.pageRank.stream($projectionName, {
                maxIterations: $maxIterations,
                dampingFactor: $dampingFactor
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE node:Chunk AND node.chunk_id IN $chunkIds
            RETURN node.chunk_id AS chunk_id, score
            """

            # First, get degree information for diagnostics
            degree_query = """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunkIds
            OPTIONAL MATCH (c)-[r_out:MENTIONS]->()
            OPTIONAL MATCH (c)<-[r_in:MENTIONS]-()
            WITH c.chunk_id AS chunk_id,
                 count(DISTINCT r_out) AS out_degree,
                 count(DISTINCT r_in) AS in_degree
            RETURN chunk_id,
                   out_degree,
                   in_degree,
                   out_degree + in_degree AS total_degree
            ORDER BY total_degree DESC
            """

            try:
                # Get degree info for diagnostics
                degree_result = session.run(degree_query, chunkIds=chunk_ids)
                degree_records = list(degree_result)

                if degree_records:
                    print(f"[gds_rerank] Graph structure for {len(degree_records)} chunks:")
                    for rec in degree_records[:5]:  # Show first 5
                        print(f"  {rec['chunk_id']}: in={rec['in_degree']}, out={rec['out_degree']}, total={rec['total_degree']}")

                # Run PageRank
                result = session.run(
                    pagerank_query,
                    projectionName=projection_name,
                    chunkIds=chunk_ids,
                    maxIterations=iterations,
                    dampingFactor=damping,
                )

                # Collect scores
                pagerank_scores = {}
                raw_scores = {}
                for record in result:
                    pagerank_scores[record["chunk_id"]] = record["score"]
                    raw_scores[record["chunk_id"]] = record["score"]

                # Print raw scores for diagnosis
                if raw_scores:
                    print(f"[gds_rerank] Raw PageRank scores:")
                    for chunk_id, score in sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  {chunk_id}: {score:.6f}")

                # Normalize scores to [0, 1] range (shared helper: identical
                # normalization to the fusion path, so scales stay comparable)
                if pagerank_scores:
                    normalized_scores = normalize_scores(pagerank_scores)

                    print(f"[gds_rerank] Computed PageRank for {len(normalized_scores)} chunks:")
                    for chunk_id, score in sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {chunk_id}: {score:.4f}")
                    return normalized_scores
                else:
                    print(f"[gds_rerank] WARNING: No PageRank scores returned")
                    return {}

            except Exception as e:
                print(f"[gds_rerank] WARNING: Failed to run PageRank: {e}")
                return {}

    except Exception as e:
        print(f"[gds_rerank] WARNING: Unexpected error in rerank_with_pagerank: {e}")
        return {}

    finally:
        # Step 3: Always clean up the projection and temp labels
        try:
            with driver.session() as session:
                drop_query = "CALL gds.graph.drop($projectionName) YIELD graphName"
                session.run(drop_query, projectionName=projection_name)
                print(f"[gds_rerank] Dropped projection '{projection_name}'")

                # Remove temporary labels
                session.run("MATCH (n:__PageRankTemp__) REMOVE n:__PageRankTemp__")
                print(f"[gds_rerank] Cleaned up temporary node labels")
        except Exception as e:
            # Log but don't raise - cleanup failure shouldn't break the pipeline
            print(f"[gds_rerank] WARNING: Failed to cleanup: {e}")
