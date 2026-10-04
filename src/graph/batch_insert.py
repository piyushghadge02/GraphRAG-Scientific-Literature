"""
Task 3 — Batch Insertion
Provides batch insertion functions for articles, chunks, entities, and edges.
Uses UNWIND + MERGE Cypher patterns, chunked at 500 records per transaction,
wrapped in retry logic with exponential backoff.
"""

import functools
import time
from typing import List, Tuple

BATCH_SIZE = 500


def retry_neo4j(retries: int = 3, backoff_in_seconds: float = 1.0):
    """Decorator to retry Neo4j write transaction functions on transient errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= retries:
                        print(f"[batch_insert] Retry limit reached. Operations failed: {e}")
                        raise e
                    sleep_time = backoff_in_seconds * (2 ** attempt)
                    print(
                        f"[batch_insert] WARNING: Write failed with error: {e}. "
                        f"Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{retries}) ..."
                    )
                    time.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator


def _chunk_list(data: list, size: int):
    """Yield successive chunks from data list."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


@retry_neo4j()
def insert_articles(driver, articles: List[dict]) -> None:
    """Insert Article nodes using UNWIND + MERGE in batches."""
    # ON MATCH SET keeps re-runs idempotent AND refreshing: rebuilding after a
    # source or embedding-model change updates stored content instead of
    # silently keeping stale values.
    query = """
    UNWIND $batch AS art
    MERGE (a:Article {article_id: art.article_id})
    ON CREATE SET a.title = art.title,
                  a.abstract_text = art.abstract_text
    ON MATCH SET a.title = art.title,
                 a.abstract_text = art.abstract_text
    """
    total = len(articles)
    print(f"[batch_insert] Inserting {total} Article nodes ...")

    with driver.session() as session:
        for chunk in _chunk_list(articles, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed Article insertions.")


@retry_neo4j()
def insert_chunks(driver, chunks: List[dict]) -> None:
    """Insert Chunk nodes (including vector embeddings) in batches."""
    # ON MATCH SET is required so re-running the build after an embedding-model
    # change actually refreshes stored vectors (MERGE alone would keep stale
    # embeddings). Uniqueness is still enforced by the chunk_id constraint.
    query = """
    UNWIND $batch AS chk
    MERGE (c:Chunk {chunk_id: chk.chunk_id})
    ON CREATE SET c.article_id = chk.article_id,
                  c.text = chk.text,
                  c.embedding = chk.embedding,
                  c.strategy = chk.strategy
    ON MATCH SET c.article_id = chk.article_id,
                 c.text = chk.text,
                 c.embedding = chk.embedding,
                 c.strategy = chk.strategy
    """
    total = len(chunks)
    print(f"[batch_insert] Inserting {total} Chunk nodes ...")

    with driver.session() as session:
        for chunk in _chunk_list(chunks, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed Chunk insertions.")


@retry_neo4j()
def insert_entities(driver, entities: List[dict]) -> None:
    """Insert Entity nodes in batches."""
    query = """
    UNWIND $batch AS ent
    MERGE (e:Entity {name: ent.name})
    ON CREATE SET e.type = ent.type
    """
    total = len(entities)
    print(f"[batch_insert] Inserting {total} Entity nodes ...")

    with driver.session() as session:
        for chunk in _chunk_list(entities, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed Entity insertions.")


@retry_neo4j()
def insert_has_chunk_edges(driver, pairs: List[Tuple[str, str]]) -> None:
    """Insert HAS_CHUNK relationships between Articles and Chunks in batches."""
    # Convert list of tuples to list of dicts for UNWIND friendliness
    batch_data = [{"article_id": p[0], "chunk_id": p[1]} for p in pairs]
    query = """
    UNWIND $batch AS link
    MATCH (a:Article {article_id: link.article_id})
    MATCH (c:Chunk {chunk_id: link.chunk_id})
    MERGE (a)-[:HAS_CHUNK]->(c)
    """
    total = len(pairs)
    print(f"[batch_insert] Inserting {total} HAS_CHUNK edges ...")

    with driver.session() as session:
        for chunk in _chunk_list(batch_data, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed HAS_CHUNK edges.")


@retry_neo4j()
def insert_mentions_edges(driver, pairs: List[Tuple[str, str]]) -> None:
    """Insert MENTIONS relationships between Chunks and Entities in batches."""
    batch_data = [{"chunk_id": p[0], "entity_name": p[1]} for p in pairs]
    query = """
    UNWIND $batch AS link
    MATCH (c:Chunk {chunk_id: link.chunk_id})
    MATCH (e:Entity {name: link.entity_name})
    MERGE (c)-[:MENTIONS]->(e)
    """
    total = len(pairs)
    print(f"[batch_insert] Inserting {total} MENTIONS edges ...")

    with driver.session() as session:
        for chunk in _chunk_list(batch_data, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed MENTIONS edges.")


@retry_neo4j()
def insert_semantic_similar_edges(driver, triples: List[Tuple[str, str, float]]) -> None:
    """Insert SEMANTIC_SIMILAR relationships between Chunk pairs in batches."""
    batch_data = [{"chunk_id_1": t[0], "chunk_id_2": t[1], "similarity": t[2]} for t in triples]
    query = """
    UNWIND $batch AS link
    MATCH (c1:Chunk {chunk_id: link.chunk_id_1})
    MATCH (c2:Chunk {chunk_id: link.chunk_id_2})
    MERGE (c1)-[r:SEMANTIC_SIMILAR]->(c2)
    SET r.similarity = link.similarity
    """
    total = len(triples)
    print(f"[batch_insert] Inserting {total} SEMANTIC_SIMILAR edges ...")

    with driver.session() as session:
        for chunk in _chunk_list(batch_data, BATCH_SIZE):
            session.execute_write(lambda tx: tx.run(query, batch=chunk))
    print("[batch_insert] ✓ Completed SEMANTIC_SIMILAR edges.")
