"""
Task 1 — Schema + Connection
Neo4j connection management using a context manager, and schema creation
(constraints and vector index) for Article, Chunk, and Entity.

PDF-required baseline (MUST remain intact):
    Chunk.embedding: LIST<FLOAT>, 384 dimensions (all-MiniLM-L6-v2)
    vector index "chunk_embedding": dimensions=384, similarity=cosine

Phase 1 extension: the vector index dimension is taken from the configured
retrieval embedding model (see src.embeddings). When the default model
(all-MiniLM-L6-v2) is configured this is exactly 384. Changing the model
requires re-embedding stored Chunks and explicitly recreating the vector
index via recreate_vector_index() (or --recreate-vector-index); the index
is NEVER dropped automatically on startup. There is only ONE vector
index; no second index is created.
"""

from neo4j import GraphDatabase
from src.config import settings
from src.embeddings import (
    BASELINE_EMBEDDING_DIM,
    VECTOR_INDEX_NAME,
    get_chunk_embedding_dimension,
    get_retrieval_model_name,
)


class Neo4jConnection:
    """Wrapper class for managing Neo4j driver connection as a context manager."""

    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self._driver = None

    def __enter__(self):
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        return self._driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._driver:
            self._driver.close()


def get_vector_index_config() -> dict:
    """Return the active vector index configuration.

    Dimensions come from the configured retrieval embedding model
    (384 for the PDF-required baseline). Similarity is always cosine.
    """
    dim = get_chunk_embedding_dimension()
    return {
        "name": VECTOR_INDEX_NAME,
        "dimensions": dim,
        "similarity_function": "cosine",
    }


def build_vector_index_cypher() -> tuple[str, str]:
    """Build the CREATE VECTOR INDEX cypher for the configured dimension."""
    cfg = get_vector_index_config()
    cypher = (
        f"CREATE VECTOR INDEX {cfg['name']} IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{ "
        f"  indexConfig: {{ "
        f"    `vector.dimensions`: {int(cfg['dimensions'])}, "
        f"    `vector.similarity_function`: '{cfg['similarity_function']}' "
        f"  }} "
        f"}}"
    )
    desc = (
        f"Vector index on Chunk.embedding "
        f"({cfg['dimensions']} dims, {cfg['similarity_function']} similarity)"
    )
    return cypher, desc


def create_constraints_and_indexes(driver) -> None:
    """Create uniqueness constraints and the vector index idempotently.

    NOTE: CREATE ... IF NOT EXISTS never alters an existing index. After
    changing EMBEDDING_MODEL_NAME, call recreate_vector_index() explicitly
    (after re-embedding) — this function will NOT drop/recreate anything.
    """
    vector_cypher, vector_desc = build_vector_index_cypher()
    model_name = get_retrieval_model_name()
    print(
        f"[schema] Embedding model: {model_name} "
        f"(PDF baseline is all-MiniLM-L6-v2, {BASELINE_EMBEDDING_DIM}D). "
        f"Changing the model requires rebuilding/reindexing embeddings."
    )
    queries = [
        # Uniqueness constraint on Article.article_id
        (
            "CREATE CONSTRAINT article_id_unique IF NOT EXISTS "
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "Uniqueness constraint on Article.article_id"
        ),
        # Uniqueness constraint on Chunk.chunk_id
        (
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "Uniqueness constraint on Chunk.chunk_id"
        ),
        # Uniqueness constraint on Entity.name
        (
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "Uniqueness constraint on Entity.name"
        ),
        # Single vector index on Chunk.embedding (dimension follows the
        # configured retrieval model; 384 for the PDF baseline).
        (vector_cypher, vector_desc)
    ]

    with driver.session() as session:
        for cypher, desc in queries:
            print(f"[schema] Running: {desc} ...")
            try:
                session.run(cypher)
                print(f"[schema] ✓ Success: {desc}")
            except Exception as e:
                print(f"[schema] ✗ Error creating {desc}: {e}")
                raise e


def recreate_vector_index(driver) -> None:
    """Explicitly drop and recreate the single vector index (model change only).

    Run ONLY after changing EMBEDDING_MODEL_NAME and re-embedding all Chunk
    nodes: the old index dimension no longer matches the new embeddings.
    This is never called automatically — invoke it deliberately, e.g. via
    ``python -m src.graph.build_graph --recreate-vector-index``.
    """
    vector_cypher, vector_desc = build_vector_index_cypher()
    drop_cypher = f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS"
    with driver.session() as session:
        print(f"[schema] Dropping vector index '{VECTOR_INDEX_NAME}' (if exists) ...")
        session.run(drop_cypher)
        print(f"[schema] Recreating: {vector_desc} ...")
        session.run(vector_cypher)
        print(f"[schema] ✓ Vector index recreated: {vector_desc}")


def clear_graph(driver, batch_size: int = 20000) -> int:
    """Delete ALL nodes and relationships (clean rebuild only).

    Deletes in batches and consumes each result so every batch actually
    commits (an unconsumed autocommit result may be rolled back on close).
    Constraints and indexes (including the vector index) survive.
    Never called automatically — only via `build_graph --clear`.
    Returns the total number of deleted nodes.
    """
    print("[schema] Clearing all nodes and relationships (batched DETACH DELETE) ...")
    deleted = 0
    with driver.session() as session:
        while True:
            record = session.execute_write(
                lambda tx: tx.run(
                    "MATCH (n) WITH n LIMIT $batch DETACH DELETE n RETURN count(n) AS d",
                    batch=batch_size,
                ).single()
            )
            batch_deleted = record["d"] if record else 0
            deleted += batch_deleted
            if not batch_deleted:
                break
    print(f"[schema] ✓ Graph cleared ({deleted} nodes deleted).")
    return deleted
