"""
Graph Builder (CLI Orchestrator) — Refactored
==============================================
Builds a Neo4j knowledge graph from data/processed/semantic_cluster.jsonl as the
single source of truth. Does NOT depend on load_abstracts(); article metadata is
derived directly from the chunk file.

Steps
-----
1. Initialize schema (constraints + vector index).
2. Load all chunks from semantic_cluster.jsonl.
3. Derive Article records from unique article_ids found in the chunk file.
4. Embed chunk texts and store embeddings on each chunk dict.
5. Extract entities with SciSpacy (batched via nlp.pipe).
6. (Optional, --compute-similarity) Compute SEMANTIC_SIMILAR edges.
7. Batch-insert in dependency order:
       Articles → Chunks → HAS_CHUNK → Entities → MENTIONS → (SEMANTIC_SIMILAR)

CLI flags
---------
  --limit N            Process only the first N chunks (dry-run / smoke-test).
  --compute-similarity Opt-in: compute + insert SEMANTIC_SIMILAR edges.
                       Skipped by default so the pipeline runs cleanly on
                       Apple Silicon without FAISS.
  --skip-entities      Skip entity extraction (fast smoke-test of schema + nodes).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.graph.batch_insert import (
    insert_articles,
    insert_chunks,
    insert_entities,
    insert_has_chunk_edges,
    insert_mentions_edges,
    insert_semantic_similar_edges,
)
from src.graph.compute_similarity import compute_chunk_embeddings, compute_similarities
from src.graph.entity_extraction import batch_extract
from src.graph.schema import Neo4jConnection, clear_graph, create_constraints_and_indexes, recreate_vector_index
from src.embeddings import get_chunk_embedding_dimension, get_embedding_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_cluster.jsonl"


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_chunks(limit: Optional[int]) -> List[dict]:
    """
    Load chunk records from semantic_cluster.jsonl.
    Stops at *limit* rows when set (None = load all).
    """
    if not CHUNKS_PATH.exists():
        print(f"[build_graph] ERROR: chunk file not found at {CHUNKS_PATH}")
        print("  Run: python -m src.chunking.chunker --strategy semantic_cluster")
        sys.exit(1)

    print(f"[build_graph] Reading {CHUNKS_PATH} …")
    chunks: List[dict] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            chunks.append(json.loads(raw))
            if limit and len(chunks) >= limit:
                print(f"[build_graph] --limit {limit} reached, stopping early.")
                break

    print(f"[build_graph] Loaded {len(chunks):,} chunks.")
    return chunks


def derive_articles(chunks: List[dict]) -> List[dict]:
    """
    Build one Article record per unique article_id found in the chunk list.
    No external data source is consulted — the chunk file is the sole authority.
    Works for both original abstract IDs (pubmed_*) and PubMedQA IDs (pqa_*).
    """
    seen: Dict[str, dict] = {}
    for c in chunks:
        aid = c["article_id"]
        if aid not in seen:
            seen[aid] = {
                "article_id": aid,
                "title": "",          # no title metadata available from the chunk file
                "abstract_text": "",  # full text not stored; chunks carry the content
            }
    return list(seen.values())


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Neo4j knowledge graph from semantic_cluster.jsonl."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N chunks (useful for dry runs).",
    )
    parser.add_argument(
        "--compute-similarity",
        dest="compute_similarity",
        action="store_true",
        default=False,
        help=(
            "Compute and insert SEMANTIC_SIMILAR edges. "
            "Skipped by default (avoids FAISS dependency on Apple Silicon)."
        ),
    )
    parser.add_argument(
        "--skip-entities",
        dest="skip_entities",
        action="store_true",
        default=False,
        help="Skip entity extraction (faster smoke-test of schema + core nodes).",
    )
    parser.add_argument(
        "--recreate-vector-index",
        dest="recreate_vector_index",
        action="store_true",
        default=False,
        help=(
            "Drop and recreate the 'chunk_embedding' vector index with the "
            "currently configured dimension. Use ONLY after changing "
            "EMBEDDING_MODEL_NAME and re-embedding (this run re-embeds all "
            "processed chunks). Never done automatically."
        ),
    )
    parser.add_argument(
        "--similarity-threshold",
        dest="similarity_threshold",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Cosine threshold for SEMANTIC_SIMILAR edges. Defaults to "
            "SEMANTIC_SIMILARITY_THRESHOLD (0.85). Only used with "
            "--compute-similarity."
        ),
    )
    parser.add_argument(
        "--clear",
        dest="clear_graph",
        action="store_true",
        default=False,
        help=(
            "DANGER: delete ALL nodes and relationships before building "
            "(clean rebuild). Constraints and indexes survive. Use for the "
            "submission rebuild to remove stale oversized-graph data."
        ),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  GraphRAG — Neo4j Graph Builder")
    print("=" * 62)

    # ── Step 1: Schema ────────────────────────────────────────────
    print("\n[1/6] Initialising Neo4j schema …")
    try:
        with Neo4jConnection() as driver:
            if args.recreate_vector_index:
                print("[build_graph] --recreate-vector-index: explicit recreation requested.")
                create_constraints_and_indexes(driver)
                recreate_vector_index(driver)
            else:
                create_constraints_and_indexes(driver)
            if args.clear_graph:
                print("[build_graph] --clear: removing all existing graph data …")
                clear_graph(driver)
    except Exception as exc:
        print(f"[build_graph] Cannot connect to Neo4j: {exc}")
        print("  Check that docker-compose is running and .env credentials are correct.")
        sys.exit(1)

    # ── Step 2: Load chunks ───────────────────────────────────────
    print("\n[2/6] Loading chunks …")
    chunks = load_chunks(args.limit)
    if not chunks:
        print("[build_graph] No chunks loaded — nothing to do.")
        sys.exit(0)

    # ── Step 3: Derive articles from chunk file ───────────────────
    print("\n[3/6] Deriving Article nodes from chunk file …")
    articles = derive_articles(chunks)

    has_chunk_pairs: List[Tuple[str, str]] = [
        (c["article_id"], c["chunk_id"]) for c in chunks
    ]

    # ── Step 4: Embed chunk texts (always needed for vector index) ─
    print("\n[4/6] Embedding chunk texts …")
    emb_cfg = get_embedding_config()
    print(
        f"[build_graph] Retrieval model: {emb_cfg['retrieval_model']} "
        f"({emb_cfg['dimension']}D, experiment {emb_cfg['experiment']}). "
        f"PDF baseline is all-MiniLM-L6-v2 (384D). Changing the model "
        f"requires rebuilding/reindexing embeddings and the vector index."
    )
    texts = [c["text"] for c in chunks]
    embeddings = compute_chunk_embeddings(texts)
    assert embeddings.shape[1] == get_chunk_embedding_dimension(), (
        f"Embedding dim {embeddings.shape[1]} != configured "
        f"{get_chunk_embedding_dimension()}"
    )
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb.tolist()   # store as plain float list for Neo4j

    # ── Step 5: Entity extraction ─────────────────────────────────
    entities_to_insert: List[dict] = []
    mentions_pairs: List[Tuple[str, str]] = []

    if args.skip_entities:
        print("\n[5/6] Skipping entity extraction (--skip-entities).")
    else:
        print("\n[5/6] Extracting entities …")
        chunk_entities = batch_extract(chunks)

        unique_entities: Dict[str, dict] = {}
        for chunk_id, ent_list in chunk_entities.items():
            for ent in ent_list:
                name = ent["name"]
                if name not in unique_entities:
                    unique_entities[name] = ent
                mentions_pairs.append((chunk_id, name))

        entities_to_insert = list(unique_entities.values())

    # ── Step 6: Similarity (opt-in) ───────────────────────────────
    similarity_triples: List[Tuple[str, str, float]] = []

    if args.compute_similarity:
        print("\n[6/6] Computing semantic similarity edges …")
        threshold = args.similarity_threshold
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            print(f"[build_graph] ERROR: --similarity-threshold must be in [0, 1], got {threshold}")
            sys.exit(1)
        similarity_triples = compute_similarities(chunks, embeddings, threshold=threshold)
    else:
        print("\n[6/6] Skipping similarity computation (use --compute-similarity to enable).")

    # ── Pre-insertion summary ─────────────────────────────────────
    print("\n" + "─" * 62)
    print("  Pre-insertion counts")
    print("─" * 62)
    print(f"  Articles:        {len(articles):>8,}")
    print(f"  Chunks:          {len(chunks):>8,}")
    print(f"  HAS_CHUNK edges: {len(has_chunk_pairs):>8,}")
    print(f"  Entities:        {len(entities_to_insert):>8,}")
    print(f"  MENTIONS edges:  {len(mentions_pairs):>8,}")
    print(f"  Similarity edges:{len(similarity_triples):>8,}")
    print("─" * 62)

    # ── Step 7: Batch insert ─────────────────────────────────────
    print("\n[7/7] Inserting data into Neo4j …")
    with Neo4jConnection() as driver:
        insert_articles(driver, articles)
        insert_chunks(driver, chunks)
        insert_has_chunk_edges(driver, has_chunk_pairs)

        if entities_to_insert:
            insert_entities(driver, entities_to_insert)
        if mentions_pairs:
            insert_mentions_edges(driver, mentions_pairs)
        if similarity_triples:
            insert_semantic_similar_edges(driver, similarity_triples)

    print("\n" + "=" * 62)
    print("  Graph building complete!")
    print("=" * 62)
    print(f"  Inserted Articles:        {len(articles):,}")
    print(f"  Inserted Chunks:          {len(chunks):,}")
    print(f"  Inserted HAS_CHUNK:       {len(has_chunk_pairs):,}")
    print(f"  Inserted Entities:        {len(entities_to_insert):,}")
    print(f"  Inserted MENTIONS:        {len(mentions_pairs):,}")
    print(f"  Inserted SEMANTIC_SIMILAR:{len(similarity_triples):,}")
    print("=" * 62)


if __name__ == "__main__":
    main()
