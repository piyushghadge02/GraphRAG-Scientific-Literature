#!/usr/bin/env python3
"""
Execute the full data pipeline end-to-end for PubMedQA integration testing.
Steps:
1. Verify fixed_token.jsonl exists and contains 5000 chunks
2. Check semantic_cluster.jsonl
3. Build Neo4j graph
4. Verify PubMed corpus was inserted into Neo4j
5. Generate and verify evaluation questions
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.schema import Neo4jConnection


def step1_verify_chunking():
    """Step 1: Verify fixed_token.jsonl contains 5000 chunks"""
    print("\n" + "="*60)
    print("STEP 1: Verify Chunking Data")
    print("="*60)

    fixed_token_path = PROJECT_ROOT / "data" / "processed" / "fixed_token.jsonl"

    if not fixed_token_path.exists():
        print(f"[FAIL] fixed_token.jsonl not found at {fixed_token_path}")
        return False

    # Count total chunks and distinct articles
    total_chunks = 0
    article_ids = set()

    with open(fixed_token_path, "r") as f:
        for line in f:
            if line.strip():
                total_chunks += 1
                data = json.loads(line)
                article_ids.add(data["article_id"])

    print(f"[OK] Total chunks in fixed_token.jsonl: {total_chunks}")
    print(f"[OK] Distinct articles: {len(article_ids)}")

    if total_chunks == 0:
        print("[FAIL] No chunks found!")
        return False

    if total_chunks < 4000:
        print(f"[WARN] Expected ~5000 chunks, found {total_chunks}")
        return False

    print(f"[OK] Chunk count within expected range")
    return True


def step2_check_semantic_cluster():
    """Step 2: Check if semantic_cluster.jsonl exists"""
    print("\n" + "="*60)
    print("STEP 2: Check Semantic Clustering Data")
    print("="*60)

    semantic_path = PROJECT_ROOT / "data" / "processed" / "semantic_cluster.jsonl"

    if not semantic_path.exists():
        print(f"[WARN] semantic_cluster.jsonl not found - build_graph requires this file")
        print(f"   Location: {semantic_path}")
        return False

    # Count chunks
    total_chunks = 0
    article_ids = set()

    with open(semantic_path, "r") as f:
        for line in f:
            if line.strip():
                total_chunks += 1
                data = json.loads(line)
                article_ids.add(data["article_id"])

    print(f"[OK] Total chunks in semantic_cluster.jsonl: {total_chunks}")
    print(f"[OK] Distinct articles: {len(article_ids)}")

    if total_chunks == 0:
        print("[WARN] No chunks found in semantic_cluster.jsonl")
        return False

    if total_chunks < 4000:
        print(f"[WARN] Expected ~5000 chunks, found {total_chunks}")
        return False

    print(f"[OK] Chunk count within expected range")
    return True


def step3_verify_neo4j():
    """Step 3: Verify Neo4j connection and basic schema"""
    print("\n" + "="*60)
    print("STEP 3: Verify Neo4j Connection")
    print("="*60)

    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 as ping")
                result.single()
        print("[OK] Neo4j connection successful")
        return True
    except Exception as e:
        print(f"[FAIL] Neo4j connection failed: {e}")
        return False


def step4_verify_corpus_in_graph():
    """Step 4: Verify PubMed corpus was inserted into Neo4j"""
    print("\n" + "="*60)
    print("STEP 4: Verify Corpus in Neo4j")
    print("="*60)

    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                # Count Article nodes
                result = session.run("MATCH (a:Article) RETURN COUNT(a) as count")
                article_count = result.single()["count"]

                # Count Chunk nodes
                result = session.run("MATCH (c:Chunk) RETURN COUNT(c) as count")
                chunk_count = result.single()["count"]

                # Count HAS_CHUNK edges
                result = session.run("MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk) RETURN COUNT(*) as count")
                has_chunk_count = result.single()["count"]

                # Count Entity nodes
                result = session.run("MATCH (e:Entity) RETURN COUNT(e) as count")
                entity_count = result.single()["count"]

                # Count MENTIONS edges
                result = session.run("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN COUNT(*) as count")
                mentions_count = result.single()["count"]

                # Check vector index exists
                result = session.run("SHOW INDEXES YIELD name, type WHERE name = 'chunk_embedding' RETURN name")
                vector_index = result.single()

    except Exception as e:
        print(f"[FAIL] Error querying Neo4j: {e}")
        return False

    print(f"[OK] Article nodes: {article_count}")
    print(f"[OK] Chunk nodes: {chunk_count}")
    print(f"[OK] HAS_CHUNK edges: {has_chunk_count}")
    print(f"[OK] Entity nodes: {entity_count}")
    print(f"[OK] MENTIONS edges: {mentions_count}")
    if vector_index:
        print(f"[OK] Vector index 'chunk_embedding': present")

    if article_count == 0:
        print("[FAIL] No Article nodes found in Neo4j!")
        return False

    if chunk_count == 0:
        print("[FAIL] No Chunk nodes found in Neo4j!")
        return False

    if article_count < 4000:
        print(f"[WARN] Expected ~5000 articles, found {article_count}")
        return False

    print(f"[OK] Corpus counts within expected range")
    return True


def step5_generate_and_verify_eval_questions():
    """Step 5: Generate and verify evaluation questions"""
    print("\n" + "="*60)
    print("STEP 5: Generate & Verify Evaluation Questions")
    print("="*60)

    # Import the evaluation filter function
    try:
        from src.evaluation.filter_questions import filter_and_save_questions
    except ImportError as e:
        print(f"[FAIL] Failed to import evaluation module: {e}")
        return False

    # Run the passage-matching evaluation filter
    try:
        print("Running evaluation question filtering (passage matching)...")
        records = filter_and_save_questions(
            output_path="data/processed/eval_questions.jsonl",
            num_samples=200,
            seed=42
        )
    except Exception as e:
        print(f"[FAIL] Evaluation filtering failed: {e}")
        return False

    # Verify the output file
    eval_path = PROJECT_ROOT / "data" / "processed" / "eval_questions.jsonl"

    if not eval_path.exists():
        print(f"[FAIL] eval_questions.jsonl not created at {eval_path}")
        return False

    # Load and verify records
    total_records = 0
    pubmed_ids = set()
    pqa_pubids = set()

    with open(eval_path, "r") as f:
        for line in f:
            if line.strip():
                total_records += 1
                data = json.loads(line)
                pubmed_ids.add(data.get("pubmed_id", ""))
                pqa_pubids.add(data.get("pqa_pubid", ""))

    print(f"[OK] Evaluation questions generated: {total_records}")
    print(f"[OK] Distinct corpus article IDs (gold): {len(pubmed_ids)}")
    print(f"[OK] Distinct PubMedQA pubids (traceability): {len(pqa_pubids)}")

    if total_records == 0:
        print("[FAIL] No evaluation questions generated!")
        return False

    # Verify gold IDs are pubmed_* format (corpus IDs)
    pubmed_count = sum(1 for pid in pubmed_ids if pid.startswith("pubmed_"))
    if pubmed_count == 0:
        print("[WARN] No corpus pubmed_* IDs found in gold IDs")
        return False

    print(f"[OK] Gold IDs use corpus article_id format: {pubmed_count}/{len(pubmed_ids)}")
    return True


def main():
    print("\nExecuting GraphRAG Pipeline End-to-End")
    print("=" * 60)

    # Step 1
    if not step1_verify_chunking():
        sys.exit(1)

    # Step 2
    if not step2_check_semantic_cluster():
        sys.exit(1)

    # Step 3
    if not step3_verify_neo4j():
        sys.exit(1)

    # Step 4
    if not step4_verify_corpus_in_graph():
        print("\n[WARN] Next steps:")
        print("   1. Run chunking with semantic_cluster strategy:")
        print("      python -m src.chunking.chunker --strategy semantic_cluster --merge-per-article")
        print("   2. Build the Neo4j graph:")
        print("      python -m src.graph.build_graph --limit 1000")
        print("   3. Run this script again to verify")
        sys.exit(1)

    # Step 5
    if not step5_generate_and_verify_eval_questions():
        print("\n[WARN] Evaluation question generation failed.")
        print("   Ensure the 5000-abstract corpus was built with --eval-ready:")
        print("      python -m src.chunking.load_data --eval-ready")
        print("   Then re-run chunking and graph building.")
        sys.exit(1)

    print("\nPipeline verification complete!\n")


if __name__ == "__main__":
    main()