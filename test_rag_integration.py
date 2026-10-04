#!/usr/bin/env python3
"""
Task 8 - Integration Test
Tests the complete RAG pipeline:
1. Connects to Neo4j
2. Runs vector search on test query
3. Runs graph expansion
4. Verifies result merging and score fusion
5. Tests prompt generation with results
"""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.generation.prompt_templates import build_rag_prompt
from src.graph.schema import Neo4jConnection
from src.retrieval.fusion import combined_score
from src.retrieval.graph_expand import expand_via_entities
from src.retrieval.retrieve import retrieve
from src.retrieval.vector_search import vector_search


def test_neo4j_connection():
    """Test Neo4j connection and verify schema exists."""
    print("[integration_test] Testing Neo4j connection...")
    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 AS status")
                record = result.single()
                if record:
                    print("  ✓ Neo4j connection successful")
                    return True
    except Exception as e:
        print(f"  ✗ Neo4j connection failed: {e}")
        print("  Make sure Neo4j is running and accessible at " + settings.NEO4J_URI)
        return False


def test_data_exists():
    """Check if graph has any chunks (prerequisite for retrieval)."""
    print("[integration_test] Checking for data in Neo4j...")
    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                # Count chunks
                result = session.run("MATCH (c:Chunk) RETURN COUNT(c) AS chunk_count")
                record = result.single()
                chunk_count = record["chunk_count"] if record else 0

                # Count entities
                result = session.run("MATCH (e:Entity) RETURN COUNT(e) AS entity_count")
                record = result.single()
                entity_count = record["entity_count"] if record else 0

                print(f"  Found {chunk_count} chunks and {entity_count} entities in Neo4j")

                if chunk_count == 0:
                    print("  ✗ No chunks found. Please run: python -m src.graph.build_graph --limit 100")
                    return False

                print("  ✓ Data exists in Neo4j")
                return True
    except Exception as e:
        print(f"  ✗ Data check failed: {e}")
        return False


def test_vector_search(driver, embedding_model):
    """Test vector search functionality."""
    print("\n[integration_test] Testing vector_search()...")
    try:
        test_query = "What are the effects of metformin on insulin resistance?"
        query_embedding = embedding_model.encode(test_query).tolist()

        results = vector_search(driver, query_embedding, top_k=5)

        if not results:
            print("  ✗ No results from vector search")
            return False

        print(f"  ✓ Vector search returned {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"    [{i}] chunk_id={result['chunk_id'][:20]}..., score={result['score']:.4f}")

        return results
    except Exception as e:
        print(f"  ✗ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_expansion(driver, vector_results):
    """Test graph expansion with results from vector search."""
    print("\n[integration_test] Testing expand_via_entities()...")
    try:
        if not vector_results:
            print("  ✗ No vector results to expand from")
            return False

        seed_chunk_ids = [r["chunk_id"] for r in vector_results]
        expanded_results = expand_via_entities(driver, seed_chunk_ids, max_depth=1)

        print(f"  ✓ Graph expansion returned {len(expanded_results)} candidate chunks")
        if expanded_results:
            for i, result in enumerate(expanded_results[:5], 1):  # Show top 5
                print(f"    [{i}] chunk_id={result['chunk_id'][:20]}..., shared_entities={result['shared_entities']}")

        return expanded_results
    except Exception as e:
        print(f"  ✗ Graph expansion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_retrieve(driver, embedding_model):
    """Test the unified retrieve function with and without graph expansion."""
    print("\n[integration_test] Testing retrieve() with expand_graph=False...")
    try:
        test_query = "What are the effects of metformin on insulin resistance?"

        results_no_expand = retrieve(
            driver,
            embedding_model,
            query=test_query,
            top_k=5,
            expand_graph=False
        )

        if not results_no_expand:
            print("  ✗ No results from retrieve with expand_graph=False")
            return False

        print(f"  ✓ retrieve() (expand_graph=False) returned {len(results_no_expand)} results")
        for i, result in enumerate(results_no_expand, 1):
            print(f"    [{i}] combined_score={result['combined_score']:.4f}, "
                  f"vector_score={result['vector_score']:.4f}, "
                  f"shared_entities={result['shared_entities']}")

    except Exception as e:
        print(f"  ✗ retrieve() (expand_graph=False) failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[integration_test] Testing retrieve() with expand_graph=True...")
    try:
        results_with_expand = retrieve(
            driver,
            embedding_model,
            query=test_query,
            top_k=5,
            expand_graph=True
        )

        if not results_with_expand:
            print("  ✗ No results from retrieve with expand_graph=True")
            return False

        print(f"  ✓ retrieve() (expand_graph=True) returned {len(results_with_expand)} results")
        for i, result in enumerate(results_with_expand, 1):
            print(f"    [{i}] combined_score={result['combined_score']:.4f}, "
                  f"vector_score={result['vector_score']:.4f}, "
                  f"shared_entities={result['shared_entities']}")

        # Compare results
        print("\n  Comparison:")
        print(f"    Results difference: {len(results_with_expand)} vs {len(results_no_expand)}")

        no_expand_ids = {r["chunk_id"] for r in results_no_expand}
        with_expand_ids = {r["chunk_id"] for r in results_with_expand}
        new_from_expansion = with_expand_ids - no_expand_ids

        if new_from_expansion:
            print(f"    New chunks added by graph expansion: {len(new_from_expansion)}")
        else:
            print("    No new chunks from graph expansion (might happen if vector search covers most entities)")

        return (results_no_expand, results_with_expand)
    except Exception as e:
        print(f"  ✗ retrieve() (expand_graph=True) failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_generation(results):
    """Test prompt generation with retrieved chunks."""
    print("\n[integration_test] Testing build_rag_prompt()...")
    try:
        test_query = "What are the effects of metformin on insulin resistance?"

        # Use results from expand_graph=True
        if isinstance(results, tuple):
            results_with_expand = results[1]
        else:
            results_with_expand = results

        prompt = build_rag_prompt(test_query, results_with_expand)

        if not prompt:
            print("  ✗ Generated prompt is empty")
            return False

        # Verify prompt structure
        checks = [
            ("[1]" in prompt, "Chunk numbering [1]"),
            (test_query in prompt, "Question in prompt"),
            ("I don't have enough information" in prompt, "Insufficient context instruction"),
            ("cite" in prompt.lower(), "Citation instruction"),
        ]

        all_passed = True
        for check, description in checks:
            if check:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description}")
                all_passed = False

        print(f"\n  Generated prompt preview (first 300 chars):")
        print(f"  {prompt[:300]}...")

        return all_passed
    except Exception as e:
        print(f"  ✗ Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("  GraphRAG RAG Pipeline Integration Test")
    print("=" * 80)

    # Step 1: Connection test
    if not test_neo4j_connection():
        print("\n[integration_test] ✗ Cannot proceed without Neo4j connection")
        sys.exit(1)

    # Step 2: Data existence check
    if not test_data_exists():
        print("\n[integration_test] ✗ Cannot proceed without data in Neo4j")
        sys.exit(1)

    # Initialize embedding model
    print(f"\n[integration_test] Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        print("  ✓ Embedding model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load embedding model: {e}")
        sys.exit(1)

    # Connect to Neo4j for tests
    try:
        with Neo4jConnection() as driver:
            # Step 3: Vector search test
            vector_results = test_vector_search(driver, embedding_model)
            if not vector_results:
                print("\n[integration_test] ✗ Vector search test failed")
                sys.exit(1)

            # Step 4: Graph expansion test
            expanded_results = test_graph_expansion(driver, vector_results)
            if expanded_results is False:
                print("\n[integration_test] ✗ Graph expansion test failed")
                sys.exit(1)

            # Step 5: Unified retrieve test
            retrieve_results = test_unified_retrieve(driver, embedding_model)
            if not retrieve_results:
                print("\n[integration_test] ✗ Unified retrieve test failed")
                sys.exit(1)

            # Step 6: Prompt generation test
            if not test_prompt_generation(retrieve_results):
                print("\n[integration_test] ✗ Prompt generation test failed")
                sys.exit(1)

    except Exception as e:
        print(f"\n[integration_test] ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  ✓ All integration tests passed!")
    print("=" * 80)
    print("\nRAG Pipeline is ready for use. You can now:")
    print("  - Use retrieve() for semantic search with optional graph expansion")
    print("  - Use build_rag_prompt() to format results into LLM prompts")
    print("  - Run compare_expansion.py to compare with/without graph expansion")
    print("=" * 80)


if __name__ == "__main__":
    main()
