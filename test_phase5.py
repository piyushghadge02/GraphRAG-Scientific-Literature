"""
Phase 5 - Test Script
Tests query decomposition, PageRank reranking, and full pipeline integration.
"""

import sys
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_rag_prompt
from src.retrieval.decompose import decompose_query, retrieve_decomposed
from src.retrieval.gds_rerank import rerank_with_pagerank
from src.retrieval.retrieve_with_pagerank import retrieve_with_pagerank


def test_query_decomposition():
    """Test query decomposition functionality."""
    print("\n" + "="*80)
    print("TEST 1: Query Decomposition")
    print("="*80)

    try:
        llm_client = LLMClient()

        # Test simple query
        simple_query = "What is apoptosis?"
        print(f"\nSimple Query: {simple_query}")
        sub_queries = decompose_query(llm_client, simple_query)
        print(f"Sub-queries: {sub_queries}")
        assert isinstance(sub_queries, list), "Should return a list"

        # Test complex query
        complex_query = "What is the relationship between mitochondrial dysfunction and programmed cell death, and how does this affect plant development?"
        print(f"\nComplex Query: {complex_query}")
        sub_queries = decompose_query(llm_client, complex_query)
        print(f"Sub-queries ({len(sub_queries)}): {sub_queries}")
        assert isinstance(sub_queries, list), "Should return a list"

        print("\n✅ Query decomposition test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Query decomposition test FAILED: {e}")
        return False


def test_pagerank_reranking():
    """Test PageRank reranking functionality."""
    print("\n" + "="*80)
    print("TEST 2: PageRank Reranking")
    print("="*80)

    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

        # Get some sample chunk IDs from the database
        with driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id LIMIT 5")
            chunk_ids = [record["chunk_id"] for record in result]

        if not chunk_ids:
            print("⚠️ No chunks found in database. Skipping PageRank test.")
            return True

        print(f"\nTesting with {len(chunk_ids)} chunks: {chunk_ids[:3]}...")

        # Test PageRank computation
        pagerank_scores = rerank_with_pagerank(driver, chunk_ids)

        print(f"\nPageRank scores computed for {len(pagerank_scores)} chunks")
        if pagerank_scores:
            for chunk_id, score in list(pagerank_scores.items())[:3]:
                print(f"  {chunk_id}: {score:.4f}")

        driver.close()

        print("\n✅ PageRank reranking test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ PageRank reranking test FAILED: {e}")
        return False


def test_end_to_end_retrieval():
    """Test end-to-end retrieval with PageRank."""
    print("\n" + "="*80)
    print("TEST 3: End-to-End Retrieval with PageRank")
    print("="*80)

    try:
        # Initialize components
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

        query = "Do mitochondria play a role in programmed cell death?"
        print(f"\nQuery: {query}")

        # Test retrieval WITHOUT PageRank
        print("\n--- Retrieval WITHOUT PageRank ---")
        results_no_pr = retrieve_with_pagerank(
            driver=driver,
            embedding_model=embedding_model,
            query=query,
            top_k=3,
            expand_graph=True,
            use_pagerank=False
        )

        print(f"Retrieved {len(results_no_pr)} chunks")
        for i, chunk in enumerate(results_no_pr, 1):
            print(f"  [{i}] {chunk['chunk_id']}: final_score={chunk['final_score']:.4f}")

        # Test retrieval WITH PageRank
        print("\n--- Retrieval WITH PageRank ---")
        results_with_pr = retrieve_with_pagerank(
            driver=driver,
            embedding_model=embedding_model,
            query=query,
            top_k=3,
            expand_graph=True,
            use_pagerank=True
        )

        print(f"Retrieved {len(results_with_pr)} chunks")
        for i, chunk in enumerate(results_with_pr, 1):
            print(f"  [{i}] {chunk['chunk_id']}: combined={chunk['combined_score']:.4f}, "
                  f"pagerank={chunk['pagerank_score']:.4f}, final={chunk['final_score']:.4f}")

        driver.close()

        print("\n✅ End-to-end retrieval test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ End-to-end retrieval test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_qa_pipeline():
    """Test full Q&A pipeline with decomposition."""
    print("\n" + "="*80)
    print("TEST 4: Full Q&A Pipeline")
    print("="*80)

    try:
        # Initialize components
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        llm_client = LLMClient()

        query = "Do mitochondria play a role in programmed cell death?"
        print(f"\nQuery: {query}")

        # Retrieve with decomposition
        print("\n--- Retrieving with query decomposition ---")
        chunks = retrieve_decomposed(
            driver=driver,
            embedding_model=embedding_model,
            question=query,
            llm_client=llm_client,
            top_k_per_subq=2,
            expand_graph=True
        )

        print(f"\nRetrieved {len(chunks)} unique chunks after merging")

        # Generate answer
        print("\n--- Generating answer ---")
        prompt = build_rag_prompt(query, chunks[:3])
        answer = llm_client.generate(prompt)

        print(f"\nAnswer ({len(answer)} chars):")
        print(answer[:300] + "..." if len(answer) > 300 else answer)

        driver.close()

        print("\n✅ Full Q&A pipeline test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Full Q&A pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 5 TEST SUITE")
    print("="*80)

    results = {
        "Query Decomposition": test_query_decomposition(),
        "PageRank Reranking": test_pagerank_reranking(),
        "End-to-End Retrieval": test_end_to_end_retrieval(),
        "Full Q&A Pipeline": test_full_qa_pipeline()
    }

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
