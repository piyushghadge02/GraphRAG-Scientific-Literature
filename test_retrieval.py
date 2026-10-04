#!/usr/bin/env python3
"""
Quick test script to verify retrieval pipeline components work together.
Tests basic imports and component integration without Neo4j.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Test imports
print("[test_retrieval] Testing imports...")
try:
    from src.retrieval.vector_search import vector_search
    print("  ✓ vector_search imported")

    from src.retrieval.graph_expand import expand_via_entities
    print("  ✓ expand_via_entities imported")

    from src.retrieval.fusion import combined_score
    print("  ✓ combined_score imported")

    from src.retrieval.retrieve import retrieve
    print("  ✓ retrieve imported")

    from src.generation.prompt_templates import build_rag_prompt
    print("  ✓ build_rag_prompt imported")

    from src.generation.llm_client import LLMClient
    print("  ✓ LLMClient imported")

    from src.config import settings
    print("  ✓ settings imported")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test combined_score function
print("\n[test_retrieval] Testing combined_score function...")
try:
    score = combined_score(
        vector_score=0.9,
        shared_entities=3,
        max_shared_entities=5,
        alpha=0.7
    )
    expected = 0.7 * 0.9 + 0.3 * (3/5)
    print(f"  Score: {score:.4f} (expected: {expected:.4f})")
    assert abs(score - expected) < 0.0001, f"Score mismatch: {score} != {expected}"
    print("  ✓ combined_score works correctly")
except Exception as e:
    print(f"  ✗ combined_score failed: {e}")
    sys.exit(1)

# Test build_rag_prompt function
print("\n[test_retrieval] Testing build_rag_prompt function...")
try:
    test_chunks = [
        {"chunk_id": "chunk_1", "text": "Metformin improves insulin sensitivity."},
        {"chunk_id": "chunk_2", "text": "Insulin resistance is linked to diabetes."}
    ]
    prompt = build_rag_prompt(
        question="What are the effects of metformin?",
        chunks=test_chunks
    )

    # Check that prompt contains expected elements
    assert "[1]" in prompt, "Missing chunk [1] reference"
    assert "[2]" in prompt, "Missing chunk [2] reference"
    assert "Metformin improves" in prompt, "Missing chunk text"
    assert "I don't have enough information" in prompt, "Missing insufficient context instruction"
    assert "What are the effects of metformin?" in prompt, "Missing question"

    print(f"  ✓ build_rag_prompt works correctly")
    print(f"    Generated prompt length: {len(prompt)} chars")
except Exception as e:
    print(f"  ✗ build_rag_prompt failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test config settings
print("\n[test_retrieval] Testing config settings...")
try:
    from src.config import settings
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"  Top-K Default: {settings.TOP_K_DEFAULT}")
    print(f"  Alpha Fusion Weight: {settings.ALPHA_FUSION_WEIGHT}")
    print(f"  LLM Backend: {settings.LLM_BACKEND}")
    print("  ✓ Config settings loaded correctly")
except Exception as e:
    print(f"  ✗ Config settings failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("  ✓ All tests passed!")
print("="*60)
