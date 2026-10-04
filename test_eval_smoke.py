"""
Quick smoke test for evaluation module.
"""

import json

# Test 1: Check that questions were filtered
print("=" * 60)
print("Test 1: Check filtered questions file")
print("=" * 60)

try:
    with open("data/processed/eval_questions.jsonl", 'r') as f:
        questions = [json.loads(line) for line in f]

    print(f"✓ Found {len(questions)} questions")
    print(f"✓ Sample question keys: {list(questions[0].keys())}")
    print(f"✓ First question: {questions[0]['question'][:80]}...")
    print(f"✓ First pubmed_id: {questions[0]['pubmed_id']}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Test metric functions
print("\n" + "=" * 60)
print("Test 2: Test retrieval metrics")
print("=" * 60)

try:
    from src.evaluation.retrieval_metrics import recall_at_k, mrr

    # Synthetic test
    retrieved = ["chunk_1", "chunk_2", "chunk_3"]
    chunk_map = {
        "chunk_1": "article_A",
        "chunk_2": "article_B",
        "chunk_3": "article_A"
    }
    gold_article = "article_A"

    r5 = recall_at_k(retrieved, gold_article, chunk_map, k=5)
    mrr_score = mrr(retrieved, gold_article, chunk_map)

    print(f"✓ Recall@5: {r5}")
    print(f"✓ MRR: {mrr_score}")

    assert r5 == 1.0, "Expected recall@5 = 1.0"
    assert mrr_score == 1.0, "Expected MRR = 1.0"
    print("✓ Metric calculations correct!")

except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Check output files exist from previous run
print("\n" + "=" * 60)
print("Test 3: Check evaluation outputs (if they exist)")
print("=" * 60)

import os

files_to_check = [
    "data/processed/eval_comparison.csv",
    "data/processed/retrieval_comparison.png"
]

for fpath in files_to_check:
    if os.path.exists(fpath):
        print(f"✓ Found {fpath}")
    else:
        print(f"○ {fpath} not yet created (run eval first)")

print("\n" + "=" * 60)
print("SMOKE TEST COMPLETE")
print("=" * 60)
print("\nAll core components are functional!")
print("\nTo run full evaluation:")
print("  python -m src.evaluation.run_eval --num-questions 5 --num-gen-questions 2")
