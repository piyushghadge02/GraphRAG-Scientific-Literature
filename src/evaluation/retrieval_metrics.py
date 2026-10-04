"""
Retrieval metrics for Phase 4 evaluation.

Computes article-level binary metrics:
- Recall@5: 1.0 if ANY of top-5 chunks belongs to gold article, else 0.0
- Recall@10: 1.0 if ANY of top-10 chunks belongs to gold article, else 0.0
- MRR: 1/rank of first chunk belonging to gold article, or 0.0 if none found
"""

import pandas as pd
from typing import List, Dict


def recall_at_k(retrieved_chunk_ids: List[str], gold_article_id: str, chunk_to_article_map: Dict[str, str], k: int) -> float:
    """
    Compute article-level binary Recall@K.

    Returns 1.0 if ANY of the top-k chunks belongs to the gold article, else 0.0.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_article_id: The gold article ID (e.g., "pqa_12345")
        chunk_to_article_map: Mapping from chunk_id to article_id
        k: Number of top results to consider

    Returns:
        1.0 if any top-k chunk belongs to gold article, else 0.0
    """
    top_k_chunks = retrieved_chunk_ids[:k]

    for chunk_id in top_k_chunks:
        article_id = chunk_to_article_map.get(chunk_id)
        if article_id == gold_article_id:
            return 1.0

    return 0.0


def mrr(retrieved_chunk_ids: List[str], gold_article_id: str, chunk_to_article_map: Dict[str, str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns 1/rank of the first chunk that belongs to the gold article.
    Returns 0.0 if no chunk from the gold article is found.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in rank order
        gold_article_id: The gold article ID (e.g., "pqa_12345")
        chunk_to_article_map: Mapping from chunk_id to article_id

    Returns:
        1/rank of first relevant chunk, or 0.0 if none found
    """
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        article_id = chunk_to_article_map.get(chunk_id)
        if article_id == gold_article_id:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    questions: List[Dict],
    retrieve_fn,
    chunk_to_article_map: Dict[str, str],
    run_name: str = "Baseline"
) -> pd.DataFrame:
    """
    Evaluate retrieval metrics for a list of questions.

    Args:
        questions: List of question dicts with 'question' and 'pubmed_id' keys
        retrieve_fn: Function that takes a question string and returns list of chunk_ids
        chunk_to_article_map: Mapping from chunk_id to article_id
        run_name: Name of this run for display purposes

    Returns:
        DataFrame with columns: pubmed_id, question, recall_5, recall_10, mrr, and a MEAN row
    """
    results = []

    for q in questions:
        question_text = q["question"]
        gold_article_id = q["pubmed_id"]

        # Retrieve chunks
        retrieved_chunk_ids = retrieve_fn(question_text)

        # Compute metrics
        r5 = recall_at_k(retrieved_chunk_ids, gold_article_id, chunk_to_article_map, k=5)
        r10 = recall_at_k(retrieved_chunk_ids, gold_article_id, chunk_to_article_map, k=10)
        mrr_score = mrr(retrieved_chunk_ids, gold_article_id, chunk_to_article_map)

        results.append({
            "pubmed_id": gold_article_id,
            "question": question_text[:80] + "..." if len(question_text) > 80 else question_text,
            "recall_5": r5,
            "recall_10": r10,
            "mrr": mrr_score
        })

    df = pd.DataFrame(results)

    # Add MEAN row
    mean_row = {
        "pubmed_id": "MEAN",
        "question": "",
        "recall_5": df["recall_5"].mean(),
        "recall_10": df["recall_10"].mean(),
        "mrr": df["mrr"].mean()
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df
