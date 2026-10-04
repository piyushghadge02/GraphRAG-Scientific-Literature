"""
Generation metrics for Phase 4 evaluation.

Computes:
- ROUGE-L F1 (using rouge_score)
- BERTScore F1 (using bert_score with model_type="distilbert-base-uncased")
"""

import pandas as pd
from typing import List, Dict
from rouge_score import rouge_scorer
import bert_score


def evaluate_generation(generated_answers: List[str], gold_answers: List[str], questions: List[Dict]) -> pd.DataFrame:
    """
    Evaluate generation quality metrics.

    Args:
        generated_answers: List of generated answer strings
        gold_answers: List of gold answer strings
        questions: List of question dicts (for display)

    Returns:
        DataFrame with columns: pubmed_id, question, rouge_l_f1, bertscore_f1, and a MEAN row
    """
    assert len(generated_answers) == len(gold_answers) == len(questions), "Mismatched lengths"

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Compute ROUGE-L scores
    rouge_scores = []
    for gen, gold in zip(generated_answers, gold_answers):
        score = scorer.score(gold, gen)
        rouge_scores.append(score['rougeL'].fmeasure)

    # Compute BERTScore
    print("Computing BERTScore (this may take a moment)...")
    P, R, F1 = bert_score.score(
        generated_answers,
        gold_answers,
        model_type="distilbert-base-uncased",
        verbose=False
    )
    bertscore_f1_list = F1.tolist()

    # Build results
    results = []
    for i, q in enumerate(questions):
        results.append({
            "pubmed_id": q["pubmed_id"],
            "question": q["question"][:80] + "..." if len(q["question"]) > 80 else q["question"],
            "rouge_l_f1": rouge_scores[i],
            "bertscore_f1": bertscore_f1_list[i]
        })

    df = pd.DataFrame(results)

    # Add MEAN row
    mean_row = {
        "pubmed_id": "MEAN",
        "question": "",
        "rouge_l_f1": df["rouge_l_f1"].mean(),
        "bertscore_f1": df["bertscore_f1"].mean()
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    return df
