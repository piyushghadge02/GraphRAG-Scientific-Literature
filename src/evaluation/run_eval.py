"""
Main evaluation runner for Phase 4.

Runs two evaluation modes:
- Run A (Baseline): Fixed token chunks with vector search only
- Run B (Full System): Semantic cluster chunks with graph expansion

Outputs:
- data/processed/eval_comparison.csv (Metric, Baseline, Full System, Delta)
- data/processed/retrieval_comparison.png
- data/processed/generation_comparison.png
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_rag_prompt
from src.retrieval.retrieve import retrieve
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.generation_metrics import evaluate_generation


def load_questions(path: str = "data/processed/eval_questions.jsonl", num_questions: int = 200) -> List[Dict]:
    """Load evaluation questions from JSONL file."""
    questions = []
    with open(path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions[:num_questions]


def load_chunks_and_build_map(chunk_file: str) -> tuple:
    """
    Load chunks from JSONL and build chunk_id -> article_id mapping.

    Returns:
        (chunks_list, chunk_to_article_map)
    """
    chunks = []
    chunk_to_article_map = {}

    with open(chunk_file, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            chunk_to_article_map[chunk["chunk_id"]] = chunk["article_id"]

    return chunks, chunk_to_article_map


def build_baseline_retriever(chunks: List[Dict], embedding_model):
    """
    Build baseline retriever using in-memory vector search.

    Args:
        chunks: List of chunk dicts with 'chunk_id' and 'text'
        embedding_model: SentenceTransformer model

    Returns:
        Function that takes query and returns list of chunk_ids
    """
    print("Encoding all baseline chunks...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Encode all chunks ONCE
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms

    print(f"Encoded {len(chunks)} chunks")

    def retrieve_baseline(query: str, top_k: int = 10) -> List[str]:
        """Retrieve top-k chunks for query using cosine similarity."""
        # Encode and normalize query
        query_emb = embedding_model.encode(query)
        query_norm = query_emb / np.linalg.norm(query_emb)

        # Compute cosine similarity
        similarities = np.dot(embeddings_normalized, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return chunk IDs
        return [chunk_ids[i] for i in top_indices]

    return retrieve_baseline


def build_full_system_retriever(driver, embedding_model):
    """
    Build full system retriever using existing retrieve() function.

    Args:
        driver: Neo4j driver
        embedding_model: SentenceTransformer model

    Returns:
        Function that takes query and returns list of chunk_ids
    """
    def retrieve_full(query: str, top_k: int = 10) -> List[str]:
        """Retrieve top-k chunks using full system with graph expansion."""
        results = retrieve(
            driver=driver,
            embedding_model=embedding_model,
            query=query,
            top_k=top_k,
            expand_graph=True
        )
        return [r["chunk_id"] for r in results]

    return retrieve_full


def run_generation_eval(questions: List[Dict], driver, embedding_model, llm_client, num_gen_questions: int = 20):
    """
    Run generation evaluation on a subset of questions.

    Args:
        questions: List of question dicts
        driver: Neo4j driver
        embedding_model: SentenceTransformer
        llm_client: LLMClient instance
        num_gen_questions: Number of questions to generate for (default: 20)

    Returns:
        (generated_answers, gold_answers, eval_questions)
    """
    print(f"\nRunning generation on {num_gen_questions} questions...")

    eval_questions = questions[:num_gen_questions]
    generated_answers = []
    gold_answers = []
    valid_questions = []

    for i, q in enumerate(eval_questions, 1):
        print(f"Generating answer {i}/{num_gen_questions}...", end=" ", flush=True)

        question_text = q["question"]
        gold_answer = q["long_answer"]

        try:
            # Use retrieve() function directly which returns chunk dicts
            chunk_results = retrieve(
                driver=driver,
                embedding_model=embedding_model,
                query=question_text,
                top_k=10,
                expand_graph=True
            )

            # Build prompt
            prompt = build_rag_prompt(question_text, chunk_results)

            # Generate answer with retry
            try:
                answer = llm_client.generate(prompt)
            except Exception as e:
                print(f"Retrying after error: {e}")
                time.sleep(2)
                try:
                    answer = llm_client.generate(prompt)
                except Exception as e2:
                    print(f"Failed again, skipping: {e2}")
                    continue

            generated_answers.append(answer)
            gold_answers.append(gold_answer)
            valid_questions.append(q)
            print("✓")

        except Exception as e:
            print(f"Failed: {e}")
            continue

    print(f"Successfully generated {len(generated_answers)} answers")

    return generated_answers, gold_answers, valid_questions


def run_full_eval(num_questions: int = 200, num_gen_questions: int = 20):
    """
    Run complete evaluation comparing baseline and full system.

    Args:
        num_questions: Number of questions to use for retrieval eval
        num_gen_questions: Number of questions to use for generation eval
    """
    print("=" * 60)
    print("PHASE 4 EVALUATION")
    print("=" * 60)

    # Load questions
    print("\n[1/7] Loading evaluation questions...")
    questions = load_questions(num_questions=num_questions)
    print(f"Loaded {len(questions)} questions")

    # Initialize models
    print("\n[2/7] Initializing models...")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    llm_client = LLMClient()
    print(f"Embedding model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"LLM: {settings.LLM_BACKEND} / {settings.LLM_MODEL}")

    # Connect to Neo4j
    print("\n[3/7] Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )
    print(f"Connected to {settings.NEO4J_URI}")

    # Load chunks and build mappings
    print("\n[4/7] Loading chunks...")
    baseline_chunks, baseline_map = load_chunks_and_build_map("data/processed/fixed_token.jsonl")
    semantic_chunks, semantic_map = load_chunks_and_build_map("data/processed/semantic_cluster.jsonl")
    print(f"Baseline chunks: {len(baseline_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    # Build retrievers
    print("\n[5/7] Building retrievers...")
    baseline_retriever = build_baseline_retriever(baseline_chunks, embedding_model)
    full_retriever = build_full_system_retriever(driver, embedding_model)

    # Run retrieval evaluation
    print("\n[6/7] Evaluating retrieval...")
    print("\n--- RUN A: BASELINE (fixed_token + vector only) ---")
    baseline_retrieval_df = evaluate_retrieval(
        questions,
        lambda q: baseline_retriever(q, top_k=10),
        baseline_map,
        run_name="Baseline"
    )

    print("\n--- RUN B: FULL SYSTEM (semantic_cluster + graph expansion) ---")
    full_retrieval_df = evaluate_retrieval(
        questions,
        lambda q: full_retriever(q, top_k=10),
        semantic_map,
        run_name="Full System"
    )

    # Extract means
    baseline_r5 = baseline_retrieval_df[baseline_retrieval_df["pubmed_id"] == "MEAN"]["recall_5"].values[0]
    baseline_r10 = baseline_retrieval_df[baseline_retrieval_df["pubmed_id"] == "MEAN"]["recall_10"].values[0]
    baseline_mrr = baseline_retrieval_df[baseline_retrieval_df["pubmed_id"] == "MEAN"]["mrr"].values[0]

    full_r5 = full_retrieval_df[full_retrieval_df["pubmed_id"] == "MEAN"]["recall_5"].values[0]
    full_r10 = full_retrieval_df[full_retrieval_df["pubmed_id"] == "MEAN"]["recall_10"].values[0]
    full_mrr = full_retrieval_df[full_retrieval_df["pubmed_id"] == "MEAN"]["mrr"].values[0]

    print("\nRETRIEVAL RESULTS:")
    print(f"Baseline  - Recall@5: {baseline_r5:.3f}, Recall@10: {baseline_r10:.3f}, MRR: {baseline_mrr:.3f}")
    print(f"Full Sys  - Recall@5: {full_r5:.3f}, Recall@10: {full_r10:.3f}, MRR: {full_mrr:.3f}")
    print(f"Delta     - Recall@5: {full_r5 - baseline_r5:+.3f}, Recall@10: {full_r10 - baseline_r10:+.3f}, MRR: {full_mrr - baseline_mrr:+.3f}")

    # Run generation evaluation
    print("\n[7/7] Evaluating generation...")
    print("\n--- GENERATION EVAL (Full System) ---")
    gen_answers, gold_answers, gen_questions = run_generation_eval(
        questions,
        driver,
        embedding_model,
        llm_client,
        num_gen_questions=num_gen_questions
    )

    if len(gen_answers) > 0:
        generation_df = evaluate_generation(gen_answers, gold_answers, gen_questions)

        rouge_mean = generation_df[generation_df["pubmed_id"] == "MEAN"]["rouge_l_f1"].values[0]
        bert_mean = generation_df[generation_df["pubmed_id"] == "MEAN"]["bertscore_f1"].values[0]

        print("\nGENERATION RESULTS:")
        print(f"ROUGE-L F1:    {rouge_mean:.3f}")
        print(f"BERTScore F1:  {bert_mean:.3f}")
    else:
        print("\nNo generation results (all attempts failed)")
        rouge_mean = 0.0
        bert_mean = 0.0
        generation_df = pd.DataFrame()

    # Save comparison CSV
    print("\n[OUTPUTS] Saving results...")
    comparison_data = {
        "Metric": ["Recall@5", "Recall@10", "MRR", "ROUGE-L F1", "BERTScore F1"],
        "Baseline": [baseline_r5, baseline_r10, baseline_mrr, "N/A", "N/A"],
        "Full_System": [full_r5, full_r10, full_mrr, rouge_mean if len(gen_answers) > 0 else "N/A", bert_mean if len(gen_answers) > 0 else "N/A"],
        "Delta": [
            full_r5 - baseline_r5,
            full_r10 - baseline_r10,
            full_mrr - baseline_mrr,
            "N/A",
            "N/A"
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("data/processed/eval_comparison.csv", index=False)
    print("✓ Saved data/processed/eval_comparison.csv")

    # Generate visualizations
    print("\n[VISUALIZATIONS] Generating charts...")

    # Retrieval comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Recall@5", "Recall@10", "MRR"]
    baseline_vals = [baseline_r5, baseline_r10, baseline_mrr]
    full_vals = [full_r5, full_r10, full_mrr]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, full_vals, width, label='Full System', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Retrieval Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/processed/retrieval_comparison.png", dpi=150)
    print("✓ Saved data/processed/retrieval_comparison.png")
    plt.close()

    # Generation comparison
    if len(gen_answers) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        gen_metrics = ["ROUGE-L F1", "BERTScore F1"]
        gen_vals = [rouge_mean, bert_mean]

        ax.bar(gen_metrics, gen_vals, alpha=0.8, color=['#2ecc71', '#3498db'])
        ax.set_ylabel('Score')
        ax.set_title('Generation Metrics (Full System)')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig("data/processed/generation_comparison.png", dpi=150)
        print("✓ Saved data/processed/generation_comparison.png")
        plt.close()

    # Close Neo4j connection
    driver.close()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nOutputs:")
    print("  - data/processed/eval_comparison.csv")
    print("  - data/processed/retrieval_comparison.png")
    if len(gen_answers) > 0:
        print("  - data/processed/generation_comparison.png")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 4 evaluation")
    parser.add_argument("--num-questions", type=int, default=200, help="Number of questions for retrieval eval")
    parser.add_argument("--num-gen-questions", type=int, default=20, help="Number of questions for generation eval")

    args = parser.parse_args()

    run_full_eval(num_questions=args.num_questions, num_gen_questions=args.num_gen_questions)
