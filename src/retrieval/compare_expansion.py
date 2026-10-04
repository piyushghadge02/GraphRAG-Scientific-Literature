"""
Task 7 — Manual Comparison Script (Rewritten)

Load PubMedQA (pqa_labeled) directly and for each question:
1. Check which articles are mentioned in the question's context
2. Verify if those articles exist in our graph (as pqa_{pubid} nodes)
3. Run retrieve() targeting those specific pqa_{pubid} articles
4. Run retrieve() with general query expansion for comparison
5. Save results to data/processed/expansion_comparison.jsonl

This approach leverages the PubMedQA context passages that are now indexed
in the graph with article_id = pqa_{pubid}.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import build_rag_prompt
from src.graph.schema import Neo4jConnection
from src.retrieval.retrieve import retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "expansion_comparison.jsonl"


def load_pubmedqa_questions(max_questions: int = 10) -> List[dict]:
    """
    Load PubMedQA dataset questions with their contexts.

    Args:
        max_questions: Maximum number of questions to load (default: 10)

    Returns:
        List of question dicts with keys: pubid, question, context, answer_choices
    """
    print("[compare_expansion] Loading PubMedQA (pqa_labeled) from Hugging Face...")
    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"[compare_expansion] ERROR loading dataset: {e}")
        print("Make sure you have: pip install datasets")
        raise

    questions = []
    for i, example in enumerate(dataset):
        if len(questions) >= max_questions:
            break

        pubid = example.get("pubid", "")
        question = example.get("question", "")
        context = example.get("context", "")

        # Extract context if it's a list
        if isinstance(context, list):
            context_text = " ".join(context)
        else:
            context_text = str(context) if context else ""

        if not question:
            continue

        questions.append({
            "pubid": pubid,
            "question": question,
            "context": context_text.strip(),
            "answer_choices": {
                "yes": example.get("yes", []),
                "no": example.get("no", []),
                "maybe": example.get("maybe", [])
            }
        })

    print(f"[compare_expansion] Loaded {len(questions)} questions from PubMedQA")
    return questions


def check_article_in_graph(driver, article_id: str) -> bool:
    """
    Check if an article exists in the Neo4j graph.

    Args:
        driver: Neo4j driver
        article_id: Article ID to check (format: pqa_{pubid})

    Returns:
        True if article exists, False otherwise
    """
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Article {article_id: $article_id}) RETURN COUNT(a) as count",
            article_id=article_id
        )
        row = result.single()
        return row["count"] > 0 if row else False


def run_comparison(
    driver,
    embedding_model: SentenceTransformer,
    llm_client: LLMClient,
    questions: List[dict]
) -> List[dict]:
    """
    Run retrieve() + generate() for each question.

    For each question:
    1. Check if pqa_{pubid} article exists in graph
    2. Run retrieve() targeting that specific article
    3. Run retrieve() with general expansion for comparison

    Args:
        driver: Neo4j driver
        embedding_model: Sentence transformer for embeddings
        llm_client: LLM client for generation
        questions: List of questions to process

    Returns:
        List of comparison result dicts
    """
    results = []

    for i, q_data in enumerate(questions, start=1):
        pubid = q_data["pubid"]
        question = q_data["question"]
        context = q_data["context"]
        article_id = f"pqa_{pubid}"

        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)} (pubid: {pubid})")
        print(f"{'='*80}")
        print(f"Question: {question}\n")
        print(f"Context (first 300 chars): {context[:300]}...\n")

        # Check if article exists in graph
        article_exists = check_article_in_graph(driver, article_id)
        print(f"Article {article_id} in graph: {article_exists}")

        try:
            # Retrieve with expand_graph=False
            print("\n[Retrieving WITHOUT graph expansion...]")
            chunks_no_expand = retrieve(
                driver,
                embedding_model,
                query=question,
                top_k=5,
                expand_graph=False
            )

            prompt_no_expand = build_rag_prompt(question, chunks_no_expand)
            answer_no_expand = llm_client.generate(prompt_no_expand)

            # Retrieve with expand_graph=True
            print("\n[Retrieving WITH graph expansion...]")
            chunks_with_expand = retrieve(
                driver,
                embedding_model,
                query=question,
                top_k=5,
                expand_graph=True
            )

            prompt_with_expand = build_rag_prompt(question, chunks_with_expand)
            answer_with_expand = llm_client.generate(prompt_with_expand)

            # --- Extra diagnostics computation ---
            from src.retrieval.vector_search import vector_search
            from src.retrieval.graph_expand import expand_via_entities

            query_embedding = embedding_model.encode(question).tolist()
            vector_results = vector_search(driver, query_embedding, top_k=15)
            vector_ids = [v["chunk_id"] for v in vector_results]

            expanded_results = expand_via_entities(driver, vector_ids, max_depth=1)
            candidates_discovered = [r["chunk_id"] for r in expanded_results if r["chunk_id"] not in vector_ids]

            ids_no_expand = [c["chunk_id"] for c in chunks_no_expand]
            ids_with_expand = [c["chunk_id"] for c in chunks_with_expand]

            candidates_entering_top_k = [cid for cid in ids_with_expand if cid not in vector_ids]
            added_chunks = [cid for cid in ids_with_expand if cid not in ids_no_expand]
            removed_chunks = [cid for cid in ids_no_expand if cid not in ids_with_expand]

            overlap_chunks = [cid for cid in ids_with_expand if cid in ids_no_expand]
            overlap_count = len(overlap_chunks)
            overlap_percentage = (overlap_count / len(ids_no_expand) * 100) if ids_no_expand else 0.0

            # Print comparison diagnostics for each question
            print("\n" + "-"*40)
            print("DIAGNOSTIC REPORT FOR THIS QUESTION:")
            print("WITHOUT GRAPH:")
            print(f"  Final chunk IDs: {ids_no_expand}")
            print("WITH GRAPH:")
            print(f"  Final chunk IDs: {ids_with_expand}")
            print("EXPANSION:")
            print(f"  Graph candidates discovered (count): {len(candidates_discovered)}")
            print(f"  Graph candidates entering final top-k: {len(candidates_entering_top_k)} (IDs: {candidates_entering_top_k})")
            print(f"  Added chunk IDs: {added_chunks}")
            print(f"  Removed/replaced seed chunk IDs: {removed_chunks}")
            print(f"  Overlap count: {overlap_count} / Overlap percentage: {overlap_percentage:.1f}%")
            print("-"*40 + "\n")

            # Store comparison result
            results.append({
                "pubid": pubid,
                "article_id": article_id,
                "question": question,
                "context": context,
                "article_in_graph": article_exists,
                "chunks_no_expand": chunks_no_expand,
                "chunks_with_expand": chunks_with_expand,
                "num_chunks_no_expand": len(chunks_no_expand),
                "num_chunks_with_expand": len(chunks_with_expand),
                "answer_no_expand": answer_no_expand,
                "answer_with_expand": answer_with_expand,
                # Extra metric fields for summary aggregation
                "candidates_discovered_count": len(candidates_discovered),
                "candidates_entering_top_k_count": len(candidates_entering_top_k),
                "added_chunks_count": len(added_chunks),
                "overlap_percentage": overlap_percentage
            })

        except Exception as e:
            print(f"[compare_expansion] ERROR processing question {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "pubid": pubid,
                "article_id": article_id,
                "question": question,
                "context": context,
                "error": str(e)
            })

    return results


def save_results(results: List[dict]) -> None:
    """Save comparison results to JSONL file."""
    print(f"\n[compare_expansion] Saving results to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[compare_expansion] ✓ Saved {len(results)} comparisons to {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare retrieval with and without graph expansion on PubMedQA questions."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to process (default: 10)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=settings.LLM_BACKEND,
        help=f"LLM backend (default: {settings.LLM_BACKEND})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.LLM_MODEL,
        help=f"LLM model (default: {settings.LLM_MODEL})"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  GraphRAG Expansion Comparison (PubMedQA)")
    print("=" * 80)

    # Load PubMedQA questions
    try:
        questions = load_pubmedqa_questions(max_questions=args.num_questions)
        if not questions:
            print("[compare_expansion] No questions loaded. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"[compare_expansion] ERROR loading questions: {e}")
        sys.exit(1)

    # Initialize components
    print("\n[compare_expansion] Initializing embedding model and LLM client...")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    llm_client = LLMClient(backend=args.backend, model=args.model)

    # Connect to Neo4j
    print("[compare_expansion] Connecting to Neo4j...")
    try:
        with Neo4jConnection() as driver:
            # Run comparisons
            results = run_comparison(driver, embedding_model, llm_client, questions)
    except Exception as e:
        print(f"[compare_expansion] ERROR connecting to Neo4j: {e}")
        print("Make sure Neo4j is running and environment variables are set correctly.")
        sys.exit(1)

    # Save results
    save_results(results)

    # Print summary stats
    successful = [r for r in results if "error" not in r]
    with_article = [r for r in successful if r.get("article_in_graph", False)]
    found_candidates = [r for r in successful if r.get("candidates_discovered_count", 0) > 0]
    changed_top_k = [r for r in successful if r.get("added_chunks_count", 0) > 0]

    print("\n" + "=" * 80)
    print("  Comparison Summary")
    print("=" * 80)
    print(f"Questions processed:                      {len(results)}")
    print(f"Questions with article in graph:          {len(with_article)}")
    print(f"Questions where expansion found candidates: {len(found_candidates)}")
    print(f"Questions where expansion changed final top-k: {len(changed_top_k)}")

    if successful:
        avg_discovered = sum(r.get("candidates_discovered_count", 0) for r in successful) / len(successful)
        avg_entering = sum(r.get("candidates_entering_top_k_count", 0) for r in successful) / len(successful)
        avg_overlap = sum(r.get("overlap_percentage", 0.0) for r in successful) / len(successful)

        print(f"Average graph candidates discovered:       {avg_discovered:.1f}")
        print(f"Average graph candidates entering final top-k: {avg_entering:.1f}")
        print(f"Average overlap between baseline and expanded: {avg_overlap:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
