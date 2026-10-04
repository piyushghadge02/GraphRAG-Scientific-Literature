"""
Phase 3 — Baseline vs modified comparison runner.

Compares the named evaluation systems (see src/evaluation/systems.py)
against the SAME questions and dataset:

- retrieval: Recall@5 / Recall@10 / MRR per system (metrics preserved
  exactly from src/evaluation/retrieval_metrics.py).
- generation: ROUGE-L / BERTScore per generation system (metrics preserved
  exactly from src/evaluation/generation_metrics.py).

Outputs (data/processed/):
- eval_comparison.csv  — Metric,Baseline,Full_System,Delta (same schema the
  Streamlit sidebar reads; Full_System = our modified "adaptive" system).
- eval_systems.csv     — per-system means for every evaluated system.
- eval_generation.csv  — per-system ROUGE-L / BERTScore means.
- retrieval_comparison.png — grouped retrieval bar chart.

Usage:
    python -m src.evaluation.compare --num-questions 200 --num-gen-questions 20

Requires populated Neo4j, chunk files, an embedding model and (for
generation) LLM credentials. No metric is fabricated: files are only
written from measured runs.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.systems import SYSTEM_ORDER, resolve_system, run_system
from src.generation.prompt_templates import build_rag_prompt

PROCESSED_DIR = Path("data/processed")
QUESTIONS_PATH = PROCESSED_DIR / "eval_questions.jsonl"
SEMANTIC_CHUNKS_PATH = PROCESSED_DIR / "semantic_cluster.jsonl"


def load_eval_questions(path: str = str(QUESTIONS_PATH), num_questions: int = 200) -> List[Dict]:
    """Load evaluation questions (same file/selection for every system).

    Raises ValueError when the file holds no VERIFIED questions: per the
    matching policy, metrics must never run on unverified or empty sets.
    """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[:num_questions]
    if not questions:
        raise ValueError(
            f"No verified evaluation questions in {path}. Generate them first with "
            f"'python -m src.evaluation.filter_questions' (only corpus-verified "
            f"PubMedQA matches are written; an empty result means no overlap)."
        )
    return questions


def load_chunk_map(chunk_file: str = str(SEMANTIC_CHUNKS_PATH)) -> Dict[str, str]:
    """Build chunk_id -> article_id mapping from a chunk JSONL file."""
    mapping = {}
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                mapping[chunk["chunk_id"]] = chunk["article_id"]
    return mapping


def evaluate_system_retrieval(
    questions: List[Dict],
    chunk_ids_fn: Callable[[str], List[str]],
    chunk_map: Dict[str, str],
    run_name: str,
) -> Dict:
    """Run article-level retrieval metrics for one system; return means + df."""
    df = evaluate_retrieval(questions, chunk_ids_fn, chunk_map, run_name=run_name)
    mean_row = df[df["pubmed_id"] == "MEAN"].iloc[0]
    return {
        "recall_5": float(mean_row["recall_5"]),
        "recall_10": float(mean_row["recall_10"]),
        "mrr": float(mean_row["mrr"]),
        "df": df,
    }


def comparison_frame(system_means: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Per-system means table: rows Metric, one column per system."""
    rows: List[Dict[str, object]] = []
    for metric in ("recall_5", "recall_10", "mrr"):
        row: Dict[str, object] = {"Metric": {"recall_5": "Recall@5", "recall_10": "Recall@10", "mrr": "MRR"}[metric]}
        for name, means in system_means.items():
            row[name] = means[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_comparison_frame(baseline_means: Dict[str, float], modified_means: Dict[str, float]) -> pd.DataFrame:
    """Metric,Baseline,Full_System,Delta table (sidebar-compatible schema)."""
    return pd.DataFrame({
        "Metric": ["Recall@5", "Recall@10", "MRR"],
        "Baseline": [baseline_means["recall_5"], baseline_means["recall_10"], baseline_means["mrr"]],
        "Full_System": [modified_means["recall_5"], modified_means["recall_10"], modified_means["mrr"]],
        "Delta": [
            modified_means["recall_5"] - baseline_means["recall_5"],
            modified_means["recall_10"] - baseline_means["recall_10"],
            modified_means["mrr"] - baseline_means["mrr"],
        ],
    })


def run_generation_for_system(
    questions: List[Dict],
    retrieve_full_fn: Callable[[str], List[dict]],
    llm_client,
) -> tuple:
    """Generate answers with one system's retrieved context.

    Returns (generated_answers, gold_answers, valid_questions).
    """
    from src.evaluation.generation_metrics import evaluate_generation  # lazy: heavy deps

    generated, gold, valid = [], [], []
    for i, q in enumerate(questions, 1):
        print(f"Generating answer {i}/{len(questions)}...", end=" ", flush=True)
        try:
            chunks = retrieve_full_fn(q["question"])[:10]
            prompt = build_rag_prompt(q["question"], chunks)
            try:
                answer = llm_client.generate(prompt)
            except Exception as e:
                print(f"Retrying after error: {e}")
                time.sleep(2)
                answer = llm_client.generate(prompt)
            generated.append(answer)
            gold.append(q["long_answer"])
            valid.append(q)
            print("done")
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if not generated:
        return [], [], []
    df = evaluate_generation(generated, gold, valid)
    mean_row = df[df["pubmed_id"] == "MEAN"].iloc[0]
    return generated, gold, {
        "rouge_l_f1": float(mean_row["rouge_l_f1"]),
        "bertscore_f1": float(mean_row["bertscore_f1"]),
        "n": len(generated),
    }


def plot_retrieval_comparison(system_means: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Grouped bar chart of Recall@5/Recall@10/MRR across systems."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["Recall@5", "Recall@10", "MRR"]
    keys = ["recall_5", "recall_10", "mrr"]
    names = [n for n in SYSTEM_ORDER if n in system_means]
    x = np.arange(len(metrics))
    width = max(0.12, min(0.8 / max(len(names), 1), 0.35))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(names):
        vals = [system_means[name][k] for k in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics by System")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_comparison(
    driver,
    embedding_model,
    llm_client=None,
    num_questions: int = 200,
    num_gen_questions: int = 20,
    systems: Optional[List[str]] = None,
    gen_systems: Optional[List[str]] = None,
    top_k: int = 10,
    out_dir: str = str(PROCESSED_DIR),
) -> Dict:
    """Run the full baseline-vs-modified (+ablation) comparison.

    Returns {"retrieval": {system: means}, "generation": {system: means}}.
    """
    systems = list(SYSTEM_ORDER) if systems is None else list(systems)
    gen_names: List[str] = ["baseline", "adaptive"] if gen_systems is None else list(gen_systems)
    for _s in systems + gen_names:
        resolve_system(_s)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3 EVALUATION — baseline vs modified (+ ablation)")
    print("=" * 60)
    questions = load_eval_questions(num_questions=num_questions)
    print(f"Loaded {len(questions)} questions (identical for every system)")
    chunk_map = load_chunk_map()
    print(f"Chunk map: {len(chunk_map)} chunks")

    # Retrieval: every system, same questions.
    retrieval_means = {}
    for name in systems:
        print(f"\n--- Retrieval: {name} ---")
        res = evaluate_system_retrieval(
            questions,
            lambda q, _n=name: [c["chunk_id"] for c in run_system(driver, embedding_model, q, top_k, _n)],
            chunk_map,
            run_name=name,
        )
        retrieval_means[name] = {k: res[k] for k in ("recall_5", "recall_10", "mrr")}
        print(f"{name}: R@5={res['recall_5']:.3f} R@10={res['recall_10']:.3f} MRR={res['mrr']:.3f}")

    comparison_frame(retrieval_means).to_csv(out / "eval_systems.csv", index=False)
    print("Saved eval_systems.csv")
    if "baseline" in retrieval_means and "adaptive" in retrieval_means:
        legacy_comparison_frame(retrieval_means["baseline"], retrieval_means["adaptive"]).to_csv(
            out / "eval_comparison.csv", index=False
        )
        print("Saved eval_comparison.csv (sidebar-compatible)")
    plot_retrieval_comparison(retrieval_means, out / "retrieval_comparison.png")

    # Generation: headline systems only (bounds LLM cost).
    generation_means = {}
    if llm_client is not None:
        for name in gen_names:
            print(f"\n--- Generation: {name} ---")
            _, _, means = run_generation_for_system(
                questions[:num_gen_questions],
                lambda q, _n=name: run_system(driver, embedding_model, q, 10, _n),
                llm_client,
            )
            if means:
                generation_means[name] = means
        if generation_means:
            pd.DataFrame([
                {"System": n, "ROUGE-L F1": m["rouge_l_f1"],
                 "BERTScore F1": m["bertscore_f1"], "n": m["n"]}
                for n, m in generation_means.items()
            ]).to_csv(out / "eval_generation.csv", index=False)
            print("Saved eval_generation.csv")
    else:
        print("\nSkipping generation (no LLM client provided)")

    print("\nEVALUATION COMPLETE — no claim is made beyond the measured tables.")
    return {"retrieval": retrieval_means, "generation": generation_means}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 baseline-vs-modified evaluation")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-gen-questions", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--systems", type=str, default=",".join(SYSTEM_ORDER),
                        help="Comma-separated subset of: " + ",".join(SYSTEM_ORDER))
    parser.add_argument("--gen-systems", type=str, default="baseline,adaptive")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from neo4j import GraphDatabase
    from src.config import settings
    from src.generation.llm_client import LLMClient
    from src.embeddings import get_embedder

    _driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    _model = get_embedder()
    _llm = None if args.skip_generation else LLMClient()
    try:
        run_comparison(
            _driver, _model, _llm,
            num_questions=args.num_questions,
            num_gen_questions=args.num_gen_questions,
            systems=[s.strip() for s in args.systems.split(",") if s.strip()],
            gen_systems=[s.strip() for s in args.gen_systems.split(",") if s.strip()],
            top_k=args.top_k,
        )
    finally:
        _driver.close()
