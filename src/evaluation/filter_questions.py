"""
Evaluation question generation with VERIFIED corpus matches.

Pipeline:
1. Load the 5,000-abstract submission corpus (local cache; synthetic pubmed_* IDs).
   This is the ONLY match set: gold IDs must be articles present in the
   submitted Neo4j graph. PubMedQA context passages are the evaluation/QA
   side, never an additional corpus here.
2. Load PubMedQA pqa_labeled/train (question + context passages + pubid).
3. Keep ONLY questions whose context matches a corpus abstract verbatim
   (see src/evaluation/match_corpus.py — ID equality is impossible because
   the corpus dataset carries no PubMed IDs).
4. Deterministically sample up to 200 matched questions (seed=42).
5. Write data/processed/eval_questions.jsonl with the corpus article_id as
   the gold 'pubmed_id' (what retrieval metrics compare against) and the
   PubMedQA identifiers preserved separately for traceability.

If fewer than 200 questions match, all valid matches are saved as-is —
never duplicated or fabricated.
"""

import json
from pathlib import Path

from src.chunking.load_data import load_abstracts
from src.evaluation.match_corpus import make_eval_record, match_all, match_by_article_presence, select_matched


def filter_and_save_questions(output_path: str = "data/processed/eval_questions.jsonl", num_samples: int = 200, seed: int = 42):
    """
    Build the evaluation question file from corpus-verified PubMedQA matches.

    Args:
        output_path: Path to save verified questions
        num_samples: Maximum questions to select (default: 200)
        seed: Random seed for reproducible sampling (default: 42)

    Returns:
        List of written eval records.
    """
    from datasets import load_dataset

    print("Loading 5,000-abstract corpus...")
    abstracts = load_abstracts()
    print(f"Corpus abstracts: {len(abstracts)}")

    print("Loading PubMedQA dataset (pqa_labeled/train)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    examples = [dataset[i] for i in range(len(dataset))]
    print(f"Total PubMedQA questions: {len(examples)}")

    matched, stats = match_all(examples, abstracts)
    print(f"Matched to corpus: {stats['matched']}")
    print(f"Unmatched: {stats['unmatched']}")
    print(f"Multi-hit questions (first corpus hit kept): {stats['multi_hit_questions']}")

    selected = select_matched(matched, n=num_samples, seed=seed)
    print(f"Selected for evaluation: {len(selected)} (requested up to {num_samples}, seed={seed})")
    if len(selected) < num_samples:
        print("WARNING: fewer verified matches than requested — saving all valid matches, no duplication.")

    by_pubid = {str(examples[i].get("pubid", "")): examples[i] for i in range(len(examples))}
    records = [make_eval_record(by_pubid[pubid], pubid, info["article_id"]) for pubid, info in selected]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} verified questions to {output_path}")
    print("Example mappings (pqa_pubid -> corpus article_id):")
    for pubid, info in selected[:5]:
        print(f"  {pubid} -> {info['article_id']}")

    return records


def collect_chunk_articles(chunk_path: str = "data/processed/semantic_cluster.jsonl") -> set:
    """Collect distinct article_ids present in a chunk JSONL file."""
    article_ids = set()
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                article_ids.add(json.loads(line)["article_id"])
    return article_ids


def filter_from_chunked_graph(output_path: str = "data/processed/eval_questions.jsonl", num_samples: int = 200, seed: int = 42,
                              chunk_path: str = "data/processed/semantic_cluster.jsonl"):
    """Fast path: verify PubMedQA questions against the chunk file directly.

    No re-chunking, no passage matching: a question is evaluation-ready iff
    its pqa_<pubid> article has chunks in the existing chunk file. Selection
    and record format are identical to filter_and_save_questions().
    """
    from datasets import load_dataset

    print(f"Scanning chunk file: {chunk_path} ...")
    chunk_articles = collect_chunk_articles(chunk_path)
    print(f"Distinct chunked articles: {len(chunk_articles)}")

    print("Loading PubMedQA dataset (pqa_labeled/train)...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    examples = [dataset[i] for i in range(len(dataset))]
    print(f"Total PubMedQA questions: {len(examples)}")

    matched, stats = match_by_article_presence(examples, chunk_articles)
    print(f"Verified against chunks: {stats['matched']}")
    print(f"Without chunks: {stats['unmatched']}")

    selected = select_matched(matched, n=num_samples, seed=seed)
    print(f"Selected for evaluation: {len(selected)} (requested up to {num_samples}, seed={seed})")
    if len(selected) < num_samples:
        print("WARNING: fewer verified matches than requested — saving all valid matches, no duplication.")

    by_pubid = {str(ex.get("pubid", "")): ex for ex in examples}
    records = [make_eval_record(by_pubid[pubid], pubid, info["article_id"]) for pubid, info in selected]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} verified questions to {output_path}")
    print("Example mappings (pqa_pubid -> chunked article_id):")
    for pubid, info in selected[:5]:
        print(f"  {pubid} -> {info['article_id']}")

    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build verified evaluation questions.")
    parser.add_argument("--from-chunks", action="store_true",
                        help="Verify against the existing chunk file instead of passage matching.")
    args = parser.parse_args()
    if args.from_chunks:
        filter_from_chunked_graph()
    else:
        filter_and_save_questions()
