"""
Task 3 — Chunking Orchestrator (CLI)
Run one or more chunking strategies over the 5000 PubMed abstracts and
write results to data/processed/{strategy}.jsonl.

Usage:
    python -m src.chunking.chunker --strategy all
    python -m src.chunking.chunker --strategy fixed_token
    python -m src.chunking.chunker --strategy sentence_boundary semantic_cluster
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List

import tiktoken
from tqdm import tqdm

from src.chunking.load_data import load_abstracts
from src.chunking import fixed_token, sentence_boundary, semantic_cluster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ENCODING = tiktoken.get_encoding("cl100k_base")

STRATEGY_MAP: dict[str, Callable] = {
    "fixed_token": fixed_token.chunk,
    "sentence_boundary": sentence_boundary.chunk,
    "semantic_cluster": semantic_cluster.chunk,
}


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def merge_article_chunks(article_id: str, chunks: List[dict]) -> List[dict]:
    """Merge one article's chunks into a single chunk (submission compliance).

    The semantic pipeline (embeddings + clustering) still runs per abstract;
    the resulting chunks are then joined into one chunk per article so the
    submitted graph holds ≈1 chunk per abstract (≈5k Chunks for 5k abstracts,
    as the assignment deliverable requires). Keeps the first chunk_id
    ("<article_id>_chunk_0") and the originating strategy label.
    """
    if not chunks:
        return []
    first = dict(chunks[0])
    first["chunk_id"] = f"{article_id}_chunk_0"
    first["text"] = " ".join(c.get("text", "") for c in chunks)
    return [first]


def run_strategy(
    strategy_name: str,
    chunk_fn: Callable,
    abstracts: List[dict],
    merge_per_article: bool = False,
) -> None:
    """Run a single strategy over all abstracts and write JSONL output."""
    output_path = OUTPUT_DIR / f"{strategy_name}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: List[dict] = []
    articles_with_chunks = 0

    for article in tqdm(abstracts, desc=f"  {strategy_name}", unit="article"):
        article_id = article["article_id"]
        text = article["abstract_text"]
        try:
            chunks = chunk_fn(article_id, text)
        except Exception as e:
            print(f"\n  [WARN] {strategy_name}: error on {article_id}: {e}")
            continue
        if merge_per_article:
            chunks = merge_article_chunks(article_id, chunks)
        if chunks:
            articles_with_chunks += 1
            all_chunks.extend(chunks)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary stats
    total_chunks = len(all_chunks)
    avg_chunks_per_article = total_chunks / max(articles_with_chunks, 1)
    token_lengths = [_token_count(c["text"]) for c in all_chunks]
    avg_tokens = sum(token_lengths) / max(len(token_lengths), 1)

    print(f"\n  ── {strategy_name} summary ──")
    print(f"     Total chunks:              {total_chunks}")
    print(f"     Articles with chunks:       {articles_with_chunks}")
    print(f"     Avg chunks / article:       {avg_chunks_per_article:.2f}")
    print(f"     Avg tokens / chunk:         {avg_tokens:.1f}")
    print(f"     Output:                     {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chunking strategies over PubMed abstracts."
    )
    parser.add_argument(
        "--strategy",
        nargs="+",
        default=["all"],
        choices=["all", "fixed_token", "sentence_boundary", "semantic_cluster"],
        help='Which strategy(ies) to run. "all" runs all three. (default: all)',
    )
    parser.add_argument(
        "--merge-per-article",
        action="store_true",
        help=(
            "Merge each article's chunks into one chunk per article. "
            "The selected chunking pipeline still runs; output is ≈1 "
            "chunk/article for the ≈5k-chunk submission deliverable."
        ),
    )
    args = parser.parse_args()

    # Resolve strategy list
    if "all" in args.strategy:
        strategies = list(STRATEGY_MAP.keys())
    else:
        strategies = args.strategy

    print("=" * 60)
    print("  GraphRAG Chunking Pipeline")
    print("=" * 60)

    # Load abstracts (the ONLY corpus documents; PubMedQA contexts must
    # never enter the chunking pipeline as corpus documents).
    abstracts = load_abstracts()
    print(f"\nLoaded {len(abstracts)} abstracts.\n")

    for name in strategies:
        print(f"▶ Running strategy: {name}")
        run_strategy(name, STRATEGY_MAP[name], abstracts,
                     merge_per_article=args.merge_per_article)

    print("=" * 60)
    print("  All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
