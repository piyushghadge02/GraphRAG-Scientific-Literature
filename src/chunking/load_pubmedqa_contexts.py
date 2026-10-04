"""
PubMedQA Context Loader

Load unique context passages from qiaojin/PubMedQA (pqa_labeled config)
and save them as a JSONL file in the same schema as abstracts_sample.jsonl.

Each unique context passage becomes a separate entry with:
  - article_id: "pqa_{pubid}" (where pubid is from PubMedQA)
  - title: "" (empty, as PubMedQA doesn't provide titles)
  - abstract_text: the context passage

This allows PubMedQA questions to be evaluated against our graph using
the source material they reference.
"""

import json
from pathlib import Path
from typing import List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_PATH = RAW_DIR / "pubmedqa_contexts.jsonl"


def load_pubmedqa_contexts() -> List[dict]:
    """
    Load PubMedQA dataset and extract unique context passages.

    Returns:
        List of dicts with keys: article_id, title, abstract_text
    """
    from datasets import load_dataset

    print("[load_pubmedqa_contexts] Loading qiaojin/PubMedQA (pqa_labeled)...")
    try:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    except Exception as e:
        print(f"[load_pubmedqa_contexts] ERROR loading dataset: {e}")
        print("Make sure you have: pip install datasets")
        raise

    records: List[dict] = []
    seen_contexts: Set[str] = set()

    for example in dataset:
        pubid = example.get("pubid", "")
        context = example.get("context", "")

        # Extract context (could be a list or single string)
        if isinstance(context, list):
            context_text = " ".join(context)
        else:
            context_text = str(context) if context else ""

        context_text = context_text.strip()

        # Skip empty contexts
        if not context_text:
            continue

        # Create a unique key to avoid duplicates
        # (same context from different questions should be deduped)
        context_key = f"{pubid}:{context_text[:100]}"
        if context_key in seen_contexts:
            continue
        seen_contexts.add(context_key)

        # Create record in same schema as abstracts
        article_id = f"pqa_{pubid}"
        records.append({
            "article_id": article_id,
            "title": "",
            "abstract_text": context_text,
        })

    print(f"[load_pubmedqa_contexts] Extracted {len(records)} unique context passages")
    return records


def save_pubmedqa_contexts(records: List[dict]) -> None:
    """
    Save context records to JSONL cache.

    Args:
        records: List of context dicts to save
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[load_pubmedqa_contexts] Saved {len(records)} contexts → {CACHE_PATH}")


def main():
    """Load and save PubMedQA contexts."""
    print("=" * 80)
    print("  PubMedQA Context Loader")
    print("=" * 80)

    contexts = load_pubmedqa_contexts()
    save_pubmedqa_contexts(contexts)

    print(f"\nTotal contexts loaded: {len(contexts)}")
    if contexts:
        sample = contexts[0]
        print(f"Sample article_id: {sample['article_id']}")
        print(f"Sample context (first 200 chars): {sample['abstract_text'][:200]}…")

    print("=" * 80)


if __name__ == "__main__":
    main()
