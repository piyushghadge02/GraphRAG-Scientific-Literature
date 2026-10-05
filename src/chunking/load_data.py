"""
Task 1 — Data Loading

Load PubMed abstracts from the ccdv/pubmed-summarization dataset on Hugging Face
(native Parquet format — no trust_remote_code needed, compatible with datasets v5+).

This replaces the original armanc/scientific_papers loader which required a custom
loading script that is no longer supported by the datasets library.

Sample 5000 abstracts with a fixed seed, cache to JSONL, and expose load_abstracts().

PMID enrichment: Includes PubMedQA source abstracts fetched by PMID from Entrez in batches.
"""

import hashlib
import json
import random
import time
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_PATH = RAW_DIR / "abstracts_sample.jsonl"

# ccdv/pubmed-summarization is a Parquet-native mirror of the PubMed portion of
# the scientific_papers benchmark.  Columns: "article" (full text) and "abstract".
# Two configs exist ("section" and "document") — both share the same abstracts;
# we use "document" which is the closest match to the original dataset layout.
DATASET_NAME = "ccdv/pubmed-summarization"
DATASET_CONFIG = "document"
SAMPLE_SIZE = 5000
SEED = 42

# Entrez settings
ENTREZ_EMAIL = "graphrag@example.com"
ENTREZ_BATCH_DELAY = 1.0  # ~1 request/second for batch efetch


def _stable_id(index: int, text: str) -> str:
    """Generate a deterministic article ID from position + content hash."""
    digest = hashlib.sha256(f"{index}:{text[:200]}".encode()).hexdigest()[:12]
    return f"pubmed_{index}_{digest}"


def compose_sample(matched_indices, total_rows: int, sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> List[int]:
    """Compose a deterministic sample: all matched indices + seeded random fill.

    Returns sorted original row indices (corpus order). Never exceeds
    sample_size; matched papers are prioritized so evaluation questions keep
    their source articles.
    """
    matched = sorted(set(matched_indices))
    if len(matched) >= sample_size:
        return matched[:sample_size]
    remaining = [i for i in range(total_rows) if i not in set(matched)]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    return sorted(matched + remaining[: sample_size - len(matched)])


def _fetch_pubmedqa_pmids() -> List[int]:
    """Fetch unique PubMed IDs from PubMedQA pqa_labeled dataset."""
    from datasets import load_dataset
    print("[load_data] Loading PubMedQA pqa_labeled for PMIDs...")
    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    pmids = set()
    for ex in pqa:
        pubid = ex.get("pubid")
        if pubid:
            try:
                pmids.add(int(pubid))
            except (ValueError, TypeError):
                pass
    print(f"[load_data] Found {len(pmids)} unique PubMedQA PMIDs")
    return sorted(pmids)


def _fetch_pubmed_abstracts_batch(pmids: List[int]) -> List[dict]:
    """Fetch PubMed abstracts for a batch of PMIDs using Entrez efetch."""
    from Bio import Entrez
    Entrez.email = ENTREZ_EMAIL
    
    records = []
    # Entrez efetch can take multiple IDs at once
    id_string = ",".join(str(p) for p in pmids)
    
    try:
        handle = Entrez.efetch(db="pubmed", id=id_string, rettype="abstract", retmode="xml")
        xml_data = Entrez.read(handle)
        
        articles = xml_data.get("PubmedArticle", [])
        for art in articles:
            try:
                medline = art.get("MedlineCitation", {})
                article = medline.get("Article", {})
                pmid = medline.get("PMID", "")
                
                # Get abstract
                abstract = article.get("Abstract", {})
                abstract_parts = abstract.get("AbstractText", [])
                if isinstance(abstract_parts, list):
                    abstract_text = " ".join(str(p) for p in abstract_parts)
                elif isinstance(abstract_parts, str):
                    abstract_text = abstract_parts
                else:
                    abstract_text = ""
                
                # Get title
                title = article.get("ArticleTitle", "")
                
                if abstract_text and abstract_text.strip() and pmid:
                    records.append({
                        "article_id": f"pmid_{pmid}",
                        "pmid": int(pmid),
                        "title": title.strip() if title else "",
                        "abstract_text": abstract_text.strip(),
                    })
            except Exception as e:
                print(f"[load_data] WARNING: Failed to parse article: {e}")
    
    except Exception as e:
        print(f"[load_data] WARNING: Batch efetch failed for {pmids[:3]}...: {e}")
    
    return records


def _fetch_pubmedqa_source_abstracts() -> List[dict]:
    """Fetch PubMed abstracts for all PubMedQA PMIDs in batches."""
    pmids = _fetch_pubmedqa_pmids()
    total = len(pmids)
    batch_size = 200  # Entrez allows up to ~200 IDs per efetch
    
    all_records = []
    for i in range(0, total, batch_size):
        batch = pmids[i:i+batch_size]
        print(f"[load_data] Fetching batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} PMIDs)...")
        batch_records = _fetch_pubmed_abstracts_batch(batch)
        all_records.extend(batch_records)
        print(f"[load_data]   Got {len(batch_records)} abstracts from this batch")
        if i + batch_size < total:
            time.sleep(ENTREZ_BATCH_DELAY)
    
    print(f"[load_data] Total PubMedQA source abstracts fetched: {len(all_records)}/{total}")
    return all_records


def _download_ccdv_abstracts(exclude_pmids: set, count: int) -> List[dict]:
    """Download ccdv abstracts, excluding any with PMIDs we already have."""
    from datasets import load_dataset
    
    print(f"[load_data] Downloading ccdv pool for {count} fill abstracts...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    total = len(ds)
    
    # Shuffle indices
    rng = random.Random(SEED)
    indices = list(range(total))
    rng.shuffle(indices)
    
    records = []
    for orig_idx in indices:
        if len(records) >= count:
            break
        row = ds[orig_idx]
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            continue
        title = ""
        for candidate in ("title", "Title", "TITLE"):
            if candidate in ds.column_names and candidate in row:
                title = row[candidate] or ""
                break
        # Generate synthetic ID for ccdv abstracts
        article_id = _stable_id(len(records), abstract)
        records.append({
            "article_id": article_id,
            "pmid": None,
            "title": title,
            "abstract_text": abstract,
        })
    return records


def _build_eval_ready_corpus() -> List[dict]:
    """Build the 5000-abstract corpus with PubMedQA source abstracts + ccdv fill."""
    
    # Step 1: Fetch PubMedQA source abstracts by PMID
    pqa_records = _fetch_pubmedqa_source_abstracts()
    fetched_count = len(pqa_records)
    fetched_pmids = {r["pmid"] for r in pqa_records}
    
    # Step 2: Fill remaining slots with ccdv abstracts
    remaining = SAMPLE_SIZE - fetched_count
    if remaining > 0:
        print(f"[load_data] Need {remaining} ccdv abstracts for fill")
        ccdv_records = _download_ccdv_abstracts(fetched_pmids, remaining)
        print(f"[load_data] Got {len(ccdv_records)} ccdv fill abstracts")
    else:
        ccdv_records = []
    
    # Combine and shuffle deterministically
    all_records = pqa_records + ccdv_records
    rng = random.Random(SEED)
    rng.shuffle(all_records)
    
    # Take exactly SAMPLE_SIZE
    final_records = all_records[:SAMPLE_SIZE]
    
    # Save cache
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    pmid_count = sum(1 for r in final_records if r.get("pmid"))
    print(f"[load_data] Built corpus: {len(final_records)} abstracts ({pmid_count} with PMIDs)")
    return final_records


def _download_and_cache() -> List[dict]:
    """Download dataset, sample, and write cache JSONL (basic version without PMIDs)."""
    from datasets import load_dataset

    print(f"[load_data] Downloading '{DATASET_NAME}' (config='{DATASET_CONFIG}') …")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    # Use the train split (pubmed-summarization exposes train / validation / test)
    split = ds.get("train", ds[list(ds.keys())[0]])
    print(f"[load_data] Split has {len(split)} rows.  Sampling {SAMPLE_SIZE} …")

    sampled = split.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(split))))

    # Inspect available columns and map defensively
    columns = sampled.column_names
    print(f"[load_data] Dataset columns: {columns}")

    records: List[dict] = []
    for i, row in enumerate(sampled):
        # Abstract text — required
        try:
            abstract = row["abstract"]
        except KeyError:
            print(f"[load_data] WARNING: 'abstract' field missing at index {i}, skipping.")
            continue

        if not abstract or not abstract.strip():
            continue

        # Title — optional (pubmed-summarization does NOT have a title column)
        title = ""
        for candidate in ("title", "Title", "TITLE"):
            if candidate in row:
                title = row[candidate] or ""
                break

        article_id = _stable_id(i, abstract)

        records.append({
            "article_id": article_id,
            "pmid": None,
            "title": title,
            "abstract_text": abstract.strip(),
        })

    # Persist to cache
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[load_data] Cached {len(records)} abstracts → {CACHE_PATH}")
    return records


def load_abstracts() -> List[dict]:
    """
    Return a list of dicts with keys: article_id, title, abstract_text, pmid (optional).
    Uses a JSONL cache to avoid re-downloading on repeated calls.
    """
    if CACHE_PATH.exists():
        print(f"[load_data] Loading cached abstracts from {CACHE_PATH}")
        records = []
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"[load_data] Loaded {len(records)} abstracts from cache.")
        return records

    return _build_eval_ready_corpus()


def sample_eval_ready_corpus(sample_size: int = SAMPLE_SIZE, seed: int = SEED) -> List[dict]:
    """Build the eval-ready corpus: PubMedQA source abstracts + ccdv fill."""
    return _build_eval_ready_corpus()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load or (re)sample the abstract corpus.")
    parser.add_argument(
        "--eval-ready",
        action="store_true",
        help="Build the 5000-abstract corpus with PubMedQA source abstracts.",
    )
    args = parser.parse_args()
    if args.eval_ready:
        abstracts = _build_eval_ready_corpus()
    else:
        abstracts = load_abstracts()
    print(f"\nTotal abstracts loaded: {len(abstracts)}")
    pmid_count = sum(1 for a in abstracts if a.get("pmid"))
    print(f"Abstracts with PMIDs: {pmid_count}")
    if abstracts:
        sample = abstracts[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Sample article_id: {sample['article_id']}")
        print(f"Sample PMID: {sample.get('pmid', 'None')}")
        print(f"Sample abstract (first 200 chars): {sample['abstract_text'][:200]}…")