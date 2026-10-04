"""
Deterministic text-based matching between PubMedQA questions and the
5,000-abstract corpus.

WHY THIS EXISTS: the corpus loader (ccdv/pubmed-summarization) exposes only
'article'/'abstract' columns — no PubMed IDs — and mints synthetic IDs
(pubmed_<index>_<hash>). PubMedQA questions reference pqa_<pubid>. These ID
spaces can never match, so evaluation gold IDs must come from VERIFIED
content overlap instead of ID equality.

STRATEGY (conservative, no ML/fuzzy pipeline):
- normalize (lowercase + collapse whitespace, punctuation kept) both sides
- a PubMedQA context passage (normalized length >= MIN_PASSAGE_CHARS)
  must appear VERBATIM as a substring of a normalized corpus abstract
- first corpus hit in deterministic corpus order wins; multi-hits counted

Only questions with such a verified match may become evaluation questions,
with the matched corpus article_id stored as the gold ID.
"""

import random
import re
from typing import Dict, List, Optional, Tuple

# Passages shorter than this are ignored: tiny strings match spuriously.
MIN_PASSAGE_CHARS = 80


def normalize_text(text: str) -> str:
    """Canonical form for matching: lowercase, punctuation removed, single spaces.

    Punctuation must go (not just whitespace): the corpus abstracts are
    tokenizer-spaced ("cells .") while PubMedQA contexts are raw ("cells."),
    so punctuation-sensitive comparison can never match across the two
    sources. With the MIN_PASSAGE_CHARS floor, collisions remain implausible.
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_context_passages(example: dict) -> List[str]:
    """Pull raw context passage strings from a PubMedQA example.

    Handles the observed dict-with-'contexts' shape as well as plain
    list/str shapes (defensive; mirrors load_pubmedqa_contexts).
    """
    context = example.get("context", "")
    if isinstance(context, dict):
        context = context.get("contexts", "")
    if isinstance(context, list):
        return [str(c) for c in context if str(c).strip()]
    if isinstance(context, str) and context.strip():
        return [context]
    return []


def build_corpus_index(abstracts: List[dict]) -> List[Tuple[str, str]]:
    """Precompute [(article_id, normalized_abstract)] in input order."""
    return [
        (a["article_id"], normalize_text(a.get("abstract_text", "")))
        for a in abstracts
    ]


def match_question(
    passages: List[str],
    corpus_index: List[Tuple[str, str]],
    min_passage_chars: int = MIN_PASSAGE_CHARS,
) -> Optional[Tuple[str, str]]:
    """Return (article_id, matched_passage_preview) for the first verified hit.

    A hit requires a normalized passage of at least min_passage_chars to be
    a verbatim substring of a normalized corpus abstract. None when no
    passage verifies.
    """
    for passage in passages:
        norm_passage = normalize_text(passage)
        if len(norm_passage) < min_passage_chars:
            continue
        for article_id, norm_abstract in corpus_index:
            if not norm_abstract or len(norm_abstract) < len(norm_passage):
                continue
            if norm_passage in norm_abstract:
                preview = norm_passage[:120]
                return article_id, preview
    return None


def match_all(
    examples: List[dict],
    abstracts: List[dict],
    min_passage_chars: int = MIN_PASSAGE_CHARS,
    progress_every: int = 200,
) -> Tuple[Dict[str, dict], dict]:
    """Match every PubMedQA example against the corpus.

    Returns ({pubid: {"article_id", "passage_preview"}}, stats) where stats
    holds totals + multi-hit count. Deterministic: corpus order decides ties.
    """
    corpus_index = build_corpus_index(abstracts)
    matched: Dict[str, dict] = {}
    multi_hits = 0
    for i, example in enumerate(examples):
        if progress_every and i and i % progress_every == 0:
            print(f"[match] {i}/{len(examples)} examples checked, {len(matched)} matched ...")
        pubid = str(example.get("pubid", ""))
        if not pubid:
            continue
        passages = extract_context_passages(example)
        hit = match_question(passages, corpus_index, min_passage_chars)
        if hit is None:
            continue
        article_id, preview = hit
        # Count questions whose passage occurs in more than one abstract.
        norm_passages = [normalize_text(p) for p in passages if len(normalize_text(p)) >= min_passage_chars]
        hits = sum(
            1 for _, na in corpus_index
            if any(np_ in na for np_ in norm_passages)
        )
        if hits > 1:
            multi_hits += 1
        matched[pubid] = {"article_id": article_id, "passage_preview": preview}
    stats = {
        "total": len(examples),
        "matched": len(matched),
        "unmatched": len(examples) - len(matched),
        "multi_hit_questions": multi_hits,
    }
    return matched, stats


def select_matched(
    matched: Dict[str, dict],
    n: int = 200,
    seed: int = 42,
) -> List[Tuple[str, dict]]:
    """Deterministically select up to n matched (pubid, info) pairs.

    Sort by pubid, shuffle with a seeded RNG, take the first n. Never
    duplicates or fabricates: returns fewer than n when fewer match.
    """
    items = sorted(matched.items(), key=lambda kv: kv[0])
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def match_by_article_presence(
    examples: List[dict],
    chunk_article_ids,
) -> Tuple[Dict[str, dict], dict]:
    """Fast exact verification against an already-chunked document set.

    A question is verified iff its ``pqa_<pubid>`` article has at least one
    chunk in the chunk file (i.e. its source material survived chunking and
    is retrievable). Returns ({pubid: {"article_id", ...}}, stats) in the
    same shape as match_all() so both paths share selection/record code.
    """
    chunk_set = set(chunk_article_ids)
    matched: Dict[str, dict] = {}
    for example in examples:
        pubid = str(example.get("pubid", ""))
        if not pubid:
            continue
        article_id = f"pqa_{pubid}"
        if article_id in chunk_set:
            matched[pubid] = {"article_id": article_id, "passage_preview": ""}
    stats = {
        "total": len(examples),
        "matched": len(matched),
        "unmatched": len(examples) - len(matched),
        "multi_hit_questions": 0,
    }
    return matched, stats


def make_eval_record(example: dict, pubid: str, article_id: str) -> dict:
    """Build one eval_questions.jsonl record with the VERIFIED corpus gold ID.

    'pubmed_id' is the corpus article_id the metrics compare against;
    PubMedQA identifiers are preserved separately for traceability.
    """
    return {
        "question": example.get("question", ""),
        "pubmed_id": article_id,
        "pqa_pubid": pubid,
        "pqa_article_id": f"pqa_{pubid}",
        "long_answer": example.get("long_answer", ""),
        "final_decision": example.get("final_decision", ""),
    }
