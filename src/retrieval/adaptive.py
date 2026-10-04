"""
Phase 2 — Adaptive Retrieval (distinctive feature).

Decides per query whether graph expansion is likely useful, instead of
always running the same pipeline. Deterministic, rule-based, explainable —
no ML classifier.

Signals (all cheap, computed from the initial vector results + query text):
- top-1 similarity and the top1-top2 gap: a dominant, high-scoring top hit
  means vector retrieval is already confident -> skip expansion.
- multi-concept cues in the query ("and", "vs", "relationship between",
  ...): the question spans related concepts -> force expansion.
- seed count: with no seeds there is nothing to expand from -> skip.

Decision output is a dict with:
    expand (bool), confidence (str), reasons (list[str]), metrics (dict)

Configure via ADAPTIVE_RETRIEVAL_ENABLED (master switch),
ADAPTIVE_TOP1_THRESHOLD and ADAPTIVE_GAP_THRESHOLD. The expand_graph flag
on retrieve() remains the hard master switch: False always means
vector-only, regardless of the adaptive decision.
"""

from typing import Dict, List, Optional

from src.config import settings

# Query-text cues suggesting the question spans multiple related concepts,
# where graph traversal is most likely to help. Kept deliberately small and
# transparent; matched case-insensitively as substrings.
MULTI_CONCEPT_CUES = (
    " and ",
    " vs ",
    " vs. ",
    " versus ",
    "compare",
    "comparison",
    "relationship between",
    "relation between",
    "difference between",
    "interaction between",
    "interplay between",
)


def find_concept_cues(query: str) -> List[str]:
    """Return the multi-concept cues present in *query* (lowercased match)."""
    lowered = f" {query.lower()} "
    return [cue.strip() for cue in MULTI_CONCEPT_CUES if cue in lowered]


def expansion_decision(
    vector_results: List[dict],
    query: str,
    enabled: Optional[bool] = None,
    top1_threshold: Optional[float] = None,
    gap_threshold: Optional[float] = None,
) -> Dict:
    """Decide whether graph expansion should run for this query.

    Args:
        vector_results: Ranked vector-search hits, each with a "score" key
            (descending). May be empty.
        query: The original user query (used for cue matching).
        enabled: Master switch override (default: ADAPTIVE_RETRIEVAL_ENABLED).
        top1_threshold: Skip expansion when top1 >= this AND the gap >=
            gap_threshold (default: ADAPTIVE_TOP1_THRESHOLD).
        gap_threshold: See above (default: ADAPTIVE_GAP_THRESHOLD).

    Returns:
        Dict with keys: expand, confidence, reasons, metrics.
    """
    on = settings.ADAPTIVE_RETRIEVAL_ENABLED if enabled is None else bool(enabled)
    top1 = float(vector_results[0].get("score", 0.0)) if vector_results else 0.0
    top2 = float(vector_results[1].get("score", 0.0)) if len(vector_results) > 1 else 0.0
    gap = top1 - top2
    mean = sum(float(r.get("score", 0.0)) for r in vector_results) / max(len(vector_results), 1)
    cues = find_concept_cues(query)
    metrics = {
        "n_seeds": len(vector_results),
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "mean_score": mean,
        "concept_cues": cues,
    }

    if not on:
        return {
            "expand": True,
            "confidence": "n/a",
            "reasons": ["adaptive retrieval disabled -> legacy always-expand behavior"],
            "metrics": metrics,
        }
    if not vector_results:
        return {
            "expand": False,
            "confidence": "none",
            "reasons": ["no vector seed chunks -> nothing to expand from"],
            "metrics": metrics,
        }
    if cues:
        return {
            "expand": True,
            "confidence": "multi-concept",
            "reasons": [f"query spans related concepts (cues: {cues}) -> graph expansion useful"],
            "metrics": metrics,
        }

    t1 = settings.ADAPTIVE_TOP1_THRESHOLD if top1_threshold is None else float(top1_threshold)
    gt = settings.ADAPTIVE_GAP_THRESHOLD if gap_threshold is None else float(gap_threshold)
    if top1 >= t1 and gap >= gt:
        return {
            "expand": False,
            "confidence": "high",
            "reasons": [
                f"top1={top1:.4f} >= {t1:.2f} with gap={gap:.4f} >= {gt:.2f} "
                f"-> vector retrieval already confident, vector-focused path"
            ],
            "metrics": metrics,
        }
    return {
        "expand": True,
        "confidence": "low",
        "reasons": [
            f"top1={top1:.4f}/gap={gap:.4f} below thresholds "
            f"({t1:.2f}/{gt:.2f}) -> graph expansion may add evidence"
        ],
        "metrics": metrics,
    }
