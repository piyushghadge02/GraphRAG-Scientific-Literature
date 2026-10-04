"""
Task 3 — Score Fusion (Phase 2: 3-term normalized graph-aware ranking).

Phase 1 API (combined_score, explain_fusion, rank_candidates) is preserved
unchanged for backward compatibility.

Phase 2 adds a normalized 3-term formulation:

    final = alpha * vector_score
          + beta  * entity_score
          + gamma * proximity_score

where every component is in [0, 1] by construction:
- vector_score: cosine similarity clipped to [0, 1]
- entity_score: shared_entities / max_shared_entities (pool-normalized)
- proximity_score: 1 / (1 + depth); seeds (depth 0) score 1.0, depth-1
  expansion 0.5, depth-2 expansion 0.33

Weights resolve from settings (ALPHA_FUSION_WEIGHT, FUSION_BETA_WEIGHT,
FUSION_GAMMA_WEIGHT), accept per-call overrides, and are rescaled to sum
to 1 so no component dominates by scale. Defaults (0.7/0.3/0.0) reproduce
the Phase 1 ranking exactly; set FUSION_GAMMA_WEIGHT > 0 to make graph
distance count. No superiority is claimed before evaluation.
"""

from typing import Dict, List, Optional, Tuple

from src.config import settings

DEFAULT_ALPHA = 0.7


def _resolve_alpha(alpha: float | None) -> float:
    """Resolve the fusion weight: explicit value wins, else config default."""
    value = settings.ALPHA_FUSION_WEIGHT if alpha is None else alpha
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {value}")
    return float(value)


def explain_fusion(
    vector_score: float,
    shared_entities: int,
    max_shared_entities: int,
    alpha: float | None = None,
) -> dict:
    """Return the score breakdown and a human-readable explanation.

    Keys: vector_score, shared_entities, max_shared_entities, alpha,
    vector_component, graph_component, entity_score, combined, explanation.
    """
    weight = _resolve_alpha(alpha)
    max_entities = max(int(max_shared_entities), 1)  # Avoid division by zero
    shared = max(int(shared_entities), 0)
    entity_score = shared / max_entities

    vector_component = weight * float(vector_score)
    graph_component = (1.0 - weight) * entity_score
    combined = vector_component + graph_component

    explanation = (
        f"combined={combined:.4f} = {weight:.2f}*vector({float(vector_score):.4f}) "
        f"+ {1.0 - weight:.2f}*graph({shared}/{max_entities}={entity_score:.4f})"
    )
    return {
        "vector_score": float(vector_score),
        "shared_entities": shared,
        "max_shared_entities": max_entities,
        "alpha": weight,
        "vector_component": vector_component,
        "graph_component": graph_component,
        "entity_score": entity_score,
        "combined": combined,
        "explanation": explanation,
    }


def combined_score(
    vector_score: float,
    shared_entities: int,
    max_shared_entities: int,
    alpha: float | None = None,
) -> float:
    """Combine vector similarity and entity overlap into a unified score.

    Formula:
        combined = alpha * vector_score + (1 - alpha) * (shared_entities / max(max_shared_entities, 1))

    Args:
        vector_score: Similarity score from vector search (typically 0.0 to 1.0)
        shared_entities: Number of shared entities linking this chunk to seed chunks
        max_shared_entities: Maximum shared_entities count in the candidate pool (for normalization)
        alpha: Weight for vector score. Defaults to settings.ALPHA_FUSION_WEIGHT
            (env ALPHA_FUSION_WEIGHT, default 0.7, so 70% vector, 30% graph).
            Passing the historical default 0.7 explicitly reproduces the
            original behavior exactly.

    Returns:
        Combined score as float between 0.0 and 1.0
    """
    # NOTE: default changed from hard-coded 0.7 to None->config so the weight
    # is configurable via environment; passing alpha=0.7 keeps exact legacy behavior.
    if alpha is None:
        # Fall back to the historical default when settings are unavailable.
        try:
            alpha = settings.ALPHA_FUSION_WEIGHT
        except Exception:
            alpha = DEFAULT_ALPHA
    return explain_fusion(vector_score, shared_entities, max_shared_entities, alpha)["combined"]


def rank_candidates(
    candidates: List[dict],
    alpha: float | None = None,
) -> List[dict]:
    """Rank candidate dicts by fused score (descending), attaching breakdowns.

    Each candidate must have: vector_score, shared_entities (plus chunk_id/text
    passthrough). Adds: combined_score, score_breakdown (dict), score_explanation
    (str). Sort is stable and identical to sorting by combined_score alone.

    Returns a NEW sorted list; input dicts are updated in place with the new keys.
    """
    weight = _resolve_alpha(alpha) if alpha is not None else _resolve_alpha(None)
    max_shared = max((int(c.get("shared_entities", 0)) for c in candidates), default=0)

    for candidate in candidates:
        breakdown = explain_fusion(
            vector_score=float(candidate.get("vector_score", 0.0)),
            shared_entities=int(candidate.get("shared_entities", 0)),
            max_shared_entities=max(max_shared, 1),
            alpha=weight,
        )
        candidate["combined_score"] = breakdown["combined"]
        candidate["score_breakdown"] = breakdown
        candidate["score_explanation"] = breakdown["explanation"]

    return sorted(candidates, key=lambda x: x["combined_score"], reverse=True)


# ── Phase 2: normalized 3-term fusion ────────────────────────────────

def normalize_weights(
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Resolve fusion weights from per-call overrides or settings.

    Each weight must be in [0, 1] and their sum must be > 0. The triple is
    rescaled to sum to 1 so components stay comparable.
    """
    a = settings.ALPHA_FUSION_WEIGHT if alpha is None else alpha
    b = settings.FUSION_BETA_WEIGHT if beta is None else beta
    g = settings.FUSION_GAMMA_WEIGHT if gamma is None else gamma
    for name, value in (("alpha", a), ("beta", b), ("gamma", g)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    total = float(a) + float(b) + float(g)
    if total <= 0.0:
        raise ValueError("fusion weights must sum to more than 0")
    return float(a) / total, float(b) / total, float(g) / total


def proximity_score(depth: int = 0) -> float:
    """Graph-proximity component in [0, 1]: 1 / (1 + depth).

    Seeds (depth 0) score 1.0; depth-1 expansion 0.5; depth-2 expansion 0.33.
    Negative depths are treated as 0.
    """
    d = max(int(depth), 0)
    return 1.0 / (1.0 + d)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a {id: score} mapping to [0, 1].

    Empty input maps to {}; when all scores are equal every entry maps to
    1.0. Shared with PageRank post-processing so both paths normalize
    identically.
    """
    if not scores:
        return {}
    values = list(scores.values())
    hi, lo = max(values), min(values)
    if hi > lo:
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    return {k: 1.0 for k in scores}


def explain_fused(
    vector_score: float,
    shared_entities: int,
    max_shared_entities: int,
    depth: int = 0,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> dict:
    """Return the 3-term score breakdown plus a human-readable explanation."""
    a, b, g = normalize_weights(alpha, beta, gamma)
    vector = min(max(float(vector_score), 0.0), 1.0)
    max_entities = max(int(max_shared_entities), 1)
    shared = max(int(shared_entities), 0)
    entity = shared / max_entities
    proximity = proximity_score(depth)

    vector_component = a * vector
    entity_component = b * entity
    proximity_component = g * proximity
    combined = vector_component + entity_component + proximity_component

    explanation = (
        f"combined={combined:.4f} = {a:.2f}*vector({vector:.4f}) "
        f"+ {b:.2f}*entity({shared}/{max_entities}={entity:.4f}) "
        f"+ {g:.2f}*proximity(depth={max(int(depth), 0)}={proximity:.4f})"
    )
    return {
        "vector_score": vector,
        "shared_entities": shared,
        "max_shared_entities": max_entities,
        "depth": max(int(depth), 0),
        "alpha": a,
        "beta": b,
        "gamma": g,
        "vector_component": vector_component,
        "entity_component": entity_component,
        "proximity_component": proximity_component,
        "entity_score": entity,
        "proximity_score": proximity,
        "combined": combined,
        "explanation": explanation,
    }


def fused_score(
    vector_score: float,
    shared_entities: int,
    max_shared_entities: int,
    depth: int = 0,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> float:
    """3-term fused relevance score in [0, 1]. See explain_fused()."""
    return explain_fused(
        vector_score, shared_entities, max_shared_entities, depth, alpha, beta, gamma
    )["combined"]


def rank_candidates_fused(
    candidates: List[dict],
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
) -> List[dict]:
    """Rank candidates with the 3-term fused score (descending).

    Each candidate may carry: vector_score, shared_entities, depth,
    path_count (plus chunk_id/text passthrough). Adds: combined_score,
    score_breakdown (dict), score_explanation (str). With default weights
    (0.7/0.3/0.0) the ordering matches the Phase 1 rank_candidates().

    Returns a NEW sorted list; input dicts are updated in place.
    """
    a, b, g = normalize_weights(alpha, beta, gamma)
    max_shared = max((int(c.get("shared_entities", 0)) for c in candidates), default=0)

    for candidate in candidates:
        breakdown = explain_fused(
            vector_score=float(candidate.get("vector_score", 0.0)),
            shared_entities=int(candidate.get("shared_entities", 0)),
            max_shared_entities=max(max_shared, 1),
            depth=int(candidate.get("depth", 0)),
            alpha=a,
            beta=b,
            gamma=g,
        )
        candidate["combined_score"] = breakdown["combined"]
        candidate["score_breakdown"] = breakdown
        candidate["score_explanation"] = breakdown["explanation"]

    return sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
