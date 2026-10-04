"""
Phase 3 — Evaluation systems (baseline vs modified + ablation).

Defines the named retrieval configurations compared in evaluation and
exposed in the Streamlit UI. All presets run against the SAME questions
and the SAME dataset; only the retrieval behavior differs.

Systems:
- vector_only : vector search only (no graph). Ablation anchor.
- baseline    : PHASE 3 BASELINE — required semantic chunking, all-MiniLM-L6-v2,
  384D, vector retrieval + required graph expansion, legacy 2-term fusion
  (alpha 0.7 / beta 0.3 / gamma 0.0), no PageRank, no adaptive skipping.
  NOTE: this differs from the legacy run_eval.py "Run A", which is an
  in-memory vector-only search over fixed_token chunks. The Phase 3 baseline
  is the ablated graph system; do not compare numbers across the two
  runners without stating which baseline was used.
- fused       : baseline + 3-term graph-aware fusion with configured weights.
- pagerank    : fused + subgraph-scoped PageRank reranking.
- adaptive    : OUR modified GraphRAG — fused + PageRank + adaptive retrieval.

UI modes map onto these presets (see UI_MODES).
"""

from typing import Dict, List, Optional

from src.retrieval.retrieve import retrieve
from src.retrieval.retrieve_with_pagerank import retrieve_with_pagerank

# Evaluation/UI ablation order (also the column order in reports).
SYSTEM_ORDER = ["vector_only", "baseline", "fused", "pagerank", "adaptive"]

# Streamlit mode labels -> system preset names.
UI_MODES = {
    "Modified GraphRAG": "adaptive",
    "Baseline": "baseline",
}

SYSTEM_PRESETS: Dict[str, dict] = {
    "vector_only": {
        "label": "A. Vector-only",
        "description": "Vector search only; graph expansion disabled.",
        "use_pagerank": False,
        "retrieve_kwargs": {"expand_graph": False},
    },
    "baseline": {
        "label": "B. Phase 3 baseline (vector + graph)",
        "description": (
            "Required expansion behavior with legacy fusion "
            "(alpha=0.7, beta=0.3, gamma=0.0); no PageRank, no adaptive skipping. "
            "Distinct from legacy run_eval Run A (fixed_token in-memory vector-only)."
        ),
        "use_pagerank": False,
        "retrieve_kwargs": {
            "expand_graph": True,
            "adaptive_enabled": False,
            "alpha": 0.7,
            "beta": 0.3,
            "gamma": 0.0,
        },
    },
    "fused": {
        "label": "C. Vector + graph + improved fusion",
        "description": "3-term graph-aware fusion with configured weights.",
        "use_pagerank": False,
        "retrieve_kwargs": {"expand_graph": True, "adaptive_enabled": False},
    },
    "pagerank": {
        "label": "D. Vector + graph + PageRank",
        "description": "Fused ranking plus subgraph-scoped PageRank reranking.",
        "use_pagerank": True,
        "retrieve_kwargs": {"expand_graph": True, "adaptive_enabled": False},
    },
    "adaptive": {
        "label": "E. Adaptive GraphRAG (ours)",
        "description": "Fused ranking + PageRank + adaptive retrieval decision.",
        "use_pagerank": True,
        "retrieve_kwargs": {"expand_graph": True, "adaptive_enabled": True},
    },
}


def resolve_system(name: str) -> dict:
    """Return the preset dict for *name* or raise ValueError."""
    try:
        return SYSTEM_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown evaluation system '{name}'. "
            f"Choose from: {sorted(SYSTEM_PRESETS)}"
        ) from None


def run_system(driver, embedding_model, query: str, top_k: int, system: str) -> List[dict]:
    """Run one named system for a single query; return full chunk dicts."""
    preset = resolve_system(system)
    if preset["use_pagerank"]:
        return retrieve_with_pagerank(
            driver=driver,
            embedding_model=embedding_model,
            query=query,
            top_k=top_k,
            **preset["retrieve_kwargs"],
        )
    return retrieve(
        driver=driver,
        embedding_model=embedding_model,
        query=query,
        top_k=top_k,
        **preset["retrieve_kwargs"],
    )


def describe_systems(systems: Optional[List[str]] = None) -> List[dict]:
    """Return [{name, label, description}] for the given (or all) systems."""
    names = list(SYSTEM_ORDER) if systems is None else list(systems)
    return [
        {"name": n, "label": SYSTEM_PRESETS[n]["label"],
         "description": SYSTEM_PRESETS[n]["description"]}
        for n in names
    ]
