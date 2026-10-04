#!/usr/bin/env python3
"""Phase 1 lightweight tests — embedding architecture + retrieval foundation.

Covers:
  1. Default model is all-MiniLM-L6-v2.
  2. Default embedding dimension is 384.
  3. Generated (chunk) embeddings have the expected dimension.
  4. Query embeddings have the expected dimension.
  5. Configuration can change the embedding model.
  6. Retrieval accepts embeddings from the configured model.
  7. Fusion/ranking still works (backward compatible + explainable).
  8. No hard-coded old dimension assumptions remain where they should not.

No live Neo4j/API services required. No model download required: tests use
small fake embedders and fake drivers instead of SentenceTransformer/Neo4j.
Run with:  python tests/test_phase1_embeddings.py
(or: python -m pytest tests/test_phase1_embeddings.py -q, if pytest is installed)
"""

import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import embeddings
from src.config import BASELINE_EMBEDDING_MODEL, settings

# NOTE: use importlib.import_module (not `from src.retrieval import X`) because
# src/retrieval/__init__.py re-exports function names (vector_search, retrieve)
# that would otherwise shadow the submodule objects we need to patch/call.
fusion = importlib.import_module("src.retrieval.fusion")
vector_search = importlib.import_module("src.retrieval.vector_search")
retrieve_mod = importlib.import_module("src.retrieval.retrieve")

BASELINE_MODEL = BASELINE_EMBEDDING_MODEL  # central constant; must stay all-MiniLM-L6-v2
BASELINE_DIM = 384
CUSTOM_DIM = 768


# ---------------------------------------------------------------------------
# Fakes (no model download, no Neo4j)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Minimal SentenceTransformer stand-in with a programmable dimension."""

    def __init__(self, dim: int):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)


class MismatchedEmbedder(FakeEmbedder):
    """Reports one dimension but encodes another (simulates model drift)."""

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(BASELINE_DIM, dtype=np.float32)
        return np.ones((len(texts), BASELINE_DIM), dtype=np.float32)


class FakeSession:
    def __init__(self, records):
        self._records = records

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, *args, **kwargs):
        return self._records


class FakeDriver:
    def __init__(self, records):
        self._records = records

    def session(self):
        return FakeSession(self._records)


def _load_module_from_path(module_name: str, relative_path: str):
    """Load a module directly from file, bypassing heavy package __init__."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"No spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_neo4j_stub():
    if "neo4j" not in sys.modules:
        stub = types.ModuleType("neo4j")

        class _Driver:
            pass

        class _GraphDatabase:
            @staticmethod
            def driver(*args, **kwargs):
                return _Driver()

        stub.__dict__["GraphDatabase"] = _GraphDatabase
        sys.modules["neo4j"] = stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Phase1EmbeddingTests(unittest.TestCase):
    def test_1_default_model_is_baseline(self):
        self.assertEqual(BASELINE_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(settings.EMBEDDING_MODEL_NAME, BASELINE_MODEL)
        self.assertEqual(settings.SEMANTIC_CLUSTER_MODEL_NAME, BASELINE_MODEL)
        self.assertIs(embeddings.BASELINE_EMBEDDING_MODEL, BASELINE_EMBEDDING_MODEL)
        self.assertEqual(embeddings.get_retrieval_model_name(), BASELINE_MODEL)
        self.assertEqual(embeddings.get_cluster_model_name(), BASELINE_MODEL)
        # Single central definition: the literal appears exactly once in config.py.
        config_src = (PROJECT_ROOT / "src" / "config.py").read_text(encoding="utf-8")
        self.assertEqual(config_src.count('"all-MiniLM-L6-v2"'), 1)
        # No other shipped module defines its own baseline literal in code.
        emb_src = (PROJECT_ROOT / "src" / "embeddings.py").read_text(encoding="utf-8")
        self.assertNotIn('"all-MiniLM-L6-v2"', emb_src)

    def test_2_default_dimension_is_384(self):
        self.assertEqual(embeddings.BASELINE_EMBEDDING_DIM, BASELINE_DIM)
        # Fast path: baseline dimension without loading any model.
        self.assertEqual(embeddings.get_embedding_dimension(BASELINE_MODEL), BASELINE_DIM)
        self.assertEqual(embeddings.get_chunk_embedding_dimension(), BASELINE_DIM)
        cfg = embeddings.get_embedding_config()
        self.assertEqual(cfg["dimension"], BASELINE_DIM)
        self.assertTrue(cfg["is_baseline"])
        self.assertEqual(cfg["experiment"], "A (baseline)")
        self.assertEqual(cfg["vector_index"]["dimensions"], BASELINE_DIM)
        self.assertEqual(cfg["vector_index"]["similarity_function"], "cosine")

    def test_3_chunk_embeddings_have_expected_dimension(self):
        out = embeddings.encode_texts(["a", "b", "c"], embedder=FakeEmbedder(BASELINE_DIM))
        self.assertEqual(out.shape, (3, BASELINE_DIM))
        # End-to-end through the real pipeline helper (no model download:
        # stub out encode_texts inside compute_similarity).
        sim_mod = _load_module_from_path(
            "phase1_compute_similarity_test", "src/graph/compute_similarity.py"
        )
        with patch.object(
            sim_mod, "encode_texts", return_value=np.ones((2, BASELINE_DIM), dtype=np.float32)
        ):
            result = sim_mod.compute_chunk_embeddings(["x", "y"])
        self.assertEqual(result.shape, (2, BASELINE_DIM))

    def test_4_query_embeddings_have_expected_dimension(self):
        out = embeddings.encode_query("what is insulin?", embedder=FakeEmbedder(BASELINE_DIM))
        self.assertEqual(len(out), BASELINE_DIM)

    def test_4b_explicit_model_validates_against_itself(self):
        # An explicitly passed custom model must NOT be validated against the
        # configured (baseline) retrieval dimension.
        out = embeddings.encode_query(
            "what is insulin?",
            model_name="custom-model",
            embedder=FakeEmbedder(CUSTOM_DIM),
        )
        self.assertEqual(len(out), CUSTOM_DIM)
        out = embeddings.encode_texts(
            ["a", "b"], model_name="custom-model", embedder=FakeEmbedder(CUSTOM_DIM)
        )
        self.assertEqual(out.shape, (2, CUSTOM_DIM))
        # ... but a model whose output disagrees with its reported dimension fails.
        with self.assertRaises(ValueError):
            embeddings.encode_texts(["a"], embedder=MismatchedEmbedder(CUSTOM_DIM))
        with self.assertRaises(ValueError):
            embeddings.encode_query("q", embedder=MismatchedEmbedder(CUSTOM_DIM))

    def test_5_configuration_can_change_model(self):
        custom_settings = SimpleNamespace(
            EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2",
            SEMANTIC_CLUSTER_MODEL_NAME=BASELINE_MODEL,  # clustering stays pinned
        )
        with patch.object(embeddings, "settings", custom_settings), patch.object(
            embeddings, "get_embedder", return_value=FakeEmbedder(CUSTOM_DIM)
        ):
            self.assertEqual(
                embeddings.get_retrieval_model_name(),
                "sentence-transformers/all-mpnet-base-v2",
            )
            self.assertEqual(embeddings.get_embedding_dimension(), CUSTOM_DIM)
            self.assertEqual(embeddings.get_chunk_embedding_dimension(), CUSTOM_DIM)
            cfg = embeddings.get_embedding_config()
            self.assertEqual(cfg["dimension"], CUSTOM_DIM)
            self.assertFalse(cfg["is_baseline"])
            self.assertEqual(cfg["experiment"], "B (custom)")
            # Clustering model is unaffected by the retrieval-model switch.
            self.assertEqual(embeddings.get_cluster_model_name(), BASELINE_MODEL)

    def test_6_retrieval_accepts_configured_model_embeddings(self):
        # vector_search passes custom-dim embeddings through when they match.
        records = [
            {"chunk_id": "c1", "text": "t1", "score": 0.9},
            {"chunk_id": "c2", "text": "t2", "score": 0.8},
        ]
        out = vector_search.vector_search(
            FakeDriver(records), [0.1] * CUSTOM_DIM, top_k=2, expected_dim=CUSTOM_DIM
        )
        self.assertEqual([r["chunk_id"] for r in out], ["c1", "c2"])
        # ... and rejects a dimension mismatch with a rebuild/reindex hint.
        with self.assertRaises(ValueError) as ctx:
            vector_search.vector_search(
                FakeDriver(records), [0.1] * BASELINE_DIM, expected_dim=CUSTOM_DIM
            )
        self.assertIn("rebuild", str(ctx.exception).lower())

        # retrieve() works end-to-end with a custom-dim model (no expansion).
        fake_model = FakeEmbedder(CUSTOM_DIM)
        with patch.object(retrieve_mod, "get_chunk_embedding_dimension", return_value=CUSTOM_DIM), \
             patch.object(retrieve_mod, "validate_embedding_dim", return_value=CUSTOM_DIM):
            results = retrieve_mod.retrieve(
                FakeDriver(records), fake_model, "query", top_k=2, expand_graph=False
            )
        self.assertEqual(len(results), 2)
        for r in results:
            for key in ("chunk_id", "text", "vector_score", "shared_entities", "combined_score"):
                self.assertIn(key, r)

    def test_6b_retrieve_rejects_incompatible_supplied_model(self):
        # Configured dim is 384 here; a supplied 768D model must be rejected
        # with a message pointing at the configured model (no silent misuse).
        records = [{"chunk_id": "c1", "text": "t1", "score": 0.9}]
        with self.assertRaises(ValueError) as ctx:
            retrieve_mod.retrieve(
                FakeDriver(records), FakeEmbedder(CUSTOM_DIM), "query",
                top_k=1, expand_graph=False,
            )
        self.assertIn(BASELINE_MODEL, str(ctx.exception))

    def test_7_fusion_ranking_still_works(self):
        # Legacy behavior preserved exactly.
        score = fusion.combined_score(0.9, 3, 5, alpha=0.7)
        self.assertAlmostEqual(score, 0.7 * 0.9 + 0.3 * (3 / 5), places=6)
        # Default alpha comes from config (0.7) and matches explicit 0.7.
        self.assertAlmostEqual(
            fusion.combined_score(0.9, 3, 5), fusion.combined_score(0.9, 3, 5, alpha=0.7), places=9
        )
        # Explainable breakdown adds up and documents itself.
        breakdown = fusion.explain_fusion(0.9, 3, 5, alpha=0.7)
        self.assertAlmostEqual(
            breakdown["combined"],
            breakdown["vector_component"] + breakdown["graph_component"],
            places=9,
        )
        self.assertIn("combined=", breakdown["explanation"])
        # rank_candidates orders by combined score and annotates each entry.
        # a: 0.7*0.5 + 0.3*(4/4) = 0.65 ; b: 0.7*0.95 + 0.3*(0/4) = 0.665
        ranked = fusion.rank_candidates(
            [
                {"chunk_id": "a", "text": "A", "vector_score": 0.5, "shared_entities": 4},
                {"chunk_id": "b", "text": "B", "vector_score": 0.95, "shared_entities": 0},
            ],
            alpha=0.7,
        )
        self.assertEqual([c["chunk_id"] for c in ranked], ["b", "a"])
        for c in ranked:
            self.assertIn("combined_score", c)
            self.assertIn("score_breakdown", c)
            self.assertIn("score_explanation", c)
        # Weight is configurable per call and validated.
        low = fusion.combined_score(0.9, 0, 5, alpha=0.1)
        high = fusion.combined_score(0.9, 0, 5, alpha=0.9)
        self.assertLess(low, high)
        with self.assertRaises(ValueError):
            fusion.combined_score(0.9, 1, 5, alpha=1.5)

    def test_8_no_hardcoded_dimension_assumptions(self):
        def read(rel):
            return (PROJECT_ROOT / rel).read_text(encoding="utf-8")

        schema_src = read("src/graph/schema.py")
        self.assertNotIn("vector.dimensions`: 384", schema_src)
        self.assertIn("get_chunk_embedding_dimension", schema_src)

        vs_src = read("src/retrieval/vector_search.py")
        self.assertNotIn("(384 dims for all-MiniLM-L6-v2)", vs_src)

        sim_src = read("src/graph/compute_similarity.py")
        self.assertNotIn("Generate 384-dimensional", sim_src)

        cluster_src = read("src/chunking/semantic_cluster.py")
        self.assertIn("SEMANTIC_CLUSTER_MODEL_NAME", cluster_src)
        self.assertNotIn("SentenceTransformer(settings.EMBEDDING_MODEL_NAME)", cluster_src)

        # The vector index follows configuration: 384 by default ...
        _ensure_neo4j_stub()
        schema_mod = _load_module_from_path("phase1_schema_test", "src/graph/schema.py")
        cypher, _desc = schema_mod.build_vector_index_cypher()
        self.assertIn("`vector.dimensions`: 384", cypher)
        self.assertIn("chunk_embedding", cypher)
        # ... and tracks a custom model (single index, no second index).
        with patch.object(schema_mod, "get_chunk_embedding_dimension", return_value=CUSTOM_DIM):
            custom_cypher, _ = schema_mod.build_vector_index_cypher()
        self.assertIn(f"`vector.dimensions`: {CUSTOM_DIM}", custom_cypher)
        self.assertNotIn("384", custom_cypher)
        self.assertEqual(custom_cypher.count("CREATE VECTOR INDEX"), 1)

    def test_8b_vector_index_recreation_is_explicit(self):
        # recreate_vector_index() issues exactly one DROP + one CREATE against
        # the single 'chunk_embedding' index; plain creation never drops.
        # (stdout is captured: src prints ✓/✗ glyphs that non-UTF-8 Windows
        # consoles cannot encode — a pre-existing codebase-wide trait.)
        import contextlib
        import io

        _ensure_neo4j_stub()
        schema_mod = _load_module_from_path("phase1_schema_recreate_test", "src/graph/schema.py")

        seen = []

        class RecordingSession:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, cypher, **kwargs):
                seen.append(cypher)
                return []

        class RecordingDriver:
            def session(self):
                return RecordingSession()

        with contextlib.redirect_stdout(io.StringIO()):
            schema_mod.recreate_vector_index(RecordingDriver())
        self.assertEqual(len(seen), 2)
        self.assertIn("DROP INDEX", seen[0])
        self.assertIn("chunk_embedding", seen[0])
        self.assertIn("CREATE VECTOR INDEX", seen[1])
        self.assertIn("`vector.dimensions`: 384", seen[1])

        seen.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            schema_mod.create_constraints_and_indexes(RecordingDriver())
        self.assertTrue(all("DROP INDEX" not in c for c in seen))
        self.assertEqual(sum(c.count("CREATE VECTOR INDEX") for c in seen), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
