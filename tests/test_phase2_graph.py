#!/usr/bin/env python3
"""Phase 2 lightweight tests — graph improvements + adaptive retrieval.

Covers:
  1. Entity normalization.
  2. Entity deduplication (+ deterministic type resolution).
  3. Graph expansion depth evidence + config defaults.
  4. Duplicate candidate removal.
  5. Graph/entity score calculation (3-term fusion).
  6. Score normalization (fusion + PageRank helper).
  7. Configurable fusion weights (incl. legacy-equivalent defaults).
  8. PageRank score normalization.
  9. PageRank-disabled retrieval path.
  10. Adaptive retrieval decision (+ pipeline integration).
  11. Vector-only retrieval still works.
  12. Vector + graph retrieval still works.

No live Neo4j/spacy/GDS required: fake drivers, fake embedders and stub
spaCy modules are used throughout.
Run with:  python tests/test_phase2_graph.py
"""

import importlib
import importlib.util
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

from src.config import settings

fusion = importlib.import_module("src.retrieval.fusion")
adaptive = importlib.import_module("src.retrieval.adaptive")
graph_expand = importlib.import_module("src.retrieval.graph_expand")
gds_rerank = importlib.import_module("src.retrieval.gds_rerank")
retrieve_mod = importlib.import_module("src.retrieval.retrieve")
retrieve_pr = importlib.import_module("src.retrieval.retrieve_with_pagerank")

DIM = 384


def _load_module_from_path(module_name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"No spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_stubs():
    if "neo4j" not in sys.modules:
        stub = types.ModuleType("neo4j")

        class _GraphDatabase:
            @staticmethod
            def driver(*args, **kwargs):
                raise RuntimeError("no live neo4j in tests")

        stub.__dict__["GraphDatabase"] = _GraphDatabase
        sys.modules["neo4j"] = stub
    if "spacy" not in sys.modules:
        spacy_stub = types.ModuleType("spacy")
        spacy_stub.__dict__["load"] = lambda *a, **k: object()
        sys.modules["spacy"] = spacy_stub
    if "scispacy" not in sys.modules:
        sys.modules["scispacy"] = types.ModuleType("scispacy")


_ensure_stubs()
entity_mod = _load_module_from_path(
    "phase2_entity_extraction_test", "src/graph/entity_extraction.py"
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbedder:
    def __init__(self, dim=DIM):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)


def _fake_ent(text, label="ENTITY"):
    return SimpleNamespace(text=text, label_=label)


def _fake_doc(*ents):
    return SimpleNamespace(ents=list(ents))


class FakeSession:
    """Routes run() by query content; records every cypher issued."""

    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **kwargs):
        self.driver.queries.append(cypher)
        self.driver.last_kwargs = kwargs
        if "db.index.vector" in cypher:
            return list(self.driver.vector_records)
        if "COUNT(DISTINCT e)" in cypher:
            return _Single({"count": self.driver.entity_count})
        return list(self.driver.expansion_records)

    def single(self):  # pragma: no cover (single() used on run() results instead)
        raise AssertionError("use _Single")


class _Single:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row

    def __iter__(self):
        return iter([self._row])


class FakeDriver:
    def __init__(self, vector_records=None, expansion_records=None, entity_count=3):
        self.vector_records = vector_records or []
        self.expansion_records = expansion_records or []
        self.entity_count = entity_count
        self.queries = []
        self.last_kwargs = {}

    def session(self):
        return FakeSession(self)

    def ran(self, fragment):
        return any(fragment in q for q in self.queries)


def _vec(score, cid):
    return {"chunk_id": cid, "text": f"text {cid}", "score": score}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class Phase2EntityTests(unittest.TestCase):
    def test_1_normalization(self):
        n = entity_mod.normalize_entity_name
        self.assertEqual(n("  Insulin   Resistance "), "insulin resistance")
        self.assertEqual(n("Patient's cells"), "patient cells")
        self.assertEqual(n("(IL-2),"), "il-2")
        self.assertEqual(n("\u201cp53\u201d"), "p53")
        self.assertEqual(n("Diabetes\nMellitus"), "diabetes mellitus")
        self.assertEqual(n("  "), "")

    def test_1b_noise_filter(self):
        self.assertTrue(entity_mod.is_noise_entity(""))
        self.assertTrue(entity_mod.is_noise_entity("x"))
        self.assertTrue(entity_mod.is_noise_entity("22"))
        self.assertTrue(entity_mod.is_noise_entity("1983"))
        self.assertTrue(entity_mod.is_noise_entity("a" * 200))
        self.assertFalse(entity_mod.is_noise_entity("il-2"))
        self.assertFalse(entity_mod.is_noise_entity("p53"))
        # Scientific distinctions preserved: no stemming/plural folding.
        self.assertNotEqual(
            entity_mod.normalize_entity_name("tumors"),
            entity_mod.normalize_entity_name("tumour"),
        )

    def test_2_dedup_and_type_resolution(self):
        doc = _fake_doc(
            _fake_ent("Insulin  Resistance", "CHEM"),
            _fake_ent("insulin resistance", "DISEASE"),
            _fake_ent("22", "NUM"),
        )
        collected = entity_mod._collect_from_doc(doc)
        self.assertEqual(collected, {"insulin resistance": "CHEM"})  # first type wins

        out = entity_mod.dedupe_entities([
            {"name": "  P53 ", "type": "GENE"},
            {"name": "p53", "type": "OTHER"},
            {"name": "22", "type": "NUM"},
            {"name": "BRCA1", "type": ""},
        ])
        self.assertEqual(
            out, [{"name": "p53", "type": "GENE"}, {"name": "brca1", "type": "ENTITY"}]
        )


class Phase2ExpansionTests(unittest.TestCase):
    def test_3_depth_evidence_and_defaults(self):
        self.assertEqual(settings.GRAPH_MAX_DEPTH, 1)
        self.assertEqual(settings.GRAPH_MIN_SHARED_ENTITIES, 2)
        self.assertEqual(settings.GRAPH_MAX_EXPANDED_CHUNKS, 500)
        self.assertAlmostEqual(settings.GRAPH_MIN_SIMILARITY, 0.3)

        recs = [
            {"chunk_id": "e1", "text": "t", "embedding": [1, 0, 0, 0],
             "shared_entities": 3, "path_count": 5, "depth": 1},
        ]
        driver = FakeDriver(expansion_records=recs)
        out = graph_expand.expand_via_entities(
            driver, ["s1"], query_embedding=[1, 0, 0, 0]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["depth"], 1)
        self.assertEqual(out[0]["path_count"], 5)
        self.assertEqual(out[0]["shared_entities"], 3)
        self.assertIn("$min_shared", driver.queries[0])

    def test_3b_depth2_and_validation(self):
        recs = [
            {"chunk_id": "e2", "text": "t", "embedding": None,
             "shared_entities": 4, "path_count": 7, "depth": 2},
        ]
        driver = FakeDriver(expansion_records=recs)
        out = graph_expand.expand_via_entities(driver, ["s1"], max_depth=2)
        self.assertEqual(out[0]["depth"], 2)
        with self.assertRaises(ValueError):
            graph_expand.expand_via_entities(driver, ["s1"], max_depth=3)
        with self.assertRaises(ValueError):
            graph_expand.expand_via_entities(driver, ["s1"], min_shared_entities=0)
        # Empty seeds: no query issued.
        driver2 = FakeDriver(expansion_records=recs)
        self.assertEqual(graph_expand.expand_via_entities(driver2, []), [])
        self.assertEqual(driver2.queries, [])

    def test_4_dedup_and_similarity_filter(self):
        recs = [
            {"chunk_id": "e1", "text": "t", "embedding": [1, 0, 0, 0],
             "shared_entities": 2, "path_count": 2, "depth": 1},
            {"chunk_id": "e1", "text": "t", "embedding": [1, 0, 0, 0],
             "shared_entities": 2, "path_count": 2, "depth": 1},  # duplicate row
            {"chunk_id": "e2", "text": "t", "embedding": [0, 1, 0, 0],
             "shared_entities": 5, "path_count": 9, "depth": 1},  # sim 0 -> dropped
        ]
        driver = FakeDriver(expansion_records=recs)
        out = graph_expand.expand_via_entities(
            driver, ["s1"], query_embedding=[1, 0, 0, 0]
        )
        self.assertEqual([r["chunk_id"] for r in out], ["e1"])

    def test_4b_per_call_overrides(self):
        driver = FakeDriver(expansion_records=[])
        graph_expand.expand_via_entities(driver, ["s1"], min_shared_entities=4)
        self.assertEqual(driver.last_kwargs.get("min_shared"), 4)
        driver2 = FakeDriver(expansion_records=[])
        graph_expand.expand_via_entities(driver2, ["s1"], max_candidates=10)
        self.assertEqual(driver2.last_kwargs.get("max_candidates"), 10)


class Phase2FusionTests(unittest.TestCase):
    def test_5_three_term_scores(self):
        # Defaults reproduce the legacy 2-term value exactly.
        self.assertAlmostEqual(
            fusion.fused_score(0.9, 3, 5), 0.7 * 0.9 + 0.3 * (3 / 5), places=9
        )
        self.assertAlmostEqual(
            fusion.fused_score(0.9, 3, 5),
            fusion.combined_score(0.9, 3, 5, alpha=0.7),
            places=9,
        )
        # Proximity differentiates depth when gamma > 0.
        near = fusion.fused_score(0.8, 2, 4, depth=0, gamma=0.5, alpha=0.25, beta=0.25)
        far = fusion.fused_score(0.8, 2, 4, depth=2, gamma=0.5, alpha=0.25, beta=0.25)
        self.assertGreater(near, far)

    def test_6_normalization(self):
        self.assertEqual(fusion.normalize_scores({}), {})
        self.assertEqual(fusion.normalize_scores({"a": 2.0, "b": 2.0}), {"a": 1.0, "b": 1.0})
        out = fusion.normalize_scores({"a": 1.0, "b": 3.0, "c": 2.0})
        self.assertAlmostEqual(out["a"], 0.0)
        self.assertAlmostEqual(out["b"], 1.0)
        self.assertAlmostEqual(out["c"], 0.5)

    def test_7_configurable_weights(self):
        self.assertEqual(settings.FUSION_BETA_WEIGHT, 0.3)
        self.assertEqual(settings.FUSION_GAMMA_WEIGHT, 0.0)
        a, b, g = fusion.normalize_weights(0.5, 0.5, 0.5)
        self.assertAlmostEqual(a + b + g, 1.0)
        self.assertAlmostEqual(a, 1 / 3)
        with self.assertRaises(ValueError):
            fusion.normalize_weights(-0.1, 0.5, 0.5)
        with self.assertRaises(ValueError):
            fusion.normalize_weights(0.0, 0.0, 0.0)
        self.assertAlmostEqual(fusion.proximity_score(0), 1.0)
        self.assertAlmostEqual(fusion.proximity_score(1), 0.5)
        self.assertAlmostEqual(fusion.proximity_score(2), 1 / 3)

    def test_7b_rank_fused(self):
        ranked = fusion.rank_candidates_fused([
            {"chunk_id": "a", "text": "A", "vector_score": 0.5,
             "shared_entities": 4, "depth": 1, "path_count": 6},
            {"chunk_id": "b", "text": "B", "vector_score": 0.95,
             "shared_entities": 0, "depth": 0, "path_count": 0},
        ])
        self.assertEqual([c["chunk_id"] for c in ranked], ["b", "a"])
        for c in ranked:
            self.assertIn("combined_score", c)
            self.assertIn("score_breakdown", c)
            self.assertIn("score_explanation", c)
        # Legacy ranker untouched.
        legacy = fusion.rank_candidates([
            {"chunk_id": "a", "text": "A", "vector_score": 0.5, "shared_entities": 4},
            {"chunk_id": "b", "text": "B", "vector_score": 0.95, "shared_entities": 0},
        ])
        self.assertEqual([c["chunk_id"] for c in legacy], ["b", "a"])


class Phase2PageRankTests(unittest.TestCase):
    def test_8_normalization_helper(self):
        # Shared helper: same normalization as the fusion path.
        self.assertIs(gds_rerank.normalize_scores, fusion.normalize_scores)
        self.assertEqual(fusion.normalize_scores({}), {})
        self.assertEqual(
            fusion.normalize_scores({"a": 5.0, "b": 5.0}), {"a": 1.0, "b": 1.0}
        )

    def test_9_disabled_path(self):
        driver = FakeDriver(vector_records=[_vec(0.9, "c1"), _vec(0.8, "c2")])
        out = retrieve_pr.retrieve_with_pagerank(
            driver, FakeEmbedder(), "query", top_k=2,
            expand_graph=False, use_pagerank=False,
        )
        self.assertEqual(len(out), 2)
        for r in out:
            self.assertEqual(r["pagerank_score"], 0.0)
            self.assertEqual(r["final_score"], r["combined_score"])

    def test_9b_enabled_blend_no_gds(self):
        # Fake driver has no GDS -> graceful {} -> final = 0.85 * combined.
        driver = FakeDriver(vector_records=[_vec(0.9, "c1"), _vec(0.8, "c2")])
        out = retrieve_pr.retrieve_with_pagerank(
            driver, FakeEmbedder(), "query", top_k=2,
            expand_graph=False, use_pagerank=True,
        )
        for r in out:
            self.assertAlmostEqual(r["final_score"], 0.85 * r["combined_score"], places=9)
            self.assertEqual(r["pagerank_weight"], 0.15)
        with self.assertRaises(ValueError):
            retrieve_pr.retrieve_with_pagerank(
                FakeDriver(vector_records=[_vec(0.9, "c1")]), FakeEmbedder(),
                "query", top_k=1, expand_graph=False,
                use_pagerank=True, pagerank_weight=1.5,
            )


class Phase2AdaptiveTests(unittest.TestCase):
    def _vecs(self, scores):
        return [_vec(s, f"c{i}") for i, s in enumerate(scores)]

    def test_10_decisions(self):
        # High confidence -> vector-focused.
        d = adaptive.expansion_decision(self._vecs([0.97, 0.80]), "What is insulin?")
        self.assertFalse(d["expand"])
        self.assertEqual(d["confidence"], "high")
        self.assertTrue(d["reasons"])
        # Low confidence -> expand.
        d = adaptive.expansion_decision(self._vecs([0.60, 0.58]), "What is insulin?")
        self.assertTrue(d["expand"])
        self.assertEqual(d["confidence"], "low")
        # Multi-concept cue forces expansion despite high scores.
        d = adaptive.expansion_decision(
            self._vecs([0.99, 0.90]),
            "What is the relationship between insulin and glucagon?",
        )
        self.assertTrue(d["expand"])
        self.assertEqual(d["confidence"], "multi-concept")
        # No seeds -> nothing to expand.
        d = adaptive.expansion_decision([], "What is insulin?")
        self.assertFalse(d["expand"])
        # Disabled -> legacy always-expand.
        d = adaptive.expansion_decision(self._vecs([0.99, 0.90]), "What is insulin?", enabled=False)
        self.assertTrue(d["expand"])
        # Deterministic: same inputs, same outputs.
        d2 = adaptive.expansion_decision(self._vecs([0.97, 0.80]), "What is insulin?")
        self.assertEqual(d2, adaptive.expansion_decision(self._vecs([0.97, 0.80]), "What is insulin?"))

    def test_10b_pipeline_integration(self):
        emb = np.ones(DIM, dtype=np.float32)
        exp = [{
            "chunk_id": "e1", "text": "expanded", "embedding": emb,
            "shared_entities": 3, "path_count": 4, "depth": 1,
        }]
        # High confidence -> expansion skipped (no MENTIONS query).
        driver = FakeDriver(vector_records=self._vecs([0.97, 0.80]), expansion_records=exp)
        out = retrieve_mod.retrieve(driver, FakeEmbedder(), "What is insulin?",
                                    top_k=2, expand_graph=True)
        self.assertEqual(len(out), 2)
        self.assertFalse(driver.ran("MENTIONS]->(ent:Entity)<-"))
        # Low confidence -> expansion runs.
        driver2 = FakeDriver(vector_records=self._vecs([0.60, 0.58]), expansion_records=exp)
        out2 = retrieve_mod.retrieve(driver2, FakeEmbedder(), "What is insulin?",
                                     top_k=3, expand_graph=True)
        self.assertTrue(driver2.ran("MENTIONS"))
        self.assertTrue(any(c["chunk_id"] == "e1" for c in out2))
        # expand_graph=False always wins (master switch).
        driver3 = FakeDriver(vector_records=self._vecs([0.60, 0.58]), expansion_records=exp)
        retrieve_mod.retrieve(driver3, FakeEmbedder(),
                              "What is the relationship between a and b?",
                              top_k=2, expand_graph=False)
        self.assertFalse(driver3.ran("MENTIONS"))

    def test_11_vector_only(self):
        driver = FakeDriver(vector_records=[_vec(0.7, "c1"), _vec(0.6, "c2")])
        out = retrieve_mod.retrieve(driver, FakeEmbedder(), "query", top_k=2, expand_graph=False)
        self.assertEqual([c["chunk_id"] for c in out], ["c1", "c2"])
        for c in out:
            self.assertEqual(c["depth"], 0)
            self.assertIn("combined_score", c)

    def test_12_vector_plus_graph(self):
        emb = np.ones(DIM, dtype=np.float32)
        exp = [{
            "chunk_id": "e1", "text": "expanded", "embedding": emb,
            "shared_entities": 3, "path_count": 4, "depth": 1,
        }]
        driver = FakeDriver(vector_records=[_vec(0.60, "c1")], expansion_records=exp)
        out = retrieve_mod.retrieve(driver, FakeEmbedder(), "query", top_k=2,
                                    expand_graph=True, adaptive_enabled=False)
        ids = {c["chunk_id"] for c in out}
        self.assertIn("e1", ids)
        e1 = next(c for c in out if c["chunk_id"] == "e1")
        self.assertEqual(e1["depth"], 1)
        self.assertEqual(e1["path_count"], 4)
        self.assertIn("proximity", e1["score_explanation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
