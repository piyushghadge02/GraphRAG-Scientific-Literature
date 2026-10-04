#!/usr/bin/env python3
"""Phase 3 lightweight tests — evaluation + product layer.

Covers:
  1. Evaluation metric calculations (Recall@K, MRR, ROUGE-L).
  2. Baseline retrieval evaluation path.
  3. Modified retrieval evaluation path.
  4. Comparison output builders.
  5. System preset resolution incl. UI mode mapping.
  6. Retrieval metadata exposed to the UI (strategy label, breakdowns).
  7. run_system dispatch (pagerank vs plain retrieve).

No live Neo4j/LLM/GDS required: fake drivers, fake embedders and a stub
BERT scorer are used throughout.
Run with:  python tests/test_phase3_eval.py
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import retrieval_metrics as rm
from src.evaluation import systems
from src.evaluation import compare

DIM = 384


class FakeEmbedder:
    def __init__(self, dim=DIM):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(self._dim, dtype=np.float32)
        return np.ones((len(texts), self._dim), dtype=np.float32)


class _Single:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row

    def __iter__(self):
        return iter([self._row])


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **kwargs):
        if "db.index.vector" in cypher:
            return list(self.driver.vector_records)
        if "COUNT(DISTINCT e)" in cypher:
            return _Single({"count": 2})
        return list(self.driver.expansion_records)


class FakeDriver:
    def __init__(self, vector_records=None, expansion_records=None):
        self.vector_records = vector_records or []
        self.expansion_records = expansion_records or []

    def session(self):
        return FakeSession(self)


def _vec(score, cid):
    return {"chunk_id": cid, "text": f"text {cid}", "score": score}


# ---------------------------------------------------------------------------
# 1. Metric calculations
# ---------------------------------------------------------------------------

class MetricTests(unittest.TestCase):
    def test_recall_and_mrr(self):
        cmap = {"c1": "a1", "c2": "a2", "c3": "a1"}
        self.assertEqual(rm.recall_at_k(["c9", "c2"], "a2", cmap, k=5), 1.0)
        self.assertEqual(rm.recall_at_k(["c9", "c8"], "a2", cmap, k=5), 0.0)
        self.assertEqual(rm.recall_at_k(["c9", "c2"], "a2", cmap, k=1), 0.0)  # outside k
        self.assertEqual(rm.recall_at_k([], "a2", cmap, k=5), 0.0)
        self.assertAlmostEqual(rm.mrr(["c9", "c2"], "a2", cmap), 0.5)
        self.assertAlmostEqual(rm.mrr(["c1"], "a1", cmap), 1.0)
        self.assertEqual(rm.mrr(["c9"], "a2", cmap), 0.0)
        self.assertEqual(rm.mrr([], "a2", cmap), 0.0)

    def test_evaluate_retrieval_frame(self):
        questions = [
            {"question": "q1", "pubmed_id": "a1"},
            {"question": "q2", "pubmed_id": "a2"},
        ]
        cmap = {"c1": "a1", "c2": "a2"}
        df = rm.evaluate_retrieval(questions, lambda q: ["c1"], cmap)
        self.assertEqual(list(df.columns), ["pubmed_id", "question", "recall_5", "recall_10", "mrr"])
        mean = df[df["pubmed_id"] == "MEAN"].iloc[0]
        self.assertAlmostEqual(mean["recall_5"], 0.5)  # only q1 hits
        self.assertAlmostEqual(mean["mrr"], 0.5)

    def test_rouge_l_identical_is_one(self):
        gen_mod = importlib.import_module("src.evaluation.generation_metrics")

        class _T:
            def __init__(self, vals):
                self._vals = vals

            def tolist(self):
                return self._vals

        class _FakeBert:
            @staticmethod
            def score(*a, **k):
                return _T([1.0]), _T([1.0]), _T([1.0])

        questions = [{"pubmed_id": "a1", "question": "q?"}]
        with patch.object(gen_mod, "bert_score", _FakeBert):
            df = gen_mod.evaluate_generation(["insulin lowers glucose"], ["insulin lowers glucose"], questions)
        mean = df[df["pubmed_id"] == "MEAN"].iloc[0]
        self.assertAlmostEqual(mean["rouge_l_f1"], 1.0, places=4)
        self.assertAlmostEqual(mean["bertscore_f1"], 1.0)
        with self.assertRaises(AssertionError):
            gen_mod.evaluate_generation(["a"], ["a", "b"], questions)


# ---------------------------------------------------------------------------
# 2-5, 7. Systems, comparison output, dispatch
# ---------------------------------------------------------------------------

class SystemTests(unittest.TestCase):
    def test_resolve_valid_and_invalid(self):
        for name in systems.SYSTEM_ORDER:
            preset = systems.resolve_system(name)
            self.assertIn("label", preset)
            self.assertIn("retrieve_kwargs", preset)
        with self.assertRaises(ValueError):
            systems.resolve_system("nope")

    def test_ui_modes_map_to_presets(self):
        for label, name in systems.UI_MODES.items():
            systems.resolve_system(name)  # must not raise
        self.assertIn("baseline", systems.UI_MODES.values())
        self.assertEqual(set(systems.SYSTEM_ORDER), set(systems.SYSTEM_PRESETS))

    def test_baseline_preset_is_legacy(self):
        kwargs = systems.SYSTEM_PRESETS["baseline"]["retrieve_kwargs"]
        self.assertTrue(kwargs["expand_graph"])
        self.assertFalse(kwargs["adaptive_enabled"])
        self.assertEqual((kwargs["alpha"], kwargs["beta"], kwargs["gamma"]), (0.7, 0.3, 0.0))
        self.assertFalse(systems.SYSTEM_PRESETS["baseline"]["use_pagerank"])
        self.assertTrue(systems.SYSTEM_PRESETS["adaptive"]["use_pagerank"])

    def test_run_system_dispatch(self):
        seen = {}

        def fake_retrieve(**kwargs):
            seen["plain"] = kwargs
            return [{"chunk_id": "c1"}]

        def fake_pr(**kwargs):
            seen["pr"] = kwargs
            return [{"chunk_id": "c2"}]

        with patch.object(systems, "retrieve", fake_retrieve), \
             patch.object(systems, "retrieve_with_pagerank", fake_pr):
            out = systems.run_system(None, None, "q", 5, "baseline")
            self.assertEqual(out, [{"chunk_id": "c1"}])
            self.assertIn("plain", seen)
            out = systems.run_system(None, None, "q", 5, "adaptive")
            self.assertEqual(out, [{"chunk_id": "c2"}])
            self.assertIn("pr", seen)
            self.assertEqual(seen["pr"]["top_k"], 5)

    def test_comparison_frames(self):
        means = {
            "baseline": {"recall_5": 0.5, "recall_10": 0.6, "mrr": 0.4},
            "adaptive": {"recall_5": 0.7, "recall_10": 0.7, "mrr": 0.5},
        }
        df = compare.comparison_frame(means)
        self.assertEqual(list(df.columns), ["Metric", "baseline", "adaptive"])
        self.assertEqual(df.iloc[0]["Metric"], "Recall@5")
        legacy = compare.legacy_comparison_frame(means["baseline"], means["adaptive"])
        self.assertEqual(list(legacy.columns), ["Metric", "Baseline", "Full_System", "Delta"])
        self.assertAlmostEqual(legacy.iloc[0]["Delta"], 0.2)
        self.assertAlmostEqual(legacy.iloc[2]["Baseline"], 0.4)

    def test_loaders_read_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            qp = Path(tmp) / "q.jsonl"
            qp.write_text(json.dumps({"question": "q?", "pubmed_id": "a1", "long_answer": "x"}) + "\n",
                          encoding="utf-8")
            self.assertEqual(len(compare.load_eval_questions(str(qp), num_questions=200)), 1)
            cp = Path(tmp) / "c.jsonl"
            cp.write_text(json.dumps({"chunk_id": "c1", "article_id": "a1", "text": "t"}) + "\n",
                          encoding="utf-8")
            self.assertEqual(compare.load_chunk_map(str(cp)), {"c1": "a1"})

    def test_evaluate_system_retrieval_means(self):
        questions = [{"question": "q1", "pubmed_id": "a1"}]
        res = compare.evaluate_system_retrieval(
            questions, lambda q: ["c1"], {"c1": "a1"}, run_name="t"
        )
        self.assertEqual(res["recall_5"], 1.0)
        self.assertEqual(res["mrr"], 1.0)
        self.assertIn("df", res)


# ---------------------------------------------------------------------------
# 6. Retrieval metadata for the UI
# ---------------------------------------------------------------------------

class StrategyMetadataTests(unittest.TestCase):
    def _driver(self, scores, expanded=True):
        emb = np.ones(DIM, dtype=np.float32)
        exp = [{
            "chunk_id": "e1", "text": "expanded", "embedding": emb,
            "shared_entities": 3, "path_count": 4, "depth": 1,
        }] if expanded else []
        vec = [_vec(s, f"c{i}") for i, s in enumerate(scores)]
        return FakeDriver(vector_records=vec, expansion_records=exp)

    def test_vector_only_label(self):
        retrieve_mod = importlib.import_module("src.retrieval.retrieve")
        out = retrieve_mod.retrieve(
            self._driver([0.9]), FakeEmbedder(), "query", top_k=1, expand_graph=False
        )
        self.assertEqual(out[0]["retrieval_strategy"], "vector-only")
        self.assertIn("score_breakdown", out[0])
        self.assertIn("score_explanation", out[0])

    def test_adaptive_labels(self):
        retrieve_mod = importlib.import_module("src.retrieval.retrieve")
        high = retrieve_mod.retrieve(
            self._driver([0.97, 0.80]), FakeEmbedder(), "What is insulin?",
            top_k=2, expand_graph=True,
        )
        self.assertEqual(high[0]["retrieval_strategy"], "adaptive-vector")
        low = retrieve_mod.retrieve(
            self._driver([0.60, 0.58]), FakeEmbedder(), "What is insulin?",
            top_k=3, expand_graph=True,
        )
        strategies = {c["retrieval_strategy"] for c in low}
        self.assertEqual(strategies, {"adaptive-graph"})
        legacy = retrieve_mod.retrieve(
            self._driver([0.60, 0.58]), FakeEmbedder(), "What is insulin?",
            top_k=3, expand_graph=True, adaptive_enabled=False,
        )
        self.assertEqual(legacy[0]["retrieval_strategy"], "graph")


if __name__ == "__main__":
    unittest.main(verbosity=2)
