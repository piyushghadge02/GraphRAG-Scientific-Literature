#!/usr/bin/env python3
"""Submission-compliance tests (PDF deliverable alignment).

- compose_sample(): matched papers prioritized, deterministic, capped.
- merge_article_chunks(): one chunk per article, first ID kept.
- clear_graph(): issues a single DETACH DELETE (fake driver).
- expand_via_entities(): passes deg_cap (0 = guard disabled by default).

No network/Neo4j/model downloads.
Run with:  python tests/test_compliance.py
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib

load_data = importlib.import_module("src.chunking.load_data")
chunker = importlib.import_module("src.chunking.chunker")
graph_expand = importlib.import_module("src.retrieval.graph_expand")


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, **kwargs):
        self.driver.seen.append((cypher, kwargs))
        return []

    def execute_write(self, fn, **kwargs):
        driver = self.driver

        class _Tx:
            @staticmethod
            def run(cypher, **kw):
                driver.seen.append((cypher, kw))

                class _R:
                    @staticmethod
                    def single():
                        return {"d": 0}

                return _R()

        return fn(_Tx())


class FakeDriver:
    def __init__(self):
        self.seen = []

    def session(self):
        return FakeSession(self)


def _load_schema():
    import importlib.util
    import types
    if "neo4j" not in sys.modules:
        stub = types.ModuleType("neo4j")

        class _GraphDatabase:
            @staticmethod
            def driver(*a, **k):
                raise RuntimeError("no live neo4j")

        stub.__dict__["GraphDatabase"] = _GraphDatabase
        sys.modules["neo4j"] = stub
    path = PROJECT_ROOT / "src" / "graph" / "schema.py"
    spec = importlib.util.spec_from_file_location("compliance_schema_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compliance_schema_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class ComposeSampleTests(unittest.TestCase):
    def test_matched_prioritized_and_deterministic(self):
        a = load_data.compose_sample([90, 10, 50], 100, 10, 42)
        b = load_data.compose_sample([90, 10, 50], 100, 10, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        for m in (10, 50, 90):
            self.assertIn(m, a)
        self.assertEqual(a, sorted(a))

    def test_caps_when_matched_exceed(self):
        out = load_data.compose_sample(list(range(30)), 100, 10, 42)
        self.assertEqual(len(out), 10)


class MergeChunksTests(unittest.TestCase):
    def test_merge_one_per_article(self):
        chunks = [
            {"article_id": "a1", "chunk_id": "a1_chunk_0", "text": "t0", "strategy": "semantic_cluster"},
            {"article_id": "a1", "chunk_id": "a1_chunk_1", "text": "t1", "strategy": "semantic_cluster"},
        ]
        out = chunker.merge_article_chunks("a1", chunks)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["chunk_id"], "a1_chunk_0")
        self.assertEqual(out[0]["text"], "t0 t1")
        self.assertEqual(out[0]["strategy"], "semantic_cluster")
        self.assertEqual(chunker.merge_article_chunks("a1", []), [])


class ClearGraphTests(unittest.TestCase):
    def test_clear_issues_batched_detach_delete(self):
        schema = _load_schema()
        driver = FakeDriver()
        deleted = schema.clear_graph(driver)
        self.assertEqual(deleted, 0)
        self.assertEqual(len(driver.seen), 1)
        self.assertIn("DETACH DELETE", driver.seen[0][0])
        self.assertIn("LIMIT", driver.seen[0][0])



class ExpansionGuardTests(unittest.TestCase):
    def test_deg_cap_defaults_to_zero(self):
        driver = FakeDriver()
        graph_expand.expand_via_entities(driver, ["s1"])
        cypher, kwargs = driver.seen[0]
        self.assertEqual(kwargs.get("deg_cap"), 0)
        self.assertIn("deg_cap", cypher)
        # Neo4j 5.26 removed size(pattern): degree guard must use COUNT {}.
        self.assertIn("COUNT {", cypher)
        self.assertNotIn("size((", cypher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
