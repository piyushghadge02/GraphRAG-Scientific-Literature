#!/usr/bin/env python3
"""Focused tests for the evaluation-data matching fix.

Verifies:
  1. Corpus article IDs become the evaluation gold IDs.
  2. Unmatched PubMedQA questions are excluded.
  3. Matched questions store the correct corpus article_id.
  4. Sampling is deterministic (seed=42).
  5. Fewer-than-200 matches: all saved, none duplicated/fabricated.
  6. Retrieval metrics work with the resulting corpus IDs.

Pure unit tests: synthetic abstracts/examples only, no HF/network/Neo4j.
Run with:  python tests/test_eval_matching.py
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import match_corpus as mc
from src.evaluation import retrieval_metrics as rm


LONG_A = (
    "Programmed cell death is the regulated death of cells within an organism "
    "and the lace plant provides a model system for studying developmental cell "
    "death processes in detail across leaf formation stages."
)
LONG_B = (
    "Mitochondrial dysfunction contributes to neurodegeneration through impaired "
    "oxidative phosphorylation and increased reactive oxygen species production "
    "in affected neuronal populations over the course of disease progression."
)


def _abstract(aid, text):
    return {"article_id": aid, "title": "", "abstract_text": text}


def _example(pubid, passages, question="Q?", long_answer="A.", decision="yes"):
    return {
        "pubid": pubid,
        "question": question,
        "context": {"contexts": passages},
        "long_answer": long_answer,
        "final_decision": decision,
    }


class MatchingTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(mc.normalize_text("  Insulin\n  Resistance "), "insulin resistance")
        self.assertEqual(mc.normalize_text(""), "")
        # Tokenizer-spaced corpus text and raw PubMedQA text converge.
        self.assertEqual(
            mc.normalize_text("cells . The  (PCD)  study, 2011 ;"),
            mc.normalize_text("cells. The (PCD) study, 2011;"),
        )

    def test_passage_extraction_shapes(self):
        self.assertEqual(mc.extract_context_passages({"context": {"contexts": ["a", "b"]}}), ["a", "b"])
        self.assertEqual(mc.extract_context_passages({"context": ["a"]}), ["a"])
        self.assertEqual(mc.extract_context_passages({"context": "solo"}), ["solo"])
        self.assertEqual(mc.extract_context_passages({}), [])

    def test_verified_match_maps_to_corpus_id(self):
        abstracts = [_abstract("pubmed_7_abc123", LONG_A), _abstract("pubmed_9_def456", LONG_B)]
        index = mc.build_corpus_index(abstracts)
        hit = mc.match_question([LONG_A], index)
        self.assertIsNotNone(hit)
        article_id, preview = hit
        self.assertEqual(article_id, "pubmed_7_abc123")  # corpus ID, not pqa_*
        self.assertTrue(preview)

    def test_case_and_whitespace_insensitive(self):
        abstracts = [_abstract("pubmed_1_x", LONG_A)]
        index = mc.build_corpus_index(abstracts)
        noisy = "  programmed   CELL\ndeath is the REGULATED death of cells within an organism " \
                "and the lace plant provides a model system for studying developmental cell " \
                "death processes in detail across leaf formation stages.  "
        self.assertIsNotNone(mc.match_question([noisy], index))

    def test_unmatched_excluded_and_short_passage_ignored(self):
        abstracts = [_abstract("pubmed_1_x", LONG_A)]
        index = mc.build_corpus_index(abstracts)
        self.assertIsNone(mc.match_question(["completely unrelated short text"], index))
        self.assertIsNone(mc.match_question(["tiny"], index))  # below MIN_PASSAGE_CHARS
        self.assertIsNone(mc.match_question([], index))

    def test_match_all_stats_and_correct_ids(self):
        abstracts = [_abstract("pubmed_7_abc123", LONG_A)]
        examples = [
            _example("111", [LONG_A]),
            _example("222", ["nothing in common " * 10]),
        ]
        matched, stats = mc.match_all(examples, abstracts, progress_every=0)
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["unmatched"], 1)
        self.assertEqual(matched["111"]["article_id"], "pubmed_7_abc123")

        rec = mc.make_eval_record(examples[0], "111", matched["111"]["article_id"])
        self.assertEqual(rec["pubmed_id"], "pubmed_7_abc123")  # gold = corpus ID
        self.assertEqual(rec["pqa_pubid"], "111")  # traceability preserved
        self.assertEqual(rec["pqa_article_id"], "pqa_111")
        self.assertIn("question", rec)
        self.assertIn("long_answer", rec)

    def test_deterministic_sampling(self):
        matched = {f"{i:04d}": {"article_id": f"pubmed_{i}"} for i in range(50)}
        first = mc.select_matched(matched, n=200, seed=42)
        second = mc.select_matched(matched, n=200, seed=42)
        self.assertEqual(first, second)  # deterministic
        self.assertEqual(len(first), 50)  # fewer than n: all saved, none fabricated
        self.assertEqual(len({p for p, _ in first}), 50)  # no duplicates

    def test_sampling_caps_at_n(self):
        matched = {f"{i:04d}": {"article_id": f"pubmed_{i}"} for i in range(300)}
        selected = mc.select_matched(matched, n=200, seed=42)
        self.assertEqual(len(selected), 200)
        self.assertEqual(len({p for p, _ in selected}), 200)


class MetricsWithCorpusIdsTests(unittest.TestCase):
    def test_metrics_use_corpus_gold_ids(self):
        chunk_map = {"pubmed_7_abc123_chunk_0": "pubmed_7_abc123",
                     "pubmed_9_def456_chunk_0": "pubmed_9_def456"}
        gold = "pubmed_7_abc123"
        self.assertEqual(rm.recall_at_k(["pubmed_7_abc123_chunk_0"], gold, chunk_map, k=5), 1.0)
        self.assertEqual(rm.recall_at_k(["pubmed_9_def456_chunk_0"], gold, chunk_map, k=5), 0.0)
        self.assertAlmostEqual(rm.mrr(["pubmed_9_def456_chunk_0", "pubmed_7_abc123_chunk_0"], gold, chunk_map), 0.5)
        # Old-style pqa_* gold IDs can no longer match corpus chunks (the fixed bug).
        self.assertEqual(rm.recall_at_k(["pubmed_7_abc123_chunk_0"], "pqa_111", chunk_map, k=5), 0.0)


class EmptyGuardTests(unittest.TestCase):
    def test_empty_questions_raise(self):
        import tempfile
        from src.evaluation import compare
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                compare.load_eval_questions(str(empty))


class CombinedSetTests(unittest.TestCase):
    def test_question_matches_own_context_record(self):
        documents = [
            _abstract("pubmed_7_abc123", LONG_A),
            {"article_id": "pqa_111", "title": "", "abstract_text": LONG_B},
        ]
        examples = [_example("111", [LONG_B])]
        matched, stats = mc.match_all(examples, documents, progress_every=0)
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(matched["111"]["article_id"], "pqa_111")
        rec = mc.make_eval_record(examples[0], "111", "pqa_111")
        self.assertEqual(rec["pubmed_id"], "pqa_111")

    def test_abstracts_only_still_excludes(self):
        documents = [_abstract("pubmed_7_abc123", LONG_A)]
        examples = [_example("111", [LONG_B])]
        matched, stats = mc.match_all(examples, documents, progress_every=0)
        self.assertEqual(stats["matched"], 0)
        self.assertEqual(stats["unmatched"], 1)


class ChunkPresenceTests(unittest.TestCase):
    def test_present_articles_verify(self):
        examples = [_example("111", ["anything"]), _example("222", ["anything"])]
        matched, stats = mc.match_by_article_presence(examples, {"pqa_111", "pubmed_1_x"})
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["unmatched"], 1)
        self.assertEqual(matched["111"]["article_id"], "pqa_111")

    def test_selection_and_record(self):
        matched = {f"{i:04d}": {"article_id": f"pqa_{i:04d}"} for i in range(250)}
        selected = mc.select_matched(matched, n=200, seed=42)
        self.assertEqual(len(selected), 200)
        self.assertEqual(selected, mc.select_matched(matched, n=200, seed=42))
        rec = mc.make_eval_record(_example("0007", ["x"]), "0007", "pqa_0007")
        self.assertEqual(rec["pubmed_id"], "pqa_0007")
        self.assertEqual(rec["pqa_pubid"], "0007")


if __name__ == "__main__":
    unittest.main(verbosity=2)
