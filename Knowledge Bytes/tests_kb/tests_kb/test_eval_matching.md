---
### Byte 1: Evaluation matching is a data-alignment test
**Builds on:** None — starting point

**In plain terms:**
This module protects the step that connects PubMedQA evaluation questions to the project's corpus articles. It checks that the evaluation dataset uses the corpus's own article IDs and does not invent matches.

**The code:**
```python
from src.evaluation import match_corpus as mc
from src.evaluation import retrieval_metrics as rm
```

The test combines matching logic with retrieval metrics because the resulting gold IDs must be usable by the evaluation pipeline.

---
### Byte 2: Synthetic abstracts keep matching tests self-contained
**Builds on:** Byte 1

**In plain terms:**
The tests construct small fake corpus articles rather than loading the real PubMed collection. Each article has an `article_id`, title, and abstract text—the fields needed for matching.

**The code:**
```python
def _abstract(aid, text):
    return {"article_id": aid, "title": "", "abstract_text": text}
```

This makes each test case explicit and avoids network, filesystem, or dataset dependencies.

---
### Byte 3: PubMedQA examples are represented in their real shape
**Builds on:** Byte 2

**In plain terms:**
A second fixture creates PubMedQA-like records containing a publication ID, question, context passages, long answer, and decision. This lets the tests exercise the matcher against realistic input structure.

**The code:**
```python
def _example(pubid, passages, question="Q?", long_answer="A.", decision="yes"):
    return {
        "pubid": pubid,
        "question": question,
        "context": {"contexts": passages},
        "long_answer": long_answer,
        "final_decision": decision,
    }
```

The helper keeps test cases concise while preserving the fields the matching code expects.

---
### Byte 4: Text normalization makes equivalent passages comparable
**Builds on:** Bytes 2–3

**In plain terms:**
Corpus text and PubMedQA text can differ in whitespace or tokenizer-added spacing even when they represent the same passage. Normalization removes those superficial differences before matching.

**The code:**
```python
self.assertEqual(
    mc.normalize_text("  Insulin
  Resistance "),
    "insulin resistance",
)
```

The test also checks punctuation-spacing convergence, showing why normalization is necessary before exact comparison.

---
### Byte 5: Context extraction accepts multiple input shapes
**Builds on:** Byte 3

**In plain terms:**
PubMedQA context may arrive as a dictionary containing `contexts`, a list directly, or a single string. The extraction helper converts these forms into a consistent list of passages.

**The code:**
```python
self.assertEqual(
    mc.extract_context_passages({"context": {"contexts": ["a", "b"]}}),
    ["a", "b"],
)
self.assertEqual(mc.extract_context_passages({"context": ["a"]}), ["a"])
self.assertEqual(mc.extract_context_passages({"context": "solo"}), ["solo"])
```

The tests protect input-shape tolerance at the boundary of the matching pipeline.

---
### Byte 6: Unmatched questions are excluded
**Builds on:** Bytes 4–5

**In plain terms:**
A question without a reliable corpus match should not be assigned a fabricated gold article. The tests verify that unmatched PubMedQA records are left out of the evaluation set.

**The code:**
```python
# Unmatched PubMedQA questions are excluded.
```

This keeps retrieval metrics honest: a missing match is treated as missing evaluation data, not as an arbitrary positive target.

---
### Byte 7: Matched questions store the corpus article ID
**Builds on:** Byte 6

**In plain terms:**
When a match is found, the evaluation record must use the article ID from the actual corpus. This ensures later retrieval metrics compare retrieved IDs against the same identifier namespace.

**The code:**
```python
# Matched questions store the correct corpus article_id.
```

The distinction matters because the PubMedQA publication identifier and the project's corpus identifier may not be interchangeable without an explicit match step.

---
### Byte 8: Sampling is deterministic
**Builds on:** Byte 7

**In plain terms:**
The test suite requires sampling to be reproducible with seed 42. The same matched population should therefore produce the same evaluation sample across runs.

**The code:**
```python
# Sampling is deterministic (seed=42).
```

Deterministic sampling makes metric changes attributable to code changes rather than to a different random evaluation subset.

---
### Byte 9: Small matched datasets are not padded with fake records
**Builds on:** Byte 8

**In plain terms:**
The matcher may find fewer than the target number of evaluation questions. The tests require all real matches to be retained without duplication or fabricated entries.

**The code:**
```python
# Fewer-than-200 matches: all saved, none duplicated/fabricated.
```

This protects the integrity of small evaluation runs instead of forcing every run to have an artificial fixed size.

---
### Byte 10: The resulting IDs must work with retrieval metrics
**Builds on:** Bytes 7–9

**In plain terms:**
The final check connects matching to evaluation: the article IDs produced by the matcher must be accepted by the retrieval-metric functions. A correct match that cannot be evaluated would still break the evaluation pipeline.

**The code:**
```python
# Retrieval metrics work with the resulting corpus IDs.
```

This is the module's end-to-end invariant: matching must produce identifiers that the downstream evaluator understands.

PUTTING IT TOGETHER

The matching tests protect the boundary between external evaluation data and the project's corpus. They normalize and extract text safely, reject unmatched questions, preserve the correct corpus article IDs, and make sampling reproducible. The final metric check ensures those matched IDs are actually usable for retrieval evaluation. Together, these tests prevent data-alignment problems from being mistaken for retrieval performance differences.
