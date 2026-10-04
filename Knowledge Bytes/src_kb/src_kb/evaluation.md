# Knowledge Bytes — `src/evaluation/`

---

### Byte 1: Evaluation First Matches Questions to the Real Corpus
**Builds on:** `chunking.md` Byte 2

**In plain terms:**
PubMedQA IDs cannot automatically be assumed to match the synthetic article IDs created by the corpus loader. `match_corpus.py` verifies that a sufficiently long normalized PubMedQA passage occurs inside a corpus abstract.

**The code:**
```python
norm_passage = normalize_text(passage)

if len(norm_passage) < min_passage_chars:
    continue

if norm_passage in norm_abstract:
    return article_id, preview
```

This creates a trustworthy mapping from evaluation question to the actual corpus article.

---

### Byte 2: Evaluation Sampling Is Deterministic
**Builds on:** Byte 1

**In plain terms:**
After matching valid questions, the evaluator shuffles them using a fixed seed and takes the requested number. This makes repeated experiments comparable.

**The code:**
```python
items = sorted(matched.items())
rng = random.Random(seed)
rng.shuffle(items)

return items[:n]
```

The system keeps the available valid matches rather than inventing missing ones.

---

### Byte 3: Recall@K Measures Whether the Gold Article Appeared
**Builds on:** Bytes 1–2

**In plain terms:**
The retrieval metric checks whether any of the top-k retrieved chunks belongs to the gold article. The project computes this for both k=5 and k=10.

**The code:**
```python
for chunk_id in retrieved_chunk_ids[:k]:
    if chunk_to_article_map.get(chunk_id) == gold_article_id:
        return 1.0

return 0.0
```

This is article-level recall, not a count of how many relevant chunks were found.

---

### Byte 4: MRR Rewards Higher-Ranked Relevant Results
**Builds on:** Byte 3

**In plain terms:**
Mean Reciprocal Rank looks for the first relevant chunk and gives it `1 / rank`. A relevant result at rank 1 therefore contributes more than one at rank 5.

**The code:**
```python
for rank, chunk_id in enumerate(
    retrieved_chunk_ids, start=1
):
    if chunk_to_article_map.get(chunk_id) == gold_article_id:
        return 1.0 / rank
return 0.0
```

The final MRR is the average reciprocal rank over evaluation questions.

---

### Byte 5: Generated Answers Need Separate Metrics
**Builds on:** Byte 3

**In plain terms:**
Correct retrieval does not guarantee a good generated answer. `generation_metrics.py` therefore evaluates generated answers against PubMedQA long answers using ROUGE-L and BERTScore.

**The code:**
```python
score = scorer.score(gold, gen)
rouge_scores.append(
    score["rougeL"].fmeasure
)

P, R, F1 = bert_score.score(
    generated_answers,
    gold_answers,
    model_type="distilbert-base-uncased",
)
```

Retrieval quality and answer quality are intentionally measured separately.

---

### Byte 6: Named Systems Make Ablations Reproducible
**Builds on:** Bytes 3–5

**In plain terms:**
`systems.py` defines named retrieval configurations such as vector-only, baseline, fused, PageRank, and adaptive variants. This lets evaluation compare design choices without rewriting the pipeline.

**The code:**
```python
SYSTEM_PRESETS = {
    "vector_only": {...},
    "baseline": {...},
    "fused": {...},
    "pagerank": {...},
    "adaptive": {...},
}
```

Each preset changes retrieval arguments while keeping the evaluation harness consistent.

---

### Byte 7: The Legacy Baseline Is Separate
**Builds on:** Byte 6

**In plain terms:**
`run_eval.py` also contains a simpler baseline: fixed-token chunks embedded in memory and searched with NumPy cosine similarity. It is not the same thing as the named GraphRAG baseline.

**The code:**
```python
similarities = np.dot(
    embeddings_normalized,
    query_norm
)
```

When reading results, identify which baseline implementation produced the numbers.

---

### Byte 8: Full Retrieval Evaluation Calls the Real Graph Pipeline
**Builds on:** `retrieval.md`

**In plain terms:**
The full evaluation path delegates to `retrieve()` with graph expansion enabled and converts its results into chunk IDs for the metrics.

**The code:**
```python
results = retrieve(
    driver=driver,
    embedding_model=embedding_model,
    query=query,
    top_k=top_k,
    expand_graph=True,
)
return [r["chunk_id"] for r in results]
```

This means the evaluation measures the actual retrieval implementation rather than a duplicate approximation.

---

### Byte 9: Manual Expansion Comparison Tests the Mechanism
**Builds on:** Byte 8

**In plain terms:**
`compare.py` compares retrieval with graph expansion disabled versus enabled for the same questions. It can also compare the generated answers from the two contexts.

**The code:**
```python
retrieve(..., expand_graph=False)
retrieve(..., expand_graph=True)
```

This is useful for understanding what graph expansion actually changes, not only its aggregate metric effect.

---

### Byte 10: `run_full_eval()` Connects the Evaluation Pipeline
**Builds on:** Bytes 2–9

**In plain terms:**
The main runner loads verified questions, initializes models and Neo4j, evaluates retrieval systems, evaluates generated answers, and writes evaluation artifacts.

**The code:**
```python
baseline_retrieval_df = evaluate_retrieval(...)
full_retrieval_df = evaluate_retrieval(...)

generation_df = evaluate_generation(
    gen_answers,
    gold_answers,
    gen_questions,
)
```

Generation evaluation is normally kept smaller because LLM calls cost more than retrieval-only evaluation.

---

### PUTTING IT TOGETHER

The evaluation package first makes sure the questions actually correspond to the corpus. It then measures retrieval using Recall@5/10 and MRR, while generated answers are measured separately with ROUGE-L and BERTScore. Named system presets and the expansion comparison make it possible to test individual GraphRAG design choices instead of only reporting one final number.
