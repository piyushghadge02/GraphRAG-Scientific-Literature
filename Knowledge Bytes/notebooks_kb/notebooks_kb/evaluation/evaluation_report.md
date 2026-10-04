# Knowledge Bytes — `notebooks/evaluation_report.ipynb`

---

### Byte 1: The Notebook Is an Evaluation Report
**Builds on:** None — starting point

**In plain terms:**
This notebook compares several GraphRAG retrieval configurations on the same PubMedQA questions. It is primarily a reporting and interpretation layer: the actual evaluation artifacts are produced by `src.evaluation.compare`.

**The code:**
```markdown
# Phase 3 Evaluation Report — Baseline vs Modified GraphRAG

Compares the PDF baseline against our modified GraphRAG
on the SAME PubMedQA questions.
```

Keeping the notebook separate from the evaluation implementation makes the report reproducible from saved CSV artifacts.

---

### Byte 2: Five Retrieval Systems Form the Experimental Ladder
**Builds on:** Byte 1

**In plain terms:**
The notebook defines an ablation sequence: vector-only, baseline graph expansion, fused scoring, PageRank, and adaptive retrieval. Each later system adds a specific retrieval component to the previous one.

**The code:**
```markdown
A. Vector-only  → vector search, no graph
B. Baseline     → expansion + legacy fusion
C. Fused        → 3-term graph-aware fusion
D. PageRank     → fused + subgraph PageRank
E. Adaptive     → fused + PageRank + adaptive retrieval
```

This ordering is important because adjacent systems are intended to isolate one architectural change.

---

### Byte 3: Retrieval and Generation Use Different Metrics
**Builds on:** Byte 1

**In plain terms:**
Retrieval is evaluated with article-level Recall@5, Recall@10, and MRR. Generated answers are evaluated separately with ROUGE-L F1 and BERTScore F1 against PubMedQA long answers.

**The code:**
```markdown
Metrics:
Recall@5 / Recall@10 / MRR
ROUGE-L F1 / BERTScore F1
```

This separates the question “Did retrieval find the relevant article?” from “How well did generation use the retrieved evidence?”

---

### Byte 4: The Notebook Reads the Project's System Definitions
**Builds on:** Bytes 1–2

**In plain terms:**
The setup imports `describe_systems()` from `src.evaluation.systems`. This gives the notebook the configured system names and descriptions instead of duplicating those definitions.

**The code:**
```python
from src.evaluation.systems import describe_systems

for s in describe_systems():
    print(f"{s['label']}: {s['description']}")
```

The notebook therefore stays aligned with the evaluation module's system configuration.

---

### Byte 5: All Processed Artifacts Live Under One Path
**Builds on:** Byte 4

**In plain terms:**
The notebook treats `data/processed` as the common location for evaluation inputs and outputs. This keeps the report independent of hard-coded individual file locations.

**The code:**
```python
PROCESSED = Path("data/processed")
```

Every later cell builds its artifact path from this base directory.

---

### Byte 6: The Notebook Checks Whether Evaluation Questions Exist
**Builds on:** Byte 5

**In plain terms:**
Before displaying results, the notebook checks for `eval_questions.jsonl`. If it is missing, it tells the user to generate it rather than inventing a dataset.

**The code:**
```python
q_path = PROCESSED / "eval_questions.jsonl"

if q_path.exists():
    n_q = sum(
        1 for _ in open(q_path, encoding="utf-8")
    )
else:
    print(
        "eval_questions.jsonl NOT found"
    )
```

This makes the report fail transparently when its input artifacts are unavailable.

---

### Byte 7: Required Chunk Files Are Checked Too
**Builds on:** Byte 6

**In plain terms:**
The notebook also checks whether `semantic_cluster.jsonl` and `fixed_token.jsonl` exist. These files represent the chunking artifacts needed by the evaluation workflow.

**The code:**
```python
for name in (
    "semantic_cluster.jsonl",
    "fixed_token.jsonl"
):
    p = PROCESSED / name
    print(
        f"{name}: "
        f"{'present' if p.exists() else 'MISSING'}"
    )
```

The notebook reports availability; it does not generate these files.

---

### Byte 8: Retrieval Results Come from `eval_systems.csv`
**Builds on:** Bytes 5–7

**In plain terms:**
The notebook reads the system-level retrieval results from a CSV produced by the evaluation pipeline. If the file is absent, it tells the user which command should be run.

**The code:**
```python
systems_csv = PROCESSED / "eval_systems.csv"

if systems_csv.exists():
    systems_df = pd.read_csv(systems_csv)
    display(systems_df)
else:
    print(
        "eval_systems.csv NOT found — "
        "run: python -m src.evaluation.compare"
    )
```

This makes the notebook a consumer of evaluation results rather than the component that calculates them.

---

### Byte 9: Baseline-vs-Modified Comparison Is Loaded Separately
**Builds on:** Byte 8

**In plain terms:**
`eval_comparison.csv` contains the more direct baseline-versus-modified comparison. Keeping this table separate from the all-systems table makes the report useful for both detailed ablation and summary comparison.

**The code:**
```python
comp_csv = PROCESSED / "eval_comparison.csv"

if comp_csv.exists():
    display(pd.read_csv(comp_csv))
else:
    print(
        "eval_comparison.csv NOT found — "
        "run: python -m src.evaluation.compare"
    )
```

No metric is recomputed inside the notebook.

---

### Byte 10: The Retrieval Chart Is a Saved Artifact
**Builds on:** Bytes 8–9

**In plain terms:**
The notebook displays `retrieval_comparison.png` if it exists. The chart is generated elsewhere and is treated here as a reporting artifact.

**The code:**
```python
chart = PROCESSED / "retrieval_comparison.png"

if chart.exists():
    display(Image(filename=str(chart)))
```

This keeps visualization generation outside the notebook's reporting cells.

---

### Byte 11: Generation Results Are Optional
**Builds on:** Byte 3

**In plain terms:**
Generation evaluation is loaded from `eval_generation.csv`. The notebook explicitly warns that generation requires LLM credentials when the file is missing.

**The code:**
```python
gen_csv = PROCESSED / "eval_generation.csv"

if gen_csv.exists():
    display(pd.read_csv(gen_csv))
else:
    print(
        "eval_generation.csv NOT found — "
        "generation needs LLM credentials"
    )
```

This reflects a real dependency difference: retrieval evaluation can use saved results, while generating new answers requires an LLM backend.

---

### Byte 12: Adjacent Systems Enable Controlled Ablations
**Builds on:** Bytes 2 and 8

**In plain terms:**
The notebook tells the reader how to interpret each step in the system ladder. Comparing adjacent systems is intended to isolate graph expansion, fusion, PageRank, and adaptive retrieval respectively.

**The code:**
```markdown
vector_only → baseline  : graph expansion
baseline → fused        : 3-term fusion
fused → pagerank        : PageRank
pagerank → adaptive     : adaptive decision
```

This is an experimental design principle: change one major component at a time where possible.

---

### Byte 13: Small Deltas Need Cautious Interpretation
**Builds on:** Byte 12

**In plain terms:**
The notebook explicitly warns that small metric changes should not automatically be treated as meaningful, especially for a relatively small evaluation set. The measured delta describes behavior on these questions; it does not by itself establish a general result.

**The code:**
```markdown
with ~200 questions, treat small deltas
as inconclusive.
```

The notebook therefore distinguishes measured comparison from broader claims.

---

### Byte 14: Retrieval Recall Is Article-Level
**Builds on:** Byte 3

**In plain terms:**
The evaluation treats retrieval as successful when the gold article is found, not necessarily when the best possible chunk is retrieved. This is an important limitation when interpreting Recall@K.

**The code:**
```markdown
Article-level binary recall rewards
finding the gold article, not the best chunk.
```

A high article-level recall therefore does not prove that the retrieved chunk is the most useful evidence passage.

---

### Byte 15: Generation Metrics Depend on the LLM Backend Too
**Builds on:** Byte 11

**In plain terms:**
The notebook notes that generation quality depends on both retrieval and the LLM backend/credentials. Therefore, a generation metric cannot be interpreted as a pure retrieval measurement.

**The code:**
```markdown
Generation quality depends on the LLM
backend/credentials as well as retrieval.
```

This is why retrieval metrics and generation metrics remain separate in the report.

---

### Byte 16: PageRank Has an External Runtime Dependency
**Builds on:** Byte 2

**In plain terms:**
The PageRank configuration depends on the Neo4j GDS plugin. If GDS is unavailable, the pipeline is designed to degrade gracefully, and the PageRank system can effectively behave like the fused system.

**The code:**
```markdown
PageRank results require the GDS plugin;
without it the pipeline degrades gracefully
and D ≈ C.
```

The notebook records this as a limitation rather than hiding the dependency.

---

### Byte 17: The Notebook Does Not Make a Superiority Claim
**Builds on:** Bytes 12–16

**In plain terms:**
The final interpretation section is deliberately framed as questions to answer from the measured tables. The limitations explicitly state that the notebook should not turn observed deltas into an unsupported claim that one system is generally superior.

**The code:**
```markdown
Interpretation (fill in ONLY from
measured tables above)

No superiority claim is made here
beyond the measured deltas.
```

This keeps the notebook grounded in the experiment's actual evidence.

---

## PUTTING IT TOGETHER

`evaluation_report.ipynb` is the presentation and interpretation layer for the Phase 3 evaluation. It defines the experimental ladder, checks that the required artifacts exist, loads retrieval and generation metrics, and displays the saved retrieval chart. The actual measurements are produced by `src.evaluation.compare`; the notebook deliberately avoids fabricating or recomputing missing results. Its ablation guide explains how adjacent systems isolate graph expansion, fusion, PageRank, and adaptive retrieval. Finally, its limitations prevent article-level retrieval metrics and LLM-dependent generation metrics from being interpreted more broadly than the measured experiment supports.
