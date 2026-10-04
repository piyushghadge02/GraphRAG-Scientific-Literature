# Knowledge Bytes — `data/processed/`

The `processed` folder contains generated chunks and evaluation artifacts produced after the raw corpus has been prepared.

---

### Byte 1: Semantic Chunks Are Stored as JSONL
**Builds on:** `raw.md` Byte 1

**In plain terms:**
`semantic_cluster.jsonl` contains the processed chunk representation used by the GraphRAG pipeline. It has 5,000 records in this project snapshot.

**The code:**
```json
{
  "article_id": "pubmed_0_f2ecaec77235",
  "chunk_id": "pubmed_0_f2ecaec77235_chunk_0",
  "text": "background : ...",
  "strategy": "semantic_cluster"
}
```

Each record carries both source identity and chunk identity, plus the strategy used to produce it.

---

### Byte 2: This Snapshot Has One Stored Semantic Chunk per Article
**Builds on:** Byte 1

**In plain terms:**
The file contains 5,000 chunk records for the 5,000-article sample, and the examples show one `chunk_0` record per article. This is the processed representation used for the current project snapshot.

**The code:**
```text
pubmed_0_..._chunk_0
pubmed_1_..._chunk_0
pubmed_2_..._chunk_0
...
```

This is important when interpreting chunk counts: the raw corpus and this processed snapshot currently have matching record counts.

---

### Byte 3: The `strategy` Field Records How a Chunk Was Produced
**Builds on:** Byte 1

**In plain terms:**
The `strategy` property preserves provenance for the chunking method. Here it is `semantic_cluster`.

**The code:**
```json
{
  "strategy": "semantic_cluster"
}
```

This lets evaluation or debugging distinguish semantic output from other possible chunking strategies.

---

### Byte 4: Evaluation Questions Are a Small Prepared Test Set
**Builds on:** `raw.md` Bytes 4–5

**In plain terms:**
`eval_questions.jsonl` contains 6 prepared PubMedQA-style questions in this snapshot. Each record connects the question to the corpus article, stores the long reference answer, and records the yes/no decision.

**The code:**
```json
{
  "question": "...",
  "pubmed_id": "...",
  "pqa_article_id": "pqa_...",
  "long_answer": "...",
  "final_decision": "yes"
}
```

The file is therefore a ready-to-run evaluation subset rather than the complete PubMedQA dataset.

---

### Byte 5: Evaluation Questions Link PubMedQA to Corpus IDs
**Builds on:** Byte 4

**In plain terms:**
Each evaluation record contains both PubMedQA identity and the project's generated `pubmed_...` corpus ID. This is what lets retrieval results be evaluated against the correct source article.

**The code:**
```json
{
  "pubmed_id": "23794696",
  "pubmed_id": "...",
  "pubmed_id": "..."
}
```

More importantly, the actual field used for the corpus mapping is `pubmed_id` in the processed evaluation record, alongside the project-specific `pubmed_...` value.

---

### Byte 6: Retrieval Comparison Stores Aggregate Metrics
**Builds on:** Bytes 4–5

**In plain terms:**
`eval_comparison.csv` compares two retrieval setups using Recall@5, Recall@10, and MRR. The file stores the baseline value, full-system value, and their numerical delta.

**The code:**
```csv
Metric,Baseline,Full_System,Delta
Recall@5,1.0,0.8333333333333334,-0.16666666666666663
Recall@10,1.0,0.8333333333333334,-0.16666666666666663
MRR,0.8333333333333334,0.75,-0.08333333333333337
```

These are results for the prepared evaluation snapshot, not a general claim about GraphRAG performance.

---

### Byte 7: System-Level Evaluation Compares Multiple Retrieval Modes
**Builds on:** Byte 6

**In plain terms:**
`eval_systems.csv` contains metrics for five named retrieval configurations: `vector_only`, `baseline`, `fused`, `pagerank`, and `adaptive`.

**The code:**
```csv
Metric,vector_only,baseline,fused,pagerank,adaptive
Recall@5,1.0,1.0,1.0,0.8333333333333334,0.8333333333333334
Recall@10,1.0,1.0,1.0,0.8333333333333334,0.8333333333333334
MRR,1.0,0.8333333333333334,0.8333333333333334,0.75,0.75
```

The purpose is comparative experimentation: the same metrics are recorded across different retrieval configurations.

---

### Byte 8: Recall@K and MRR Have Different Meanings
**Builds on:** Bytes 6–7

**In plain terms:**
Recall@5/10 asks whether a relevant article appears within the first 5 or 10 retrieved results. MRR is sensitive to where the first relevant result appears, so it captures ranking position as well as retrieval success.

**The code:**
```text
Recall@5
Recall@10
MRR
```

This distinction matters when reading the CSVs: identical recall can coexist with different MRR because the relevant result can appear at different ranks.

---

### Byte 9: The Retrieval Comparison PNG Is a Visualization Artifact
**Builds on:** Bytes 6–8

**In plain terms:**
`retrieval_comparison.png` is a generated visual artifact corresponding to retrieval comparison work. It is 1500×900 pixels and is not itself an input to the retrieval pipeline.

**The code:**
```text
data/processed/retrieval_comparison.png
1500 × 900
```

The underlying numeric comparison is stored in CSV form, while the PNG is intended for visual inspection/reporting.

---

### Byte 10: `.gitkeep` Is Not Application Data
**Builds on:** None — edge case

**In plain terms:**
The `processed/.gitkeep` file exists to preserve an otherwise empty directory in version control. It has no data-processing responsibility and therefore does not need a Knowledge Byte.

**The code:**
```text
processed/.gitkeep
```

The same applies to `raw/.gitkeep`.

---

## PUTTING IT TOGETHER

`data/processed/` is where the pipeline's generated and evaluation-facing artifacts live. `semantic_cluster.jsonl` represents the current chunked corpus, while `eval_questions.jsonl` supplies the prepared evaluation questions and reference answers. The CSV files summarize retrieval experiments across different systems, and the PNG provides a visual reporting artifact. Together, these files connect the raw corpus to measurable retrieval behavior without being part of the live Neo4j database itself.
