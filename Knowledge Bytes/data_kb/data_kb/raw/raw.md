# Knowledge Bytes — `data/raw/`

The `raw` folder contains the source data consumed by the GraphRAG pipeline. It is intentionally separate from generated chunks and evaluation outputs.

---

### Byte 1: The Raw Corpus Contains 5,000 PubMed Abstracts
**Builds on:** None — starting point

**In plain terms:**
`abstracts_sample.jsonl` is the main working corpus: 5,000 PubMed-style scientific abstracts. JSONL means each line is one independent JSON record, which makes the file convenient for streaming and batch processing.

**The code:**
```json
{
  "article_id": "pubmed_0_f2ecaec77235",
  "title": "",
  "abstract_text": "background : ... results : ..."
}
```

Each record gives downstream code a stable `article_id` plus the actual abstract text. The project uses this corpus as the input to chunking and later graph construction.

---

### Byte 2: `article_id` Connects Raw Data to the Rest of the Pipeline
**Builds on:** Byte 1

**In plain terms:**
The `article_id` is the identity of the source article. Later, chunk records derive their `chunk_id` from this article ID, allowing the system to trace a retrieved chunk back to its source article.

**The code:**
```json
{
  "article_id": "pubmed_0_f2ecaec77235",
  "title": "",
  "abstract_text": "..."
}
```

This identifier is therefore more important than the title field for pipeline joins.

---

### Byte 3: The Corpus Is a Sample, Not the Entire Dataset
**Builds on:** Byte 1

**In plain terms:**
The file contains 5,000 records, representing the manageable project sample rather than an unrestricted PubMed collection. Keeping a fixed working corpus makes local chunking, indexing, and evaluation practical.

**The code:**
```text
abstracts_sample.jsonl
5,000 lines
```

This fixed corpus also establishes the population against which the project's retrieval experiments are run.

---

### Byte 4: PubMedQA Contexts Are Stored Separately
**Builds on:** Byte 1

**In plain terms:**
`pubmedqa_contexts.jsonl` contains PubMedQA context material separately from the main 5,000-abstract corpus. It contains 1,000 records and uses IDs such as `pqa_<pubmed_id>`.

**The code:**
```json
{
  "article_id": "pqa_21645374",
  "title": "",
  "abstract_text": "{'contexts': ['Programmed cell death ...']}"
}
```

This separation matters because evaluation questions originate from PubMedQA while retrieval operates over the project's selected PubMed corpus.

---

### Byte 5: PubMedQA Contexts Can Contain Multiple Passages
**Builds on:** Byte 4

**In plain terms:**
The PubMedQA record stores a serialized structure containing a `contexts` list rather than a plain abstract string. Multiple context passages can therefore belong to one PubMedQA article.

**The code:**
```text
"{'contexts': [
    'Programmed cell death ...',
    'The following paper elucidates ...',
    ...
]}"
```

Evaluation code must therefore treat these records differently from the simple `abstract_text` records in `abstracts_sample.jsonl`.

---

### Byte 6: Raw Data Is Input, Not the Retrieval Index
**Builds on:** Bytes 1–5

**In plain terms:**
Nothing in `raw/` represents the final Neo4j graph or vector index. These files are the source material from which chunking, embeddings, entities, and graph nodes are later produced.

**The code:**
```text
raw abstracts
    ↓
chunking
    ↓
processed chunks
    ↓
Neo4j graph + vector index
```

Keeping source and generated artifacts separate makes the pipeline easier to rebuild and debug.

---

## PUTTING IT TOGETHER

`data/raw/` is the starting point for the GraphRAG data pipeline. The 5,000-abstract JSONL file is the primary corpus, while PubMedQA contexts provide separate evaluation-related material. `article_id` provides the stable identity used as data moves into chunking and evaluation. The raw files themselves are not the graph; they are the source from which the graph and retrieval artifacts are produced.
