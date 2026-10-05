# Byte 16: Phase 4 Evaluation

**Builds on:** Byte 14

## In plain terms

Phase 4 measures whether the system retrieves the right article/chunks and whether its generated answers resemble the provided reference answers. The assignment asks for about 200 PubMedQA questions matching the 5,000-paper corpus.

## The code

```text
# Retrieval metrics
Recall@5
Recall@10
MRR

# Generation metrics
ROUGE-L
BERTScore
```

## What's happening

For each evaluation question, the gold PubMed article is known from the PubMed ID. The evaluator compares the retrieved top-5/top-10 results with that gold article's chunks, then generates an answer and compares it with the PubMedQA long answer.

## Why it matters

A demo can look good while retrieving weak evidence. Metrics make the project measurable and let us quantify retrieval quality and answer similarity instead of relying only on visual inspection.
