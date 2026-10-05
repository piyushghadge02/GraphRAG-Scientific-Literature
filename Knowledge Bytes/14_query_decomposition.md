# Byte 15: Query Decomposition

**Builds on:** Byte 14

## In plain terms

Complex multi-hop questions can be split into smaller sub-questions. Each sub-question can be retrieved separately and the results can then be merged and reranked.

## The code

```text
complex question
      ↓
sub-question 1
sub-question 2
sub-question 3
      ↓
retrieve each
      ↓
merge + deduplicate
      ↓
rerank
      ↓
LLM
```

## What's happening

Phase 5 calls for a lightweight LLM to decompose a complex query, retrieve evidence for each part, and merge the results. The current UI includes a Query Decomposition option.

## Why it matters

Decomposition can improve coverage for questions that contain several relationships or reasoning steps. It is an optimization for multi-hop questions, not a replacement for the base retriever.
