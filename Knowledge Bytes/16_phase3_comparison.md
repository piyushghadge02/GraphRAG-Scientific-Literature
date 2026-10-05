# Byte 17: Phase 3: Graph Expansion ON vs OFF

**Builds on:** Byte 10

## In plain terms

Phase 3 requires manual testing on 10 PubMedQA questions with graph expansion disabled and enabled. The goal is to observe whether adding graph context changes retrieval and answer quality.

## The code

```text
# Conceptual experiment
for question in test_questions:
    answer_off = run(question, expand_graph=False)
    answer_on  = run(question, expand_graph=True)

    compare(answer_off, answer_on)
```

## What's happening

The vector retrieval stage remains the common baseline. The experimental variable is graph expansion. A comparison table can record retrieved chunks, answer quality, evidence coverage and qualitative differences.

## Why it matters

This isolates the value of the graph component. Without an ON/OFF comparison, it is harder to argue that the graph actually contributes beyond ordinary vector RAG.
