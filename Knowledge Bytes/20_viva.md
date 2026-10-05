# Byte 21: Limitations and Viva Answers

**Builds on:** Bytes 1–20

## In plain terms

The safest viva strategy is to explain the actual architecture first, then the reason for each technology. Do not claim a feature is better just because it exists; explain how it is measured.

## The code

```text
5000 abstracts
 → chunking
 → embeddings
 → Neo4j graph + vector index
 → vector retrieval
 → graph expansion
 → reranking
 → LLM
 → evaluation
 → Streamlit
```

## What's happening

### Q1. What is GraphRAG?
**Answer:** GraphRAG combines retrieval based on semantic similarity with graph-based traversal so that connected entities and chunks can contribute to the final context.

### Q2. Why Neo4j?
**Answer:** Scientific information naturally has relationships between articles, chunks and biomedical entities. Neo4j represents and traverses these relationships directly.

### Q3. Why embeddings?
**Answer:** Embeddings convert text into vectors so questions and chunks can be compared by semantic similarity.

### Q4. Why `all-MiniLM-L6-v2`?
**Answer:** It is the sentence-transformer specified by the assignment and provides 384-dimensional embeddings suitable for semantic retrieval.

### Q5. Why semantic chunking?
**Answer:** It tries to keep semantically related sentences together instead of cutting text only by token count.

### Q6. What is graph expansion?
**Answer:** Starting from vector-retrieved chunks, the system follows shared entities using `Chunk → Entity ← Chunk` to discover related chunks.

### Q7. Why reranking?
**Answer:** Graph expansion can produce many candidates, so reranking selects the most useful evidence for the final context.

### Q8. What is PageRank doing?
**Answer:** It provides a graph-centrality signal that can help prioritize structurally important chunks during Phase 5 reranking.

### Q9. Why 200 PubMedQA questions?
**Answer:** Phase 4 asks for about 200 questions whose PubMed IDs match the 5,000-abstract corpus. This gives a known gold article for retrieval evaluation.

### Q10. What are Recall@5 and Recall@10?
**Answer:** They measure whether the gold article/chunk appears within the first 5 or 10 retrieved results.

### Q11. What is MRR?
**Answer:** Mean Reciprocal Rank measures how high the first relevant result appears. Rank 1 contributes 1, rank 2 contributes 0.5, rank 5 contributes 0.2.

### Q12. ROUGE-L vs BERTScore?
**Answer:** ROUGE-L emphasizes sequence overlap using the longest common subsequence, while BERTScore measures contextual semantic similarity.

### Q13. Why compare graph expansion ON and OFF?
**Answer:** It isolates the contribution of graph expansion against the vector retrieval baseline.

### Q14. What does Streamlit do?
**Answer:** It provides the interactive demonstration interface; it is not the retrieval engine itself.

### Q15. What does FastAPI do?
**Answer:** It provides the backend API layer that connects the UI to the GraphRAG pipeline.

### Q16. What does ngrok do?
**Answer:** It exposes the local Streamlit service through a public HTTPS URL. It is not part of the RAG algorithm.

### Q17. What are the main limitations?
**Answer:** Retrieval quality depends on chunk quality and embedding quality; graph expansion can introduce noisy neighbors; LLM answers can still be imperfect; local Neo4j and the local services must remain running for the demo; and evaluation metrics do not fully replace expert scientific judgment.

## Why it matters

The strongest viva answer is not a list of libraries. It is the reasoning:

> **We chose semantic chunking to improve retrieval units, MiniLM to represent meaning, Neo4j to represent relationships, vector search for semantic seeds, graph expansion for connected evidence, reranking to control the candidate set, an LLM for natural-language generation, and quantitative metrics to evaluate the result.**

## PUTTING IT TOGETHER

The system starts with scientific abstracts and transforms them into chunks that can be searched. Embeddings provide semantic retrieval while Neo4j provides graph structure through entities and relationships. The query pipeline combines those signals, reranks the evidence, and gives the selected context to the LLM. Phase 4 measures retrieval and generation quality, and Phase 5 exposes the complete pipeline through an interactive Streamlit application.
