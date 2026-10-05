# Byte 12: LLM Answer Generation

**Builds on:** Byte 11

## In plain terms

The LLM is the final language-generation component. It does not replace retrieval; it receives the question plus retrieved scientific context and produces a natural-language answer grounded in that context.

## The code

```text
prompt = f"""
Answer the question using the retrieved scientific context.

Question:
{question}

Context:
{retrieved_chunks}
"""
```

## What's happening

The backend sends a prompt containing the user question and selected evidence to the configured LLM backend. The current project uses Groq for model inference, with the model configured through environment variables.

## Why it matters

Retrieval supplies evidence; the LLM turns that evidence into a readable answer. Keeping these responsibilities separate makes the system easier to evaluate and debug.
