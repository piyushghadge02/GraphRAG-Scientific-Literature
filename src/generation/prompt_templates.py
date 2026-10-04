"""
Task 6 — Prompt Templates
Builds RAG prompts that guide the LLM to use retrieved context.
Includes chunk numbering, citation instructions, and explicit handling of insufficient context.
"""

from typing import List


def build_rag_prompt(question: str, chunks: List[dict]) -> str:
    """
    Build a RAG prompt that instructs the LLM to answer using provided context.

    The prompt:
    - Numbers each retrieved chunk [1], [2], etc. with its full text
    - Instructs the LLM to cite which chunk(s) support each claim
    - Requires explicit "I don't have enough information" if context is insufficient

    Args:
        question: User's question
        chunks: List of retrieved chunks, each with 'text' key (and optionally chunk_id for reference)

    Returns:
        Formatted prompt string ready to send to the LLM
    """

    # Build context string with numbered chunks
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        context_lines.append(f"[{i}] (from {chunk_id}):\n{chunk_text}\n")

    context_str = "\n".join(context_lines)

    # Build the full prompt
    prompt = f"""You are a scientific question-answering assistant. Your task is to answer the following question based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using the provided context below. Do not use any external knowledge.
2. For each claim you make, explicitly cite which chunk(s) support it by referencing [1], [2], etc.
3. If the provided context does not contain enough information to answer the question, explicitly state: "I don't have enough information to answer this question based on the provided context."
4. Be precise and factual. Quote relevant sections if needed.
5. If the question cannot be fully answered from the context, provide what you can and note what information is missing.

RETRIEVED CONTEXT:
{context_str}

QUESTION:
{question}

ANSWER:
"""

    return prompt
