"""
Phase 5 - Query Decomposition
Decomposes complex queries into simpler sub-queries using LLM.
Retrieves results for each sub-query and merges them intelligently.
"""

import json
from typing import List, Dict
from src.retrieval.retrieve import retrieve


def decompose_query(llm_client, question: str) -> List[str]:
    """
    Use LLM to determine if query decomposition is needed and generate sub-queries.

    The LLM analyzes the question and decides whether to:
    - Return the original question as-is (simple query)
    - Break it into multiple sub-queries (complex query)

    Args:
        llm_client: LLMClient instance for generation
        question: User's original question

    Returns:
        List of sub-queries. Returns [question] as fallback on any error.
    """

    system_prompt = """You are a query analysis assistant. Your task is to determine if a scientific question should be decomposed into simpler sub-questions for better retrieval.

Return a JSON array of questions. For simple questions, return the original question in an array. For complex questions, break them into focused sub-questions.

Examples:
Input: "What is apoptosis?"
Output: ["What is apoptosis?"]

Input: "What is the relationship between mitochondrial dysfunction and programmed cell death, and how does this affect plant development?"
Output: ["What is mitochondrial dysfunction?", "What is programmed cell death?", "How does mitochondrial dysfunction relate to programmed cell death?", "How does this relationship affect plant development?"]

Input: "Do mitochondria play a role in programmed cell death?"
Output: ["Do mitochondria play a role in programmed cell death?"]

Rules:
- Return ONLY a valid JSON array of strings
- No explanations, no markdown, just the JSON array
- If the question is simple and focused, return it as-is
- If the question has multiple parts or asks about relationships between concepts, decompose it
- Each sub-question should be self-contained and answerable independently
"""

    prompt = f"""Question: {question}

Sub-questions (as JSON array):"""

    try:
        response = llm_client.generate(prompt, system_prompt=system_prompt)

        # Try to extract JSON from response (handles cases where LLM adds markdown)
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            response = response.replace("```json", "").replace("```", "").strip()

        # Parse JSON array
        sub_queries = json.loads(response)

        # Validate it's a list of strings
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            if len(sub_queries) == 0:
                return [question]
            return sub_queries
        else:
            print(f"[decompose] WARNING: Invalid response format. Returning original question.")
            return [question]

    except json.JSONDecodeError as e:
        print(f"[decompose] WARNING: Failed to parse JSON from LLM response: {e}")
        print(f"[decompose] Response was: {response}")
        return [question]
    except Exception as e:
        print(f"[decompose] WARNING: Error during query decomposition: {e}")
        return [question]


def retrieve_decomposed(
    driver,
    embedding_model,
    question: str,
    llm_client,
    top_k_per_subq: int = 3,
    expand_graph: bool = True
) -> List[Dict]:
    """
    Decompose query, retrieve for each sub-query, merge and dedupe results.

    Pipeline:
    1. Decompose query into sub-queries using LLM
    2. Retrieve top_k_per_subq results for each sub-query
    3. Merge all results and dedupe by chunk_id
    4. For duplicates, keep the one with highest combined_score
    5. Re-sort by combined_score and return top results

    Args:
        driver: Neo4j driver connection
        embedding_model: SentenceTransformer model
        question: User's original question
        llm_client: LLMClient for query decomposition
        top_k_per_subq: Number of results to retrieve per sub-query (default: 3)
        expand_graph: Whether to use graph expansion during retrieval (default: True)

    Returns:
        List of deduplicated chunks sorted by combined_score
    """

    # Step 1: Decompose query
    sub_queries = decompose_query(llm_client, question)

    print(f"\n[decompose] Query decomposition:")
    print(f"  Original: {question}")
    print(f"  Sub-queries ({len(sub_queries)}): {sub_queries}")

    # Step 2: Retrieve for each sub-query
    all_results = {}  # chunk_id -> chunk dict (keep highest combined_score)

    for i, sub_q in enumerate(sub_queries, 1):
        print(f"[decompose] Retrieving for sub-query {i}/{len(sub_queries)}: {sub_q}")

        results = retrieve(
            driver=driver,
            embedding_model=embedding_model,
            query=sub_q,
            top_k=top_k_per_subq,
            expand_graph=expand_graph
        )

        # Merge results, keeping highest combined_score for duplicates
        for chunk in results:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in all_results:
                all_results[chunk_id] = chunk
            else:
                # Keep the chunk with higher combined_score
                if chunk["combined_score"] > all_results[chunk_id]["combined_score"]:
                    all_results[chunk_id] = chunk

    # Step 3: Sort by combined_score descending
    merged_results = sorted(
        all_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )

    print(f"[decompose] Merged results: {len(merged_results)} unique chunks after deduplication")

    return merged_results
