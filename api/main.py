"""
Step 2 — Minimal FastAPI layer over the EXISTING GraphRAG pipeline.

Exposes the exact Streamlit query flow (see app/main.py) without duplicating
any retrieval/generation logic:

    POST /query   -> retrieve_with_pagerank | retrieve | retrieve_decomposed
                     -> build_rag_prompt -> LLMClient.generate
    GET  /health  -> {"status": "ok"}

Run locally:  uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

app = FastAPI(title="GraphRAG API")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Scientific question")
    system_mode: str = Field(
        "Modified GraphRAG",
        description='Same modes as the Streamlit UI: "Modified GraphRAG" or "Baseline"',
    )
    expand_graph: bool = True
    use_adaptive: bool = True
    use_gds: bool = True
    use_decomposition: bool = False
    top_k: int = Field(5, ge=1, le=20)


# Lazily-initialized singletons (mirror app/main.py cached resources).
_driver = None
_embedding_model = None
_llm_client = None


def get_driver():
    """Neo4j driver singleton (same construction as the Streamlit app)."""
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        with _driver.session() as session:
            session.run("RETURN 1")
    return _driver


def get_embedding_model():
    """Embedding model singleton (same construction as the Streamlit app)."""
    global _embedding_model
    if _embedding_model is None:
        from src.embeddings import get_embedder, get_retrieval_model_name

        _embedding_model = get_embedder(get_retrieval_model_name())
    return _embedding_model


def get_llm_client():
    """LLM client singleton (same construction as the Streamlit app)."""
    global _llm_client
    if _llm_client is None:
        from src.generation.llm_client import LLMClient

        _llm_client = LLMClient(
            backend=settings.LLM_BACKEND,
            model=settings.LLM_MODEL,
            temperature=0.0,
        )
    return _llm_client


@app.get("/health")
def health() -> dict:
    """Liveness probe (touches no backends)."""
    return {"status": "ok"}


@app.get("/graph")
def graph_data(chunk_ids: str = "") -> dict:
    """Viz data for the Knowledge Graph tab (same queries graph_view ran locally).

    Query param: comma-separated chunk IDs. Returns Neo4j records as plain
    JSON-serializable dicts so Streamlit Cloud never touches Neo4j.
    """
    ids = [c.strip() for c in chunk_ids.split(",") if c.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="chunk_ids must not be empty.")
    try:
        driver = get_driver()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend unavailable (Neo4j): {e}")
    try:
        with driver.session() as session:
            records = [
                {
                    "chunk_id": r["chunk_id"],
                    "chunk_text": r["chunk_text"],
                    "entity_name": r["entity_name"],
                    "entity_type": r["entity_type"],
                }
                for r in session.run(
                    """
                    MATCH (c:Chunk)-[r:MENTIONS]->(e:Entity)
                    WHERE c.chunk_id IN $chunkIds
                    RETURN c.chunk_id AS chunk_id,
                           c.text AS chunk_text,
                           e.name AS entity_name,
                           'Entity' AS entity_type
                    LIMIT 200
                    """,
                    chunkIds=ids,
                )
            ]
            try:
                similar_pairs = [
                    {
                        "from_id": r["from_id"],
                        "to_id": r["to_id"],
                        "similarity": r["similarity"],
                    }
                    for r in session.run(
                        """
                        MATCH (a:Chunk)-[r:SEMANTIC_SIMILAR]->(b:Chunk)
                        WHERE a.chunk_id IN $chunkIds AND b.chunk_id IN $chunkIds
                        RETURN DISTINCT a.chunk_id AS from_id, b.chunk_id AS to_id,
                               r.similarity AS similarity
                        LIMIT 200
                        """,
                        chunkIds=ids,
                    )
                ]
            except Exception:
                similar_pairs = []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Graph query failed: {e}")
    return {"records": records, "similar_pairs": similar_pairs}


@app.post("/query")
def run_query(req: QueryRequest) -> dict:
    """Run one GraphRAG query with the existing pipeline; return answer + evidence."""
    from src.generation.prompt_templates import build_rag_prompt
    from src.retrieval.decompose import retrieve_decomposed
    from src.retrieval.retrieve import retrieve
    from src.retrieval.retrieve_with_pagerank import retrieve_with_pagerank

    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if req.system_mode not in ("Modified GraphRAG", "Baseline"):
        raise HTTPException(
            status_code=400,
            detail='system_mode must be "Modified GraphRAG" or "Baseline".',
        )

    try:
        driver = get_driver()
        embedding_model = get_embedding_model()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend unavailable (Neo4j/embedding): {e}")

    # Same branching as app/main.py.
    try:
        if req.use_decomposition:
            retrieved_chunks = retrieve_decomposed(
                driver=driver,
                embedding_model=embedding_model,
                question=query,
                llm_client=get_llm_client(),
                top_k_per_subq=3,
                expand_graph=req.expand_graph,
            )
            retrieved_chunks = retrieved_chunks[: req.top_k]
        elif req.system_mode == "Baseline":
            retrieved_chunks = retrieve(
                driver=driver,
                embedding_model=embedding_model,
                query=query,
                top_k=req.top_k,
                expand_graph=req.expand_graph,
                adaptive_enabled=False,
                alpha=0.7,
                beta=0.3,
                gamma=0.0,
            )
            for c in retrieved_chunks:
                c["pagerank_score"] = 0.0
                c["final_score"] = c["combined_score"]
        else:
            retrieved_chunks = retrieve_with_pagerank(
                driver=driver,
                embedding_model=embedding_model,
                query=query,
                top_k=req.top_k,
                expand_graph=req.expand_graph,
                use_pagerank=req.use_gds,
                adaptive_enabled=req.use_adaptive,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}")

    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    try:
        prompt = build_rag_prompt(query, retrieved_chunks)
        answer = get_llm_client().generate(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    return {
        "answer": answer,
        "query": query,
        "system_mode": req.system_mode,
        "retrieval_strategy": retrieved_chunks[0].get("retrieval_strategy", "n/a"),
        "chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
    }
