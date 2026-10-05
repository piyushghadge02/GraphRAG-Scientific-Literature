# Byte 13: FastAPI Backend

**Builds on:** Byte 12

## In plain terms

FastAPI is the backend HTTP layer. It exposes the GraphRAG functionality so the UI can call the retrieval/generation pipeline without containing all backend logic itself.

## The code

```text
uvicorn api.main:app --host 127.0.0.1 --port 8000

# UI-side concept
POST /... 
{
  "question": "...",
  "top_k": 5,
  "expand_graph": true
}
```

## What's happening

Uvicorn runs the FastAPI application on port 8000. The API coordinates retrieval, graph access and LLM generation and returns structured results to the application layer.

## Why it matters

Separating the API from the UI keeps the system modular. It also made local debugging easier because FastAPI and Streamlit could be tested independently.
