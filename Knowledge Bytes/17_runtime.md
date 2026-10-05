# Byte 18: Runtime, Environment and Services

**Builds on:** Byte 1

## In plain terms

The project runs as multiple local services. Neo4j stores the graph, FastAPI serves the backend, and Streamlit serves the UI. Environment variables keep credentials and service URLs outside source code.

## The code

```text
# FastAPI
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Streamlit
python -m streamlit run app.py --server.port 8502

# Important configuration concept
NEO4J_URI=...
NEO4J_USER=...
NEO4J_PASSWORD=...
GROQ_API_KEY=...
GRAPHRAG_API_URL=http://localhost:8000
```

## What's happening

Neo4j must be running before graph operations work. FastAPI listens on 8000 and Streamlit on 8502. The `.env` file contains secrets and local configuration and must not be committed to GitHub.

## Why it matters

Most runtime errors are dependency/service errors rather than GraphRAG algorithm errors. Keeping the service boundaries clear makes troubleshooting much faster.
