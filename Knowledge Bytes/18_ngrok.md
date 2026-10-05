# Byte 19: ngrok Hosting

**Builds on:** Byte 18

## In plain terms

ngrok exposes the local Streamlit port through a public HTTPS URL. It is a hosting/tunneling layer, not part of the GraphRAG retrieval algorithm.

## The code

```text
# Public tunnel
ngrok http 8502

# Concept
Internet
   ↓
ngrok HTTPS URL
   ↓
localhost:8502
   ↓
Streamlit
   ↓
FastAPI :8000
   ↓
Neo4j :7687
```

## What's happening

The demonstrated deployment exposes Streamlit port 8502. Streamlit continues to call the local FastAPI service, which talks to Neo4j and the LLM provider on the same machine.

## Why it matters

ngrok makes the demo accessible to an evaluator without deploying Neo4j or the backend to a cloud server. The tunnel remains dependent on the laptop and running local services.
