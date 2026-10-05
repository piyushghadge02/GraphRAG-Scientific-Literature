# Byte 4: Sentence Embeddings with MiniLM

**Builds on:** Byte 3

## In plain terms

An embedding converts text into a numeric vector that represents its meaning. The project uses `all-MiniLM-L6-v2`, which produces 384-dimensional sentence embeddings.

## The code

```text
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(text)
```

## What's happening

The same embedding model is used conceptually for both stored chunks and incoming questions. At query time, the question becomes a vector that can be compared with chunk vectors using cosine similarity.

## Why it matters

Using the same embedding space makes question-to-chunk semantic retrieval possible. The assignment specifies 384-dimensional embeddings and cosine similarity.
