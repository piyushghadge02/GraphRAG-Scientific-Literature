from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from src.config import settings

uri = settings.NEO4J_URI
user = settings.NEO4J_USER
password = settings.NEO4J_PASSWORD
driver = GraphDatabase.driver(uri, auth=(user, password))

embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
query = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"

from src.retrieval.vector_search import vector_search
from src.retrieval.graph_expand import expand_via_entities
from src.retrieval.fusion import combined_score
import numpy as np

query_embedding = embedding_model.encode(query).tolist()
vector_results = vector_search(driver, query_embedding, top_k=15)
vector_chunk_ids = [r["chunk_id"] for r in vector_results]

# 1. Run expansion with seeds allowed
expanded_results = expand_via_entities(driver, vector_chunk_ids, max_depth=1)

# Build candidate dicts
candidates = {}
for r in vector_results:
    cid = r["chunk_id"]
    candidates[cid] = {
        "chunk_id": cid,
        "is_vector_seed": True,
        "vector_score": r["score"],
        "shared_entities": 0
    }

query_arr = np.array(query_embedding, dtype=np.float32)
norm_q = np.linalg.norm(query_arr)
if norm_q > 0:
    query_arr = query_arr / norm_q

for r in expanded_results:
    cid = r["chunk_id"]
    shared = r["shared_entities"]

    if cid in candidates:
        candidates[cid]["shared_entities"] = shared
    else:
        # Compute vector score on the fly
        cand_emb = r.get("embedding")
        vector_score = 0.0
        if cand_emb:
            cand_arr = np.array(cand_emb, dtype=np.float32)
            norm_c = np.linalg.norm(cand_arr)
            if norm_c > 0:
                cand_arr = cand_arr / norm_c
            vector_score = float(np.dot(query_arr, cand_arr))

        candidates[cid] = {
            "chunk_id": cid,
            "is_vector_seed": False,
            "vector_score": vector_score,
            "shared_entities": shared
        }

# Compute combined scores for different alphas
max_shared_entities = max((c["shared_entities"] for c in candidates.values()), default=0)

for alpha in [0.6, 0.5, 0.4]:
    print(f"\n=== RANKING WITH ALPHA = {alpha} ===")
    for cid, c in candidates.items():
        c["final_combined_score"] = combined_score(
            vector_score=c["vector_score"],
            shared_entities=c["shared_entities"],
            max_shared_entities=max(max_shared_entities, 1),
            alpha=alpha
        )

    ranked = sorted(candidates.values(), key=lambda x: x["final_combined_score"], reverse=True)
    for idx, r in enumerate(ranked[:5]):
        is_seed = r["chunk_id"] in vector_chunk_ids
        print(f"  {idx+1}. {r['chunk_id']} (is_seed={is_seed}, combined={r['final_combined_score']:.4f}, vector={r['vector_score']:.4f}, shared_ent={r['shared_entities']})")

driver.close()
