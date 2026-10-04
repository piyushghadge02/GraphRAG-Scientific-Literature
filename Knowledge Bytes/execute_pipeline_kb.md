---
### Byte 1: End-to-end pipeline verifier
**Builds on:** None — starting point

**In plain terms:**
`execute_pipeline.py` is a verification script for the PubMedQA integration path of GraphRAG. It checks that processed data exists, Neo4j is reachable, and PubMedQA chunks and entities reached the graph.

**The code:**
```python
"""
Execute the full data pipeline end-to-end for PubMedQA integration testing.
Steps:
1. Verify fixed_token.jsonl exists and check for pqa_ articles
2. Check semantic_cluster.jsonl
3. Build Neo4j graph
4. Verify pqa_ chunks were inserted
"""
```

The script mainly verifies state; it does not automatically perform the graph build described in its header.

---
### Byte 2: Establish the project root
**Builds on:** Byte 1

**In plain terms:**
The script uses its own location as the project root and adds that directory to Python's import path. This makes the `src` package importable without depending on the directory from which the command was launched.

**The code:**
```python
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.schema import Neo4jConnection
```

The root is also reused for locating processed data files.

---
### Byte 3: Verify the fixed-token chunk file
**Builds on:** Byte 2

**In plain terms:**
The first check requires `fixed_token.jsonl` to exist under `data/processed`. JSONL stores one JSON object per line, which lets the script inspect chunks incrementally.

**The code:**
```python
fixed_token_path = PROJECT_ROOT / "data" / "processed" / "fixed_token.jsonl"

if not fixed_token_path.exists():
    print(f"❌ fixed_token.jsonl not found at {fixed_token_path}")
    return False
```

This is the first hard gate: without the expected processed artifact, the later verification steps are not meaningful.

---
### Byte 4: Count PubMedQA chunks by article ID
**Builds on:** Byte 3

**In plain terms:**
The script counts every non-empty chunk and separately counts records whose `article_id` starts with `pqa_`. That prefix is the convention used here to identify PubMedQA-derived articles.

**The code:**
```python
total_chunks = 0
pqa_chunks = 0

with open(fixed_token_path, "r") as f:
    for line in f:
        if line.strip():
            total_chunks += 1
            data = json.loads(line)
            if data["article_id"].startswith("pqa_"):
                pqa_chunks += 1
```

The resulting counts provide a concrete check that PubMedQA data is present in the processed corpus.

---
### Byte 5: Require PubMedQA data
**Builds on:** Byte 4

**In plain terms:**
Finding the file is not enough; it must contain at least one PubMedQA chunk. If the `pqa_` count is zero, Step 1 fails and the main workflow exits.

**The code:**
```python
if pqa_chunks == 0:
    print("❌ No PubMedQA chunks found!")
    return False
```

The percentage printed afterward is informational; the presence of at least one PubMedQA chunk is the actual gate.

---
### Byte 6: Check semantic-cluster data
**Builds on:** Bytes 3–5

**In plain terms:**
The second step checks for `semantic_cluster.jsonl`, another processed artifact needed by the graph-building workflow. The function treats an existing file with zero PubMedQA chunks as non-critical.

**The code:**
```python
semantic_path = PROJECT_ROOT / "data" / "processed" / "semantic_cluster.jsonl"

if not semantic_path.exists():
    print("⚠️  semantic_cluster.jsonl not found - build_graph requires this file")
    return False
```

This gives the workflow a separate check for the graph-oriented chunking artifact.

---
### Byte 7: Count semantic-cluster coverage
**Builds on:** Byte 6

**In plain terms:**
When the semantic-cluster file exists, the script repeats the total/PubMedQA counting pattern. This makes the data coverage visible before the script talks to Neo4j.

**The code:**
```python
with open(semantic_path, "r") as f:
    for line in f:
        if line.strip():
            total_chunks += 1
            data = json.loads(line)
            if data["article_id"].startswith("pqa_"):
                pqa_chunks += 1
```

If the PubMedQA count is zero, the function still returns `True` because that condition is explicitly treated as non-critical.

---
### Byte 8: Verify Neo4j connectivity
**Builds on:** Bytes 6–7

**In plain terms:**
Before querying the graph, the script performs a minimal Neo4j connectivity check. It opens the project's connection, creates a session, and runs a trivial query.

**The code:**
```python
with Neo4jConnection() as driver:
    with driver.session() as session:
        result = session.run("RETURN 1 as ping")
        result.single()
```

This separates database infrastructure failure from missing-data failure.

---
### Byte 9: Fail fast when Neo4j is unavailable
**Builds on:** Byte 8

**In plain terms:**
If the Neo4j check raises an exception, the step returns `False`. The main workflow then stops instead of attempting graph queries against an unavailable database.

**The code:**
```python
except Exception as e:
    print(f"❌ Neo4j connection failed: {e}")
    return False
```

The broad exception handling is appropriate for this verification script because its job is to report that the infrastructure check failed.

---
### Byte 10: Check PubMedQA chunks in Neo4j
**Builds on:** Bytes 4 and 8

**In plain terms:**
Once Neo4j is reachable, the script counts `Chunk` nodes whose article IDs start with `pqa_`. This verifies that PubMedQA data moved from processed files into the graph.

**The code:**
```python
result = session.run(
    'MATCH (c:Chunk) WHERE c.article_id STARTS WITH "pqa_" RETURN COUNT(c) as count'
)
count = result.single()["count"]
```

The query uses the same article-ID convention as the earlier file checks, so the two stages are testing the same population.

---
### Byte 11: Report an empty PubMedQA graph
**Builds on:** Byte 10

**In plain terms:**
If no PubMedQA chunks exist in Neo4j, the verification fails and prints the command for building the graph. The script deliberately does not mutate the database itself.

**The code:**
```python
if count == 0:
    print("⚠️  No PubMedQA chunks found in Neo4j graph")
    print("   Run `python -m src.graph.build_graph --limit 1000` to build the graph")
    return False
```

This keeps verification separate from graph construction and gives the developer an explicit recovery action.

---
### Byte 12: Verify linked entities
**Builds on:** Byte 10

**In plain terms:**
The script also counts distinct entities connected to PubMedQA chunks through `MENTIONS` relationships. This checks that the graph contains entity structure, not just chunk nodes.

**The code:**
```python
result = session.run(
    """MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
       WHERE c.article_id STARTS WITH "pqa_"
       RETURN COUNT(DISTINCT e) as entity_count"""
)
entity_count = result.single()["entity_count"]
```

Using `DISTINCT` prevents one entity from being counted repeatedly when multiple chunks mention it.

---
### Byte 13: Main function enforces the workflow gates
**Builds on:** Bytes 3–12

**In plain terms:**
`main()` runs the verification steps in dependency order. Missing fixed-token data or an unavailable Neo4j connection stops the script, while the semantic-cluster step decides its own criticality.

**The code:**
```python
if not step1_verify_chunking():
    sys.exit(1)

step2_check_semantic_cluster()

if not step3_verify_neo4j():
    sys.exit(1)

if not step4_check_pqa_in_graph():
    ...
```

The ordering matters: validate processed data first, then database connectivity, then graph contents.

---
### Byte 14: Provide recovery commands
**Builds on:** Byte 13

**In plain terms:**
When graph verification fails, `main()` prints the commands needed to regenerate semantic-cluster chunks, build the Neo4j graph, and rerun the verifier. These commands are guidance, not automatic actions.

**The code:**
```python
print("\n⚠️  Next steps:")
print("   1. Run chunking with semantic_cluster strategy:")
print("      python -m src.chunking.chunker --strategy semantic_cluster --merge-per-article")
print("   2. Build the Neo4j graph:")
print("      python -m src.graph.build_graph --limit 1000")
print("   3. Run this script again to verify")
```

This makes the script useful for diagnosis without hiding potentially expensive or state-changing operations behind the verifier.

---
### Byte 15: Direct execution guard
**Builds on:** Byte 13

**In plain terms:**
The standard Python entry-point guard runs `main()` when the file is executed directly, but not when it is imported by another module.

**The code:**
```python
if __name__ == "__main__":
    main()
```

This makes the verification workflow usable as a standalone command while keeping the module safe to import.

PUTTING IT TOGETHER

`execute_pipeline.py` is a verification and diagnostic script for the PubMedQA integration path, rather than the implementation of the underlying pipeline. It first validates processed chunk artifacts, then checks Neo4j connectivity, and finally confirms that PubMedQA chunks and their linked entities exist in the graph. Critical failures stop execution, while missing graph data produces explicit commands for rebuilding the required artifacts. The entry-point guard makes the complete verification sequence runnable as a standalone script.
