# tests Knowledge Base

This Knowledge Base explains the complete `tests` folder of the GraphRAG project.

The tests are organized around the project's development phases and submission requirements:

- `test_phase1_embeddings.py` — embedding configuration and retrieval foundation.
- `test_phase2_graph.py` — graph expansion, fusion, PageRank, and adaptive retrieval.
- `test_phase3_eval.py` — evaluation metrics, system comparison, and UI-facing retrieval metadata.
- `test_eval_matching.py` — evaluation-question/corpus matching correctness.
- `test_compliance.py` — checks required by the submission/compliance specification.

The tests deliberately avoid live external services. They use fake embedders, fake Neo4j drivers, stub modules, temporary files, and mocks so that the important logic can be exercised deterministically.

The phase-oriented tests build conceptually from the embedding foundation, through graph-aware retrieval, to evaluation and product-facing behavior. The matching and compliance tests then protect important data-alignment and deliverable requirements.
