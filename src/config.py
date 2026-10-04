import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PDF-required baseline embedding model. Central definition — all modules
# must reference this constant instead of duplicating the literal.
# Default MUST remain the baseline below (384D).
BASELINE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@dataclass(frozen=True)
class Config:
    # Neo4j Settings
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Groq & LLM Settings
    GROQ_API_KEY1: str = os.getenv("GROQ_API_KEY1", os.getenv("GROQ_API_KEY", ""))
    GROQ_API_KEY2: str = os.getenv("GROQ_API_KEY2", "")
    GROQ_API_KEY3: str = os.getenv("GROQ_API_KEY3", "")
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    @property
    def GROQ_API_KEYS(self) -> list[str]:
        # Skip None or empty strings, preserve order
        keys = [self.GROQ_API_KEY1, self.GROQ_API_KEY2, self.GROQ_API_KEY3]
        return [k.strip() for k in keys if k and k.strip()]

    # Retrieval / Chunking Settings
    # PDF-required baseline retrieval embedding model. Default MUST remain
    # the baseline (see BASELINE_EMBEDDING_MODEL). Change via
    # EMBEDDING_MODEL_NAME env var for Experiment B. Changing the model
    # requires rebuilding/reindexing stored Chunk embeddings and the Neo4j
    # vector index.
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", BASELINE_EMBEDDING_MODEL)
    # PDF-required semantic-clustering model. MUST remain the baseline.
    # Kept separate from EMBEDDING_MODEL_NAME so retrieval embeddings can
    # vary (Experiment B) while semantic chunking stays on the baseline.
    SEMANTIC_CLUSTER_MODEL_NAME: str = os.getenv("SEMANTIC_CLUSTER_MODEL_NAME", BASELINE_EMBEDDING_MODEL)
    CHUNK_TOKEN_LIMIT: int = int(os.getenv("CHUNK_TOKEN_LIMIT", "100"))
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "5"))
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))
    ALPHA_FUSION_WEIGHT: float = float(os.getenv("ALPHA_FUSION_WEIGHT", "0.7"))

    # Phase 2 — Graph expansion settings (PDF traversal: Chunk->Entity->Chunk, depth <= 2)
    GRAPH_EXPANSION_ENABLED: bool = os.getenv("GRAPH_EXPANSION_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    GRAPH_MAX_DEPTH: int = int(os.getenv("GRAPH_MAX_DEPTH", "1"))
    GRAPH_MAX_EXPANDED_CHUNKS: int = int(os.getenv("GRAPH_MAX_EXPANDED_CHUNKS", "500"))
    GRAPH_MIN_SHARED_ENTITIES: int = int(os.getenv("GRAPH_MIN_SHARED_ENTITIES", "2"))
    GRAPH_MIN_SIMILARITY: float = float(os.getenv("GRAPH_MIN_SIMILARITY", "0.3"))
    # Maximum MENTIONS degree an Entity may have to participate in expansion.
    # 0 (default) disables the cap. Set when hub entities ("cells", ...) make
    # expansion fan out uncontrollably; traversal itself is unchanged.
    GRAPH_MAX_ENTITY_DEGREE: int = int(os.getenv("GRAPH_MAX_ENTITY_DEGREE", "0"))

    # Phase 2 — 3-term fusion weights: alpha*vector + beta*entity + gamma*proximity
    # (rescaled to sum to 1; defaults reproduce the Phase 1 ranking exactly).
    FUSION_BETA_WEIGHT: float = float(os.getenv("FUSION_BETA_WEIGHT", "0.3"))
    FUSION_GAMMA_WEIGHT: float = float(os.getenv("FUSION_GAMMA_WEIGHT", "0.0"))

    # Phase 2 — PageRank (GDS) reranking settings (optional, disabled path preserved)
    ENABLE_PAGERANK: bool = os.getenv("ENABLE_PAGERANK", "true").strip().lower() in ("1", "true", "yes")
    PAGERANK_WEIGHT: float = float(os.getenv("PAGERANK_WEIGHT", "0.15"))
    PAGERANK_MAX_ITERATIONS: int = int(os.getenv("PAGERANK_MAX_ITERATIONS", "20"))
    PAGERANK_DAMPING_FACTOR: float = float(os.getenv("PAGERANK_DAMPING_FACTOR", "0.85"))

    # Phase 2 — Adaptive retrieval: skip graph expansion when vector retrieval
    # is already high-confidence (deterministic, rule-based, logged).
    ADAPTIVE_RETRIEVAL_ENABLED: bool = os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    ADAPTIVE_TOP1_THRESHOLD: float = float(os.getenv("ADAPTIVE_TOP1_THRESHOLD", "0.90"))
    ADAPTIVE_GAP_THRESHOLD: float = float(os.getenv("ADAPTIVE_GAP_THRESHOLD", "0.05"))

    @property
    def IS_BASELINE_EMBEDDING(self) -> bool:
        """True when the retrieval model is the PDF-required baseline."""
        return self.EMBEDDING_MODEL_NAME == BASELINE_EMBEDDING_MODEL

    @property
    def EXPERIMENT_LABEL(self) -> str:
        """Human-readable experiment label: 'A (baseline)' or 'B (custom)'."""
        return "A (baseline)" if self.IS_BASELINE_EMBEDDING else "B (custom)"

# Global config instance
settings = Config()
