# src/config.py — Central configuration for the RAG pipeline

# ── Answer mode ──────────────────────────────────────────────────────────────
# "STRICT_RAG" → only answer from the document; "HYBRID" → fallback to general knowledge
MODE = "HYBRID"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 4             # number of top chunks to return per query

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"    # sentence-transformers model
LLM_MODEL          = "google/flan-t5-small" # local HuggingFace generative model
MAX_RESPONSE_LENGTH = 300

# ── Output format ─────────────────────────────────────────────────────────────
# "ANSWER_ONLY"  → clean plain answer
# "WITH_SOURCES" → answer + source citations
ANSWER_MODE = "ANSWER_ONLY"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_documents"
CHROMA_DB_PATH  = "chroma_db"   # directory for persistent ChromaDB storage