# src/config.py — Central configuration for the RAG pipeline

# ── Answer mode ──────────────────────────────────────────────────────────────
# "STRICT_RAG" → only answer from the document; "HYBRID" → fallback to general knowledge
MODE = "HYBRID"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000  # characters per chunk (per page; pdfplumber extracts full pages)
CHUNK_OVERLAP = 100   # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 4             # number of top chunks after reranking

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"     # local sentence-transformers embedding model
LLM_PROVIDER       = "groq"                # default LLM provider: groq | openai | ollama
LLM_MODEL          = "openai/gpt-oss-120b"        # Groq model name
MAX_RESPONSE_LENGTH = 1024

# ── Output format ─────────────────────────────────────────────────────────────
# "ANSWER_ONLY"  → clean plain answer
# "WITH_SOURCES" → answer + source citations
ANSWER_MODE = "WITH_SOURCES"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_documents"
CHROMA_DB_PATH  = "data/chroma_db"   # directory for persistent ChromaDB storage
