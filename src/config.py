# config.py - Central place for easy tweaking
# strict_rag: answer only from document else said i dont know
# hybrid: use document  if relaevant , otherwise answer from general knowlwdge
MODE="HYBRID"
CHUNK_SIZE = 400          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks for context
TOP_K = 4                 # Number of chunks to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight and fast
LLM_MODEL = "google/flan-t5-small"        # Easy local; replace with better if needed
MAX_RESPONSE_LENGTH = 300
ANSWER_MODE="ANSWER_ONLY"