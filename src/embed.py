# src/embed.py
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import numpy as np

# Load model once at import time
model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(chunks: list[str]) -> np.ndarray:
    """Generate embeddings for a list of text chunks."""
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return model.encode([query], convert_to_numpy=True)