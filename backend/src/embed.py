# src/embed.py
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from config import EMBEDDING_MODEL
except ModuleNotFoundError:
    from src.config import EMBEDDING_MODEL

# Suppress the FutureWarning from transformers about clean_up_tokenization_spaces
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

# Load model once at import time (cached for the entire process lifetime)
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def generate_embeddings(chunks: list) -> list:
    """
    Generate embeddings for a list of text chunks.
    Returns a plain Python list of float lists (ChromaDB-compatible).
    """
    model = _get_model()
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # Convert numpy array → list of plain Python lists for ChromaDB compatibility
    return embeddings.tolist()


def embed_query(query: str) -> list:
    """
    Embed a single query string.
    Returns a 1-D plain Python list (ChromaDB expects a flat vector, not 2-D).
    """
    model = _get_model()
    # encode() with a single string returns shape (dim,) — flat and correct
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding.tolist()