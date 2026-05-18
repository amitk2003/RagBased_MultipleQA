# src/index.py  — Vector store backed by ChromaDB
import os

import chromadb
from chromadb.config import Settings

try:
    from config import TOP_K, COLLECTION_NAME, CHROMA_DB_PATH
except ModuleNotFoundError:
    from src.config import TOP_K, COLLECTION_NAME, CHROMA_DB_PATH

# ---------------------------------------------------------------------------
# ChromaDB persistent client — lazy singleton
# ---------------------------------------------------------------------------
# Resolve the DB path relative to the project root (one level above src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_PROJECT_ROOT, CHROMA_DB_PATH)

_chroma_client: chromadb.PersistentClient = None


def _get_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB client, creating it on first call."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_DB_PATH)
    return _chroma_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chroma_index(embeddings: list, chunks: list, collection_name: str = None) -> str:
    """
    Build (or rebuild) a ChromaDB collection from embeddings + text chunks.

    Parameters
    ----------
    embeddings      : list[list[float]]  — one flat embedding vector per chunk
    chunks          : list[str]          — original text chunks
    collection_name : str                — ChromaDB collection name

    Returns
    -------
    str — the collection name (store this in Streamlit session_state)
    """
    if collection_name is None:
        collection_name = COLLECTION_NAME

    client = _get_client()

    # Delete existing collection so we always start fresh when re-indexing
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    # cosine similarity is the standard for sentence-transformer embeddings
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB requires string IDs
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,   # list[list[float]]
        documents=chunks,        # raw text stored alongside embeddings
    )

    print(f"[index] ChromaDB indexing complete — {len(chunks)} vectors stored "
          f"in collection '{collection_name}'.")
    return collection_name


def retrieve_chunks(collection_name: str, query_vector: list, top_k: int = None):
    """
    Retrieve the top-k most similar chunks to a query vector.

    Parameters
    ----------
    collection_name : str
    query_vector    : list[float]  — 1-D flat embedding (NOT 2-D)
    top_k           : int

    Returns
    -------
    (list[str], list[str])  — (text chunks, IDs)
    """
    if top_k is None:
        top_k = TOP_K

    client = _get_client()

    # Check collection exists
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        print(f"[index] Collection '{collection_name}' not found. "
              "Please index documents first.")
        return [], []

    collection = client.get_collection(name=collection_name)

    # Sanity check — don't query an empty collection
    if collection.count() == 0:
        print(f"[index] Collection '{collection_name}' is empty.")
        return [], []

    results = collection.query(
        query_embeddings=[query_vector],  # must be a list of vectors
        n_results=min(top_k, collection.count()),
        include=["documents"],            # we only need the text back
    )

    # results["documents"] is [[chunk1, chunk2, ...]] (outer list = one query)
    retrieved_chunks = results["documents"][0]
    chunk_ids        = results["ids"][0]

    return retrieved_chunks, chunk_ids