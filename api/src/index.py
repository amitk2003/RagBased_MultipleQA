import pickle
import os
from rank_bm25 import BM25Okapi
from dependencies import get_chroma_client
from src.embed import generate_embeddings
from typing import List, Dict

BM25_INDEX_PATH = "data/bm25_index.pkl"
COLLECTION_NAME = "documents"


def _load_existing_bm25() -> dict:
    """Load existing BM25 index from disk, or return empty structure."""
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"chunks": [], "ids": []}


def build_indices(chunks: List[Dict[str, str]], collection_name: str = COLLECTION_NAME):
    """
    Full pipeline step: embed chunks → upsert into ChromaDB → update BM25.

    - Uses the same SentenceTransformer model as retrieval so embeddings match.
    - Appends new chunks to the existing BM25 index (so re-uploads don't wipe old docs).
    - Uses upsert so uploading the same doc twice is safe.
    """
    if not chunks:
        return collection_name

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    documents, metadatas, ids = [], [], []
    for i, chunk in enumerate(chunks):
        doc_id = (
            f"chunk_{chunk['metadata'].get('source', 'unknown')}"
            f"_{chunk['metadata'].get('chunk_index', i)}"
        )
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        ids.append(doc_id)

    # ── 2. Generate embeddings with the SAME model used at query time ─────────
    print(f"[index] Generating embeddings for {len(documents)} chunks…")
    embeddings = generate_embeddings(documents)

    # ── 3. Upsert into ChromaDB ───────────────────────────────────────────────
    chroma_client = get_chroma_client()
    if chroma_client is None:
        print("[index] WARNING: ChromaDB client unavailable — skipping vector index.")
    else:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"[index] ChromaDB upserted {len(documents)} chunks into '{collection_name}'.")

    # ── 4. Append to BM25 index (don't rebuild from scratch) ─────────────────
    existing = _load_existing_bm25()
    # Merge: filter out any chunk ids already in the index (idempotent)
    existing_ids = set(existing["ids"])
    new_chunks = [c for c, cid in zip(chunks, ids) if cid not in existing_ids]
    new_ids    = [cid for cid in ids if cid not in existing_ids]

    all_chunks = existing["chunks"] + new_chunks
    all_ids    = existing["ids"]    + new_ids

    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks, "ids": all_ids}, f)
    print(f"[index] BM25 index updated — total docs: {len(all_chunks)}.")

    return collection_name


def load_bm25_index() -> dict | None:
    """Load the BM25 index from disk. Returns None if not yet created."""
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return None