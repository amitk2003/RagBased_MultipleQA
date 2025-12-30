# src/index.py
import chromadb
from chromadb.config import Settings
from config import TOP_K

client = chromadb.Client(
    Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="rag_documents"
)

def build_faiss_index(embeddings, chunks):
    # Clear collection safely
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )
    return collection

def retrieve_chunks(collection, query_embedding, chunks):
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K
    )

    retrieved_docs = results["documents"][0]
    retrieved_ids = [int(i) for i in results["ids"][0]]

    return retrieved_docs, retrieved_ids
