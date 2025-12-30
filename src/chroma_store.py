import chromadb
from chromadb.config import Settings

class ChromaStore:
    def __init__(self, collection_name="rag_docs"):
        self.client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.create_collection(
            name=collection_name
        )

    def build(self, texts, embeddings):
        ids = [str(i) for i in range(len(texts))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )

    def search(self, query_embedding, k):
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return result["documents"][0]
