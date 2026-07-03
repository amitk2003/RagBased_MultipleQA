# vector_store.py

class VectorStore:
    def build(self, texts, embeddings):
        raise NotImplementedError

    def search(self, query_embedding, k):
        raise NotImplementedError
