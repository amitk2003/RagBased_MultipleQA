from sentence_transformers import CrossEncoder
from typing import List, Dict

# Using a lightweight, performant cross-encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

def rerank_results(query: str, retrieved_chunks: List[Dict[str, str]], top_k: int = 3) -> List[Dict[str, str]]:
    """
    Reranks the retrieved chunks against the query using a Cross-Encoder.
    """
    if not retrieved_chunks:
        return []
        
    pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
    
    # Predict scores for each pair
    scores = cross_encoder.predict(pairs)
    
    # Pair chunks with their scores
    scored_chunks = list(zip(retrieved_chunks, scores))
    
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K chunks without scores (or with scores if needed)
    return [chunk for chunk, score in scored_chunks[:top_k]]
