from typing import List, Dict, Tuple
from dependencies import get_chroma_client, get_neo4j_driver
from src.index import load_bm25_index, COLLECTION_NAME
from src.reranker import rerank_results
from src.embed import embed_query
import os

def get_vector_results(query: str, collection_name: str = COLLECTION_NAME, k: int = 10) -> List[Dict]:
    client = get_chroma_client()
    if not client:
        return []
    try:
        # get_or_create so the app never crashes before first upload
        collection = client.get_or_create_collection(name=collection_name)
        if collection.count() == 0:
            print("[retrieval] Vector index is empty — no documents indexed yet.")
            return []

        query_embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
        )

        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "id":       results["ids"][0][i],
                    "text":     results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return formatted_results
    except Exception as e:
        print(f"[retrieval] Vector search error: {e}")
        return []


def get_bm25_results(query: str, k: int = 10) -> List[Dict]:
    index_data = load_bm25_index()
    if not index_data:
        return []
        
    bm25 = index_data['bm25']
    chunks = index_data['chunks']
    ids = index_data['ids']
    
    tokenized_query = query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    
    # Get top k indices
    top_n = scores.argsort()[::-1][:k]
    
    formatted_results = []
    for idx in top_n:
        formatted_results.append({
            'id': ids[idx],
            'text': chunks[idx]['text'],
            'metadata': chunks[idx]['metadata']
        })
    return formatted_results

def get_graph_results(query: str) -> List[str]:
    """Basic graph traversal based on query entities (simplified)."""
    driver = get_neo4j_driver()
    if not driver:
        return []
        
    # Extract entities from query using a simple heuristic or LLM (simplified here)
    # A real implementation would use LLM to extract entities from query, then query Neo4j.
    # We will simulate returning graph context if it matches.
    with driver.session() as session:
        # Example query: find chunks related to entities in the query (mock implementation)
        result = session.run(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk) "
            "WHERE toLower(e.id) CONTAINS toLower($query) "
            "RETURN c.id LIMIT 5",
            query=query[:10] # Naive matching for demonstration
        )
        return [record["c.id"] for record in result]

def reciprocal_rank_fusion(results_list: List[List[Dict]], k=60) -> List[Dict]:
    """Fuses multiple ranked lists using RRF."""
    fused_scores = {}
    chunk_map = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                chunk_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (rank + k)
            
    # Sort by RRF score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[doc_id] for doc_id, _ in sorted_docs]

def hybrid_retrieve_and_rerank(query: str, collection_name: str = "documents", top_k: int = 3) -> List[Dict]:
    """
    1. Retrieve from Vector DB
    2. Retrieve from BM25
    3. Retrieve from Graph (optional enhancement)
    4. Fuse with RRF
    5. Rerank with Cross-Encoder
    """
    # 1 & 2. Parallel retrieval
    vector_results = get_vector_results(query, collection_name, k=10)
    bm25_results = get_bm25_results(query, k=10)
    
    # 3. Fuse
    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])
    
    # 4. Rerank
    final_results = rerank_results(query, fused_results, top_k=top_k)
    return final_results
