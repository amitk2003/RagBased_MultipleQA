import pytest
from unittest.mock import patch, MagicMock
from src.retrieval import hybrid_retrieve_and_rerank, reciprocal_rank_fusion

@pytest.fixture
def mock_chroma_results():
    return [
        {"id": "doc1", "text": "This is doc 1", "metadata": {"source": "test.pdf"}},
        {"id": "doc2", "text": "This is doc 2", "metadata": {"source": "test.pdf"}},
    ]

@pytest.fixture
def mock_bm25_results():
    return [
        {"id": "doc2", "text": "This is doc 2", "metadata": {"source": "test.pdf"}},
        {"id": "doc3", "text": "This is doc 3", "metadata": {"source": "test.pdf"}},
    ]

def test_reciprocal_rank_fusion(mock_chroma_results, mock_bm25_results):
    # doc2 is in both, should rank highest
    fused = reciprocal_rank_fusion([mock_chroma_results, mock_bm25_results])
    
    assert len(fused) == 3
    assert fused[0]['id'] == 'doc2' # Should have highest score because it's in both lists

@patch('src.retrieval.get_vector_results')
@patch('src.retrieval.get_bm25_results')
@patch('src.retrieval.rerank_results')
def test_hybrid_retrieve_and_rerank(mock_rerank, mock_bm25, mock_vector, mock_chroma_results, mock_bm25_results):
    mock_vector.return_value = mock_chroma_results
    mock_bm25.return_value = mock_bm25_results
    
    # Mock reranker to just return the top k elements from fused results
    def mock_reranker_fn(query, fused_results, top_k):
        return fused_results[:top_k]
        
    mock_rerank.side_effect = mock_reranker_fn
    
    results = hybrid_retrieve_and_rerank("test query", top_k=2)
    
    assert len(results) == 2
    assert results[0]['id'] == 'doc2' # Top result after fusion
