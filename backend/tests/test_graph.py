import pytest
from unittest.mock import patch, MagicMock
from src.graph_builder import process_chunk_for_graph, extract_graph_data, GraphData, Node, Relationship

def test_extract_graph_data():
    # Test that the Pydantic models for extraction are working correctly
    node1 = Node(id="Alice", label="PERSON")
    node2 = Node(id="Bob", label="PERSON")
    rel = Relationship(source="Alice", target="Bob", type="KNOWS")
    
    graph_data = GraphData(nodes=[node1, node2], relationships=[rel])
    
    assert len(graph_data.nodes) == 2
    assert len(graph_data.relationships) == 1
    assert graph_data.relationships[0].type == "KNOWS"

@patch('src.graph_builder.extract_graph_data')
@patch('src.graph_builder.load_graph_to_neo4j')
def test_process_chunk_for_graph(mock_load, mock_extract):
    # Setup mock return data
    node1 = Node(id="CompanyX", label="ORGANIZATION")
    rel = Relationship(source="CompanyX", target="Tech", type="INDUSTRY")
    mock_data = GraphData(nodes=[node1], relationships=[rel])
    
    mock_extract.return_value = mock_data
    
    # Run function
    process_chunk_for_graph("CompanyX is in the Tech industry.", "chunk_1")
    
    # Check that extract was called with correct text
    mock_extract.assert_called_once_with("CompanyX is in the Tech industry.")
    
    # Check that load was called with correct data and chunk_id
    mock_load.assert_called_once_with(mock_data, "chunk_1")
