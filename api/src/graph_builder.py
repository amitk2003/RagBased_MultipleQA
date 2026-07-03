from typing import List, Dict
from src.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dependencies import get_neo4j_driver

# Define extraction schema
class Node(BaseModel):
    id: str = Field(description="Unique identifier for the entity.")
    label: str = Field(description="Type of the entity (e.g., PERSON, ORGANIZATION, CONCEPT).")

class Relationship(BaseModel):
    source: str = Field(description="ID of the source node.")
    target: str = Field(description="ID of the target node.")
    type: str = Field(description="Type of the relationship (e.g., FOUNDED, BASED_IN, RELATES_TO).")
    
class GraphData(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

def extract_graph_data(text: str) -> GraphData:
    """Uses Gemini LLM to extract nodes and relationships from text."""
    llm = get_llm(provider="gemini", model_name="gemini-1.5-pro", temperature=0)
    structured_llm = llm.with_structured_output(GraphData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting knowledge graphs from text. Extract entities (nodes) and their relationships."),
        ("human", "Extract graph data from the following text:\n\n{text}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"text": text})
    return result

def load_graph_to_neo4j(graph_data: GraphData, chunk_id: str):
    """Loads extracted nodes and relationships into Neo4j."""
    driver = get_neo4j_driver()
    if not driver:
        print("Neo4j driver not initialized.")
        return
        
    def _create_graph_tx(tx, nodes, relationships, chunk_id):
        # Create chunk node
        tx.run("MERGE (c:Chunk {id: $chunk_id})", chunk_id=chunk_id)
        
        # Create entity nodes
        for node in nodes:
            tx.run(
                "MERGE (n:Entity {id: $id}) "
                "ON CREATE SET n.label = $label "
                "MERGE (n)-[:MENTIONED_IN]->(c)",
                id=node.id, label=node.label, chunk_id=chunk_id
            )
            
        # Create relationships
        for rel in relationships:
            tx.run(
                "MATCH (source:Entity {id: $source}) "
                "MATCH (target:Entity {id: $target}) "
                "MERGE (source)-[r:RELATION {type: $type}]->(target)",
                source=rel.source, target=rel.target, type=rel.type
            )
            
    with driver.session() as session:
        session.execute_write(_create_graph_tx, graph_data.nodes, graph_data.relationships, chunk_id)

def process_chunk_for_graph(text: str, chunk_id: str):
    """Orchestrates extraction and loading for a single chunk."""
    try:
        graph_data = extract_graph_data(text)
        if graph_data:
            load_graph_to_neo4j(graph_data, chunk_id)
    except Exception as e:
        print(f"Failed to process chunk {chunk_id} for graph: {e}")
