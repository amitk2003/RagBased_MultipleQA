from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from src.extract import extract_and_chunk_pdf
from src.graph_builder import process_chunk_for_graph
from src.index import build_indices
from src.retrieval import hybrid_retrieve_and_rerank
from src.generate import generate_answer
from src.evaluation import record_for_evaluation, run_evaluation
from dependencies import get_neo4j_driver, get_chroma_client

app = FastAPI(title="Advanced RAG System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str
    
class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float
    hallucination_flag: bool

@app.get("/")
def read_root():
    return {"message": "Welcome to the Advanced RAG API"}

def _build_graph_in_background(chunks: list):
    """Runs entity extraction + Neo4j loading in a background task."""
    for chunk in chunks:
        chunk_id = (
            f"chunk_{chunk['metadata'].get('source')}"
            f"_{chunk['metadata'].get('chunk_index')}"
        )
        try:
            process_chunk_for_graph(chunk["text"], chunk_id)
        except Exception as e:
            print(f"[graph] Failed for {chunk_id}: {e}")

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Full ingestion pipeline:
      Save PDF → Extract text (pdfplumber + pytesseract OCR)
               → Chunk → Embed → ChromaDB upsert → BM25 update
               → (background) Graph entity extraction → Neo4j
    """
    saved_files = []

    # ── Step 1: Save uploaded files ──────────────────────────────────────────
    for file in files:
        file_location = f"data/raw/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        try:
            contents = await file.read()
            with open(file_location, "wb") as f:
                f.write(contents)
            saved_files.append(file_location)
            print(f"[upload] Saved: {file_location}")
        except Exception as e:
            print(f"[upload] Failed to save {file.filename}: {e}")

    if not saved_files:
        return {"message": "No files could be saved.", "chunks_processed": 0}

    # ── Step 2: Extract text + chunk ─────────────────────────────────────────
    all_chunks = []
    for filepath in saved_files:
        try:
            print(f"[upload] Extracting: {filepath}")
            chunks = extract_and_chunk_pdf(filepath)
            print(f"[upload] Got {len(chunks)} chunks from {filepath}")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[upload] Extraction failed for {filepath}: {e}")

    if not all_chunks:
        return {"message": "No text could be extracted from the uploaded files.", "chunks_processed": 0}

    # ── Step 3: Embed + Index (ChromaDB + BM25) ───────────────────────────────
    try:
        print(f"[upload] Indexing {len(all_chunks)} chunks…")
        build_indices(all_chunks)
        print(f"[upload] Indexing complete.")
    except Exception as e:
        print(f"[upload] Indexing failed: {e}")
        return {"message": f"Indexing failed: {str(e)}", "chunks_processed": 0}

    # ── Step 4: Graph extraction (non-blocking background task) ───────────────
    background_tasks.add_task(_build_graph_in_background, all_chunks)

    return {
        "message": "Files uploaded and indexed successfully. Graph extraction running in background.",
        "chunks_processed": len(all_chunks),
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 1. Hybrid Retrieval & Reranking
    retrieved_chunks = hybrid_retrieve_and_rerank(request.query)

    # Guard: if nothing was retrieved, return a helpful message immediately
    if not retrieved_chunks:
        return ChatResponse(
            answer="⚠️ No documents have been indexed yet. Please upload a PDF first using the sidebar.",
            sources=[],
            confidence_score=0.0,
            hallucination_flag=False,
        )
    
    # 2. Generation & Confidence Scoring
    answer, confidence, is_hallucinated = generate_answer(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        session_id=request.session_id
    )
    
    # 3. Record for Evaluation
    contexts = [c['text'] for c in retrieved_chunks]
    record_for_evaluation(request.query, answer, contexts)
    
    sources = [f"{c['metadata'].get('source', 'Unknown')} (Page {c['metadata'].get('page', '?')})" for c in retrieved_chunks]
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence_score=confidence,
        hallucination_flag=is_hallucinated
    )

@app.get("/evaluate")
def evaluate_endpoint():
    result = run_evaluation()
    return {"metrics": result}

@app.get("/graph")
def graph_endpoint():
    """Returns entities and relationships from Neo4j for visualisation."""
    from dependencies import get_neo4j_driver
    driver = get_neo4j_driver()
    if not driver:
        return {"nodes": [], "edges": [], "error": "Neo4j not connected"}
    try:
        with driver.session() as session:
            nodes_result = session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.label AS label LIMIT 200"
            )
            nodes = [{"id": r["id"], "label": r["label"]} for r in nodes_result]

            edges_result = session.run(
                "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
                "RETURN a.id AS source, b.id AS target, r.type AS type LIMIT 300"
            )
            edges = [{"source": r["source"], "target": r["target"], "type": r["type"]} for r in edges_result]

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
