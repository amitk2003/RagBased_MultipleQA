# RAG Document QA System POC

A simple Retrieval-Augmented Generation system for question answering over PDF documents.

## Features
- PDF text extraction & smart chunking
- Semantic search with sentence-transformers + FAISS
- Grounded responses with source citations
- Reduces LLM hallucinations

## Setup
1. Place your PDF(s) in `data/raw/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt# RagBased_MultipleQA
