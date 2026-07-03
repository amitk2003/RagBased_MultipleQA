# 📚 Advanced RAG-Based Multi-Document QA System

> An enterprise-grade, microservices-based **Retrieval-Augmented Generation (RAG)** pipeline for intelligent question answering over your documents. 
> Featuring Hybrid Retrieval, Graph RAG, and Conversational Memory!

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white)

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🐳 **Dockerized Microservices** | Easy deployment using Docker Compose (UI, API, Graph DB, SQL DB) |
| 📄 **Multi-PDF Upload** | Upload and index multiple PDFs via the UI with background processing |
| 🔍 **Hybrid Retrieval** | Combines ChromaDB vector search with BM25 keyword reranking |
| 🕸️ **Graph RAG** | Uses Neo4j to extract and visualize entities and relationships |
| 🧠 **Conversational Memory** | Powered by PostgreSQL to maintain chat history context |
| 🤖 **Gemini LLM Integration** | Uses Google Gemini for high-quality, grounded answer generation |
| 📊 **Built-in Evaluation** | Uses `ragas` to evaluate hallucination and generation quality |
| 🖥️ **Streamlit UI** | Clean, interactive browser-based chat interface |

---

## 🏗️ Architecture Stack

* **Frontend:** Streamlit (`ui/`)
* **Backend:** FastAPI (`api/`)
* **Vector Store:** ChromaDB
* **Graph Database:** Neo4j (Entity Extraction & Graph RAG)
* **Memory Database:** PostgreSQL
* **LLM / Embedding:** Google Gemini (`langchain-google-genai`), Sentence Transformers (`all-MiniLM-L6-v2`)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amitk2003/RagBased_MultipleQA.git
cd RagBased_MultipleQA
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys.

```env
# ── Google Gemini (Required) ─────────────────────────────────────────────────
GOOGLE_API_KEY=your_google_api_key_here

# ── Neo4j (matches docker-compose.yml defaults) ──────────────────────────────
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# ── PostgreSQL Memory ────────────────────────────────────────────────────────
POSTGRES_CONNECTION_STRING=postgresql://raguser:ragpassword@postgres:5432/memorydb
```

### 3. Launch with Docker Compose

Ensure you have [Docker](https://www.docker.com/) installed and running.

```bash
docker-compose up --build
```

### 4. Access the Services

* **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Backend (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Neo4j Browser:** [http://localhost:7474](http://localhost:7474) *(Login: neo4j / password)*

---

## 🖥️ Using the Application

1. Open the UI at `http://localhost:8501`.
2. **Upload PDF(s)** from the sidebar. Wait for the extraction and indexing to complete (Graph RAG entities will be extracted in the background).
3. Type your question in the main chat panel.
4. The system will retrieve context via **Hybrid Search (ChromaDB + BM25)**, augment it with Graph data if applicable, and generate an answer using **Google Gemini**.
5. You can view the retrieved sources and confidence scores alongside your answers!

---

## 🔬 Pipeline Deep Dive

1. **Ingestion:** PDFs are parsed via `pdfplumber` and `pytesseract` (OCR).
2. **Chunking & Indexing:** Text is split and stored in ChromaDB (dense vectors) and BM25 (sparse keyword index).
3. **Graph Extraction (Background):** Chunks are processed to extract entities/relationships and loaded into Neo4j.
4. **Retrieval:** A user query triggers hybrid retrieval. The best chunks are reranked and provided as context.
5. **Generation:** LangChain queries Google Gemini with the retrieved context and conversation history to produce a grounded response.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-enhancement`
3. Commit your changes: `git commit -m "feat: add my enhancement"`
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ using Python, FastAPI, Streamlit, Neo4j, Docker and Google Gemini</sub>
</div>
