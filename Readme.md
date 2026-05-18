# 📚 RAG-Based Multi-Document QA System

> A fully local **Retrieval-Augmented Generation (RAG)** pipeline for intelligent question answering over your own PDF documents — no cloud APIs, no subscriptions, no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-4A90D9?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🧠 What Is RAG?

**Retrieval-Augmented Generation** combines the power of a vector search engine with a language model:

1. **Index** — your documents are split into chunks and embedded as vectors.
2. **Retrieve** — when you ask a question, semantically similar chunks are fetched.
3. **Generate** — a local LLM reads the retrieved context and produces a grounded answer.

This eliminates hallucinations and keeps answers traceable to your actual documents.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 📄 **Multi-PDF Upload** | Upload and index multiple PDFs at once via the UI |
| 🔍 **Semantic Search** | `all-MiniLM-L6-v2` sentence-transformer embeddings |
| 🗄️ **ChromaDB Vector Store** | Persistent, local vector database (cosine similarity) |
| 🤖 **Local LLM Generation** | `google/flan-t5-small` runs entirely on your hardware |
| 🔀 **Dual Answer Modes** | `STRICT_RAG` (document-only) or `HYBRID` (fallback to general knowledge) |
| 📌 **Source Citations** | Optionally display retrieved source chunks alongside answers |
| ⚙️ **Centralized Config** | All tuneable parameters in a single `src/config.py` |
| 🖥️ **Streamlit UI** | Clean, interactive browser-based interface |

---

## 🏗️ Project Structure

```
RAG_BASED_QA/
│
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/                    # Core RAG pipeline modules
│   ├── config.py           # ⚙️  All configuration constants
│   ├── extract.py          # PDF text extraction & chunking
│   ├── embed.py            # Sentence-transformer embedding (singleton)
│   ├── index.py            # ChromaDB vector store (build + retrieve)
│   ├── generate.py         # LLM answer generation (flan-t5)
│   ├── main.py             # CLI entry point (optional)
│   └── vector_store.py     # (Legacy stub)
│
├── data/
│   ├── raw/                # 📥 Drop your source PDFs here
│   └── processed/          # Intermediate processed files
│
├── notebooks/              # Jupyter exploration notebooks
└── .gitignore
```

---

## ⚙️ Configuration Reference

All pipeline behaviour is controlled by **`src/config.py`**. Edit this file to tune the system without touching any other code.

```python
# Answer mode — HYBRID or STRICT_RAG
MODE = "HYBRID"

# Chunking
CHUNK_SIZE    = 400   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# Retrieval
TOP_K = 4             # top-k chunks to retrieve per query

# Models
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"     # HuggingFace sentence-transformer
LLM_MODEL           = "google/flan-t5-small"  # Local generative model
MAX_RESPONSE_LENGTH = 300

# Output format — ANSWER_ONLY or WITH_SOURCES
ANSWER_MODE = "ANSWER_ONLY"

# Vector store
COLLECTION_NAME = "rag_documents"
CHROMA_DB_PATH  = "chroma_db"
```

### Answer Modes

| Mode | Behaviour |
|---|---|
| `STRICT_RAG` | Only answers from document context; returns *"I don't know"* if context is absent |
| `HYBRID` | Uses document context first; falls back to general knowledge with a clear disclaimer |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amitk2003/RagBased_MultipleQA.git
cd RagBased_MultipleQA
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv env
.\env\Scripts\Activate.ps1

# macOS / Linux
python -m venv env
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the embedding model (`~90 MB`) and the LLM (`~300 MB`) from HuggingFace. Subsequent runs use the cached models.

### 4. Launch the App

```bash
streamlit run app.py
```

The app opens automatically at **`http://localhost:8501`**.

---

## 🖥️ Using the Application

```
┌─────────────────────────────────────────────────────┐
│  Sidebar                   │  Main Panel            │
│  ─────────────             │  ──────────────        │
│  ⚙️ Select Answer Mode      │  2️⃣ Enter Question     │
│  1️⃣ Upload PDF(s)          │                        │
│  📥 Process & Index         │  🔍 Generate Answer    │
│                             │  📄 View Retrieved     │
│                             │     Context (expander) │
└─────────────────────────────────────────────────────┘
```

**Step-by-step:**

1. Open the sidebar → choose **Answer Mode** (`HYBRID` recommended).
2. Upload one or more **PDF files**.
3. Click **"📥 Process & Index Documents"** — chunks are embedded and stored in ChromaDB.
4. Type your question in the main panel and click **"🔍 Generate Answer"**.
5. Expand **"📄 View Retrieved Context"** to see exactly which passages were used.

---

## 🔬 Pipeline Deep Dive

```
PDF(s)
  │
  ▼
[extract.py]  ──── PyPDF2 extraction → LangChain RecursiveCharacterTextSplitter
  │                (chunk_size=400, overlap=50)
  ▼
[embed.py]    ──── SentenceTransformer("all-MiniLM-L6-v2") → float vectors
  │
  ▼
[index.py]    ──── ChromaDB PersistentClient (cosine space) → stored collection
  │
  ▼  (at query time)
[embed.py]    ──── embed_query() → 1-D query vector
  │
  ▼
[index.py]    ──── collection.query(top_k=4) → retrieved chunks + IDs
  │
  ▼
[generate.py] ──── flan-t5-small text2text pipeline → final answer string
  │
  ▼
[app.py]      ──── st.success(answer) displayed in Streamlit
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | latest | Web UI |
| `PyPDF2` | 3.0.1 | PDF text extraction |
| `sentence-transformers` | 3.0.1 | Dense embeddings |
| `transformers` | 4.44.2 | LLM pipeline (flan-t5) |
| `torch` | 2.4.1 | Model inference backend |
| `chromadb` | latest | Local vector store |
| `langchain-text-splitters` | latest | Intelligent text chunking |

---

## 🔄 Upgrade History

| Version | Changes |
|---|---|
| **v1.0** | Initial POC — FAISS + basic single-PDF support |
| **v2.0** | Migrated to Qdrant vector store, modular `src/` layout |
| **v3.0 (current)** | Migrated vector store to **ChromaDB** (persistent, no server needed), multi-PDF upload, HYBRID/STRICT answer modes, lazy-singleton model loading, improved Streamlit UI |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| *"No text could be extracted"* | Ensure the PDF is not scanned/image-only. Use OCR tools first. |
| *"No relevant context found"* | Re-index your documents. The ChromaDB collection may have been reset. |
| Slow first load | Model download is one-time only; subsequent runs use the local cache. |
| `ImportError` for torch/transformers | Run `pip install -r requirements.txt` inside the activated virtual environment. |
| Port already in use | Run `streamlit run app.py --server.port 8502` to use an alternate port. |

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
  <sub>Built with ❤️ using Python, Streamlit, ChromaDB, and HuggingFace Transformers</sub>
</div>
