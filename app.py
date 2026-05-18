# app.py  — Streamlit front-end for the RAG Document QA system
import os
import sys
import streamlit as st

# Ensure src/ is on the path so modules import correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.extract import extract_text_from_pdf, chunk_text
from src.embed import generate_embeddings, embed_query
from src.index import build_chroma_index, retrieve_chunks
from src.generate import generate_answer
import src.config as config

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Document QA", page_icon="📚", layout="wide")

st.title("📚 RAG Document QA System")
st.markdown("Upload PDFs, index them, then ask questions!")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(DATA_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration & document upload
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
mode_select = st.sidebar.selectbox(
    "Answer mode",
    ["HYBRID", "STRICT_RAG"],
    index=0 if config.MODE == "HYBRID" else 1,
)
config.MODE = mode_select

st.sidebar.markdown("---")
st.sidebar.header("1️⃣  Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if st.sidebar.button("📥 Process & Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing documents…"):
            all_text = ""
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                text = extract_text_from_pdf(file_path)
                if text:
                    all_text += text + "\n\n"

            if not all_text.strip():
                st.sidebar.error("No text could be extracted from the uploaded PDFs.")
            else:
                chunks = chunk_text(all_text)
                st.sidebar.info(f"Extracted **{len(chunks)}** chunks.")

                embeddings = generate_embeddings(chunks)
                collection_name = build_chroma_index(embeddings, chunks)

                # Persist the collection name across Streamlit reruns
                st.session_state["collection_name"] = collection_name
                st.session_state["index_ready"] = True
                st.sidebar.success("✅ Vector index built successfully!")

# ─────────────────────────────────────────────────────────────────────────────
# Main panel — QA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("2️⃣  Ask a Question")

# Warn the user clearly if they haven't indexed yet
if not st.session_state.get("index_ready", False):
    st.info(
        "ℹ️  No documents have been indexed yet. "
        "Please upload one or more PDFs and click **Process & Index Documents** first."
    )

query = st.text_input("Enter your question:", placeholder="e.g. What is the main topic of the document?")

if st.button("🔍 Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.get("index_ready", False):
        st.error("❌ No index found. Upload and index documents first.")
    else:
        with st.spinner("Searching for answer…"):
            # embed_query now returns a correct 1-D flat list
            query_emb = embed_query(query)
            collection_name = st.session_state.get("collection_name", config.COLLECTION_NAME)

            retrieved_chunks, chunk_ids = retrieve_chunks(collection_name, query_emb)

            if not retrieved_chunks:
                st.warning("⚠️  No relevant context found. The index might be empty.")
            else:
                answer = generate_answer(query, retrieved_chunks, chunk_ids)
                st.markdown("### 💬 Answer")
                st.success(answer)

                with st.expander("📄 View Retrieved Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Source {i + 1}:**")
                        st.text(chunk)
