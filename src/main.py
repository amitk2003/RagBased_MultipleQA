# src/main.py  — CLI entry-point for the RAG QA pipeline
import os
import sys

# Make sure src/ is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from extract import extract_text_from_pdf, chunk_text
from embed import generate_embeddings, embed_query
from index import build_chroma_index, retrieve_chunks
from generate import generate_answer

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "raw",
)


def main():
    print("=== RAG Document QA System ===\n")

    os.makedirs(DATA_PATH, exist_ok=True)

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF found in data/raw/. Please add one and re-run.")
        return

    # ── Step 1: Extract & chunk ──────────────────────────────────────────────
    all_text = ""
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        print(f"Loading: {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            all_text += text + "\n\n"

    if not all_text.strip():
        print("No text could be extracted from the PDFs.")
        return

    chunks = chunk_text(all_text)
    print(f"Extracted {len(chunks)} chunks.\n")

    # ── Step 2: Embed ────────────────────────────────────────────────────────
    print("Generating embeddings…")
    embeddings = generate_embeddings(chunks)

    # ── Step 3: Index ────────────────────────────────────────────────────────
    collection_name = build_chroma_index(embeddings, chunks)
    print("Vector index ready.\n")

    # ── Step 4: Interactive QA loop ──────────────────────────────────────────
    print("Ask questions about the documents (type 'quit' to exit):\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue

        query_emb = embed_query(query)
        retrieved_chunks, chunk_ids = retrieve_chunks(collection_name, query_emb)

        print("\nGenerating answer…\n")
        answer = generate_answer(query, retrieved_chunks, chunk_ids)
        print(f"Answer: {answer}\n")
        print("-" * 80)


if __name__ == "__main__":
    main()
