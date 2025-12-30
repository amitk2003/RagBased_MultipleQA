# src/main.py
import os
from extract import extract_text_from_pdf, chunk_text
from embed import generate_embeddings, embed_query
from index import build_faiss_index, retrieve_chunks
from generate import generate_answer

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "raw"
)

def main():
    print("=== RAG Document QA System POC ===\n")

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF found in data/raw/. Please add one.")
        return

    pdf_path = os.path.join(DATA_PATH, pdf_files[0])
    print(f"Loading document: {pdf_files[0]}")

    # Step 1: Extract & chunk
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text extracted from PDF.")
        return

    chunks = chunk_text(text)
    print(f"Extracted {len(chunks)} chunks.\n")

    # Step 2: Embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    # Step 3: Build vector index (Chroma backend)
    index = build_faiss_index(embeddings, chunks)
    print("Vector index built (local backend).\n")

    print("Ask questions about the document (type 'quit' to exit):\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue

        # Step 4: Retrieve
        query_emb = embed_query(query)
        retrieved_chunks, chunk_ids = retrieve_chunks(
            index, query_emb, chunks
        )

        # Step 5: Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(query, retrieved_chunks, chunk_ids)
        print(f"Answer: {answer}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
