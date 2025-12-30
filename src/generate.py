from transformers import pipeline
from config import LLM_MODEL, MAX_RESPONSE_LENGTH, MODE

generator = pipeline("text2text-generation", model=LLM_MODEL)

def generate_answer(query: str, retrieved_chunks: list[str], chunk_ids: list[int]) -> str:
    """
    Hybrid Answer Generator:
    - STRICT_RAG → Only document-based
    - HYBRID → Document if relevant, else general knowledge
    """

    # ---------- Case 1: No useful document context ----------
    if not retrieved_chunks or all(len(chunk.strip()) < 30 for chunk in retrieved_chunks):

        if MODE == "STRICT_RAG":
            return "I don't know based on the document."

        # HYBRID MODE → General knowledge
        prompt = f"""
You are a knowledgeable assistant.

The uploaded document does NOT contain the answer.
Answer the question using general knowledge.
Clearly mention that this answer is NOT from the document.

Question: {query}

Answer (start with: "Based on general knowledge:"):
"""
        result = generator(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3, #important

            temperature=0.2,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id
        )

        return result[0]["generated_text"].split("Answer:", 1)[-1].strip()

    # ---------- Case 2: Document-based answer (RAG) ----------
    context = "\n\n".join(
        f"[Source {i+1}]: {chunk.strip()}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""
You are a strict document-based QA assistant.

Rules:
- Use ONLY the information present in the context.
- Do NOT use outside knowledge.
- If the answer is not explicitly present, say:
  "I don't know based on the document."

Context:
{context}

Question: {query}

Answer:
"""

    result = generator(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3, #important

            temperature=0.2,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id
        )

    answer = result[0]["generated_text"].split("Answer:", 1)[-1].strip()

    return format_answer(answer, retrieved_chunks)
from config import ANSWER_MODE

def format_answer(answer: str, retrieved_chunks: list[str]) -> str:
    answer = answer.strip()

    if ANSWER_MODE == "ANSWER_ONLY":
        # Return clean single-paragraph answer
        lines = answer.split("\n")
        return " ".join(lines).strip()

    # Else → include citations
    formatted = "### 📌 Answer\n\n"
    formatted += f"{answer}\n\n### 📚 Sources\n"

    for i, chunk in enumerate(retrieved_chunks):
        preview = chunk.replace("\n", " ")[:100]
        formatted += f"- Source {i+1}: {preview}...\n"

    return formatted
