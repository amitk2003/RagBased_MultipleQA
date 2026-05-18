# src/generate.py
import warnings

from transformers import pipeline

# All config imports at the top — never mid-file
try:
    from config import LLM_MODEL, MAX_RESPONSE_LENGTH, MODE, ANSWER_MODE
except ModuleNotFoundError:
    from src.config import LLM_MODEL, MAX_RESPONSE_LENGTH, MODE, ANSWER_MODE

# Suppress FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Lazy singleton for the generative pipeline (avoids re-loading on each call)
# ---------------------------------------------------------------------------
_generator = None


def _get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=LLM_MODEL,
            tokenizer_kwargs={"clean_up_tokenization_spaces": True},
        )
    return _generator


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, retrieved_chunks: list, chunk_ids: list) -> str:
    """
    Hybrid Answer Generator:
      STRICT_RAG → Only document-based answers.
      HYBRID     → Document if relevant, else general knowledge.
    """
    generator = _get_generator()

    # ── Case 1: No useful document context ──────────────────────────────────
    if not retrieved_chunks or all(len(c.strip()) < 30 for c in retrieved_chunks):

        if MODE == "STRICT_RAG":
            return "I don't know based on the document."

        # HYBRID MODE → fall back to general knowledge
        prompt = (
            "You are a knowledgeable assistant.\n"
            "The uploaded document does NOT contain the answer.\n"
            "Answer the question using general knowledge.\n"
            "Clearly mention that this answer is NOT from the document.\n\n"
            f"Question: {query}\n\n"
            "Answer (start with: 'Based on general knowledge:'):"
        )

        result = generator(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
        return result[0]["generated_text"].split("Answer:", 1)[-1].strip()

    # ── Case 2: Document-based answer (RAG) ─────────────────────────────────
    context = "\n\n".join(
        f"[Source {i + 1}]: {chunk.strip()}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = (
        "You are a strict document-based QA assistant.\n\n"
        "Rules:\n"
        "- Use ONLY the information present in the context below.\n"
        "- Do NOT use outside knowledge.\n"
        "- If the answer is not present, say: 'I don't know based on the document.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
    )

    answer = result[0]["generated_text"].split("Answer:", 1)[-1].strip()
    return _format_answer(answer, retrieved_chunks)


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _format_answer(answer: str, retrieved_chunks: list) -> str:
    answer = answer.strip()

    if ANSWER_MODE == "ANSWER_ONLY":
        # Return clean single-paragraph answer
        return " ".join(answer.split("\n")).strip()

    # WITH_SOURCES → include citations
    formatted = "### 📌 Answer\n\n"
    formatted += f"{answer}\n\n### 📚 Sources\n"
    for i, chunk in enumerate(retrieved_chunks):
        preview = chunk.replace("\n", " ")[:100]
        formatted += f"- Source {i + 1}: {preview}...\n"

    return formatted
