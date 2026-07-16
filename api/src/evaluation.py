import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict

# Configure Ragas to use Groq LLM + local HuggingFace embeddings (avoids needing OpenAI or Google keys)
_ragas_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.1-8b-instant", temperature=0)
)
_ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# In-memory store for the current server session
# Postgres persistence for long-term storage can be added here
_evaluations: List[Dict] = []

def record_for_evaluation(question: str, answer: str, contexts: List[str], ground_truth: str = ""):
    """Stores QA pair for later Ragas evaluation."""
    _evaluations.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth
    })

def run_evaluation() -> Dict:
    """Runs Ragas evaluation on stored history using Groq LLM."""
    if not _evaluations:
        return {"status": "No data to evaluate. Ask some questions first."}

    df = pd.DataFrame(_evaluations)
    dataset = Dataset.from_pandas(df)

    try:
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=_ragas_llm,
            embeddings=_ragas_embeddings,
        )
        # Convert to plain dict so FastAPI can serialise it
        return result.to_pandas().mean(numeric_only=True).to_dict()
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        return {"error": str(e)}
