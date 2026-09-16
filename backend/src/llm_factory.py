import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from src.config import LLM_PROVIDER,LLM_MODEL
def get_llm(provider= LLM_PROVIDER, model_name = LLM_MODEL, temperature: float = 0) -> BaseChatModel:
    """Factory to get the appropriate LLM."""
  
    if provider == "groq":
        # Requires GROQ_API_KEY environment variable
        return ChatGroq(model=model_name, temperature=temperature)
   
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
