import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
def get_llm(provider: str = "groq", model_name: str = "llama-3.1-8b-instant", temperature: float = 0) -> BaseChatModel:
    """Factory to get the appropriate LLM."""
  
    if provider == "groq":
        # Requires GROQ_API_KEY environment variable
        return ChatGroq(model=model_name, temperature=temperature)
   
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
