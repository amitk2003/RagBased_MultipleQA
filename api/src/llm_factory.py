import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(provider: str = "gemini", model_name: str = "gemini-1.5-pro", temperature: float = 0) -> BaseChatModel:
    """Factory to get the appropriate LLM."""
    if provider == "gemini":
        # Requires GOOGLE_API_KEY environment variable
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "ollama":
        return ChatOllama(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
