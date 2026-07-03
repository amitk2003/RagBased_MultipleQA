from src.llm_factory import get_llm
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
# ConversationalRetrievalChain not used directly; chain built manually below
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
import os

POSTGRES_CONNECTION = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://user:password@postgres:5432/memorydb")

def get_memory_for_session(session_id: str) -> ConversationBufferMemory:
    message_history = PostgresChatMessageHistory(
        connection_string=POSTGRES_CONNECTION,
        session_id=session_id,
        table_name="chat_history"
    )
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True
    )

class AnswerEval(BaseModel):
    confidence_score: float = Field(description="Score between 0.0 and 1.0 indicating confidence in the answer based ONLY on the context.")
    is_hallucination: bool = Field(description="True if the answer contains information not found in the context.")

def evaluate_answer(query: str, answer: str, context: str) -> AnswerEval:
    """Self-reflection step to compute confidence and detect hallucination."""
    llm = get_llm(provider="gemini", model_name="gemini-1.5-pro", temperature=0)
    structured_llm = llm.with_structured_output(AnswerEval)
    
    prompt = f"""
    Given the following user query, retrieved context, and generated answer:
    
    Query: {query}
    Context: {context}
    Answer: {answer}
    
    Evaluate the generated answer. Provide a confidence score (0.0 to 1.0) and flag if it hallucinates information not present in the context.
    """
    
    try:
        return structured_llm.invoke(prompt)
    except Exception as e:
        print(f"Error during self-reflection: {e}")
        return AnswerEval(confidence_score=0.5, is_hallucination=False)

def generate_answer(query: str, retrieved_chunks: List[Dict], session_id: str) -> Tuple[str, float, bool]:
    llm = get_llm()
    memory = get_memory_for_session(session_id)
    
    # Construct context string
    context_str = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template="""You are an expert Q&A assistant. Use the following retrieved context and chat history to answer the question.
If you don't know the answer, just say that you don't know. Do not hallucinate.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""
    )
    
    # Simple generation using LCEL
    chain = prompt_template | llm
    
    # Get chat history string
    history_msgs = memory.load_memory_variables({})['chat_history']
    history_str = "\\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])
    
    response = chain.invoke({
        "chat_history": history_str,
        "context": context_str,
        "question": query
    })
    
    answer_text = response.content
    
    # Save to memory
    memory.save_context({"input": query}, {"output": answer_text})
    
    # Evaluate
    eval_result = evaluate_answer(query, answer_text, context_str)
    
    return answer_text, eval_result.confidence_score, eval_result.is_hallucination
