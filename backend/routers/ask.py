"""
Router for handling user questions about processed documents.

This module provides the API endpoint to accept user queries, retrieve relevant document
chunks using semantic search, and generate an answer using the Gemini LLM.

Components:
    AskRequest: Pydantic model for the incoming question payload.
    ask_question: The FastAPI route handler for answering questions.

Dependencies:
    - fastapi: For building the API router and handling HTTP exceptions.
    - pydantic: For request payload validation.
    - utils.retriever: To fetch relevant document context.
    - utils.qa_engine: To generate answers based on context.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.retriever import get_relevant_chunks
from utils.qa_engine import generate_answer

router = APIRouter()

# ── Models ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """
    Pydantic model representing the expected payload for the /ask endpoint.

    Attributes:
        question (str): The user's question regarding the uploaded documents.
    """
    question: str

# ── Endpoints ────────────────────────────────────────────────────

@router.post("/ask")
async def ask_question(payload: AskRequest):
    """
    Asynchronously takes a question from the user and returns a generated answer.

    This function retrieves relevant chunks from the FAISS vector database
    and uses the QA engine to formulate an answer with justification.

    Args:
        payload (AskRequest): The incoming request payload containing the question.

    Returns:
        dict: A dictionary containing the generated answer.

    Raises:
        HTTPException: If no context is found (404) or on internal generation error (500).

    Example:
        response = await ask_question(AskRequest(question="What is the main topic?"))
        # response => {"answer": "The main topic is ..."}
    """
    try:
        # Retrieve top relevant text chunks to provide context for the answer
        chunks = get_relevant_chunks(payload.question)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        # Generate response using the retrieved context
        answer = generate_answer(question=payload.question, context_chunks=chunks)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")