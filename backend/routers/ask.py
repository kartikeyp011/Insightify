from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.retriever import get_relevant_chunks
from backend.utils.qa_engine import generate_answer

router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(payload: AskRequest):
    """
    Takes a question from user and returns Gemini answer with justification.
    """
    try:
        chunks = get_relevant_chunks(payload.question)
        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        answer = generate_answer(question=payload.question, context_chunks=chunks)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")