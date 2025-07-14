from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.utils.qa_engine import generate_logic_questions, evaluate_user_answers
import re

router = APIRouter()

class ChallengeResponse(BaseModel):
    answers: list[str]

def extract_questions_from_text(text: str) -> list[str]:
    """
    Splits Gemini's raw response into a clean list of 3 questions.
    Handles cases where there are extra newlines or spacing.

    Parameters:
        text (str): The full response from Gemini

    Returns:
        list[str]: List of up to 3 question strings
    """
    # If Gemini returned a list instead of string (edge case), join it
    if isinstance(text, list):
        text = "\n".join(text)

    # Use regex to split text on numbered patterns like "1. ...", "2. ..."
    raw_questions = re.split(r'\n?\s*\d+\.\s+', text.strip())

    # Remove any empty or whitespace-only entries
    cleaned = [q.strip() for q in raw_questions if q.strip()]

    # Return only the first 3 questions (if available)
    return cleaned[:3]

@router.get("/challenge")
async def get_challenge_questions():
    """
    Endpoint: /api/challenge [GET]

    - Calls Gemini to generate 3 logic-based questions from uploaded document
    - Parses and returns questions in JSON format
    """
    try:
        # 🧠 Call Gemini model to generate raw text
        raw_output = generate_logic_questions()
        print("🧪 RAW GEMINI OUTPUT:\n", raw_output)

        # ✂️ Clean and split into 3 numbered questions
        questions = extract_questions_from_text(raw_output)
        print("✅ Parsed Questions:", questions)

        if len(questions) < 3:
            raise Exception("Less than 3 valid questions extracted from Gemini output.")

        return {"questions": questions}

    except Exception as e:
        print("❌ Error generating questions:", e)
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@router.post("/evaluate")
async def evaluate_challenge(response: ChallengeResponse):
    """
    Endpoint: /api/evaluate [POST]

    - Accepts user answers from frontend
    - Compares with ideal answers using Gemini
    - Returns score + justification for each answer
    """
    try:
        feedback = evaluate_user_answers(response.answers)
        return {"feedback": feedback}
    except Exception as e:
        print("❌ Error evaluating answers:", e)
        raise HTTPException(status_code=500, detail=f"Error evaluating answers: {str(e)}")