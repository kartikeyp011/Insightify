"""
Router for handling challenge logic generation and evaluation.

This module provides endpoints for generating interactive logic questions
based on the uploaded documents and evaluating the user's answers using Gemini.

Components:
    ChallengeResponse: Pydantic model for answer submissions.
    extract_questions_from_text: Helper function to parse raw Gemini output.
    get_challenge_questions: Route to generate challenge questions.
    evaluate_challenge: Route to score and evaluate user answers.

Dependencies:
    - fastapi: For building the router and handling exceptions.
    - re: For regex-based string parsing.
    - pydantic: For request validation.
    - utils.qa_engine: To interact with Gemini for generation and evaluation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.qa_engine import generate_logic_questions, evaluate_user_answers
import re

router = APIRouter()

# ── Models ───────────────────────────────────────────────────────

class ChallengeResponse(BaseModel):
    """
    Pydantic model for the incoming /evaluate endpoint payload.

    Attributes:
        answers (list[str]): The user's submitted answers to the logic questions.
    """
    answers: list[str]

# ── Helper Functions ─────────────────────────────────────────────

def extract_questions_from_text(text: str) -> list[str]:
    """
    Splits Gemini's raw response into a clean list of 3 questions.

    Handles cases where there are extra newlines or spacing. Uses regex
    to look for numbered list patterns standard to language model outputs.

    Args:
        text (str): The full raw response from the Gemini API.

    Returns:
        list[str]: A list of up to 3 question strings.

    Example:
        questions = extract_questions_from_text("1. What is X?\\n2. What is Y?")
        # questions => ["What is X?", "What is Y?"]
    """
    # If Gemini returned a list instead of string (edge case), join it
    if isinstance(text, list):
        text = "\n".join(text)

    # Use regex to split text on numbered patterns like "1. ...", "2. ..."
    raw_questions = re.split(r'\n?\s*\d+\.\s+', text.strip())

    # Remove any empty or whitespace-only entries resulting from the split
    cleaned = [q.strip() for q in raw_questions if q.strip()]

    # Return only the first 3 questions to ensure a consistent experience
    return cleaned[:3]

# ── Endpoints ────────────────────────────────────────────────────

@router.get("/challenge")
async def get_challenge_questions():
    """
    Asynchronously calls Gemini to generate logic-based questions from document context.

    Endpoint: /api/challenge [GET]

    Returns:
        dict: A JSON response containing a list of questions under the "questions" key.

    Raises:
        HTTPException: If unable to generate or parse at least 3 valid questions.
    """
    try:
        # Call Gemini model to generate raw text containing the logic questions
        raw_output = generate_logic_questions()
        print("🧪 RAW GEMINI OUTPUT:\n", raw_output)

        # Clean and split into exactly 3 numbered questions
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
    Asynchronously evaluates user answers against ideal answers using Gemini.

    Endpoint: /api/evaluate [POST]

    Args:
        response (ChallengeResponse): The user's submitted logic answers.

    Returns:
        dict: A structured feedback format containing scores and justifications.

    Raises:
        HTTPException: If the evaluation process fails.
    """
    try:
        # Pass the answers to the QA engine where Gemini evaluates logical correctness
        feedback = evaluate_user_answers(response.answers)
        return {"feedback": feedback}
    except Exception as e:
        print("❌ Error evaluating answers:", e)
        raise HTTPException(status_code=500, detail=f"Error evaluating answers: {str(e)}")