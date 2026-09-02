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
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from utils.qa_engine import generate_logic_questions, evaluate_user_answers
import re
import os

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

    # Use regex to find all matches that look like questions ("1.", "Q1:", "2)", etc.)
    matches = list(re.finditer(r'(?:^|\n)\s*(?:Q?\d+[:\.)])\s+', text))
    if not matches:
        # Fallback if no numbered pattern is found
        raw_questions = [q.strip() for q in text.strip().split('\n') if q.strip()]
        return raw_questions[:3]

    cleaned = []
    for i in range(len(matches)):
        start = matches[i].end()
        if i + 1 < len(matches):
            end = matches[i+1].start()
            q_text = text[start:end].strip()
        else:
            # Last question might have trailing commentary separated by double newline
            rest = text[start:].strip()
            q_text = rest.split('\n\n')[0].strip()
        
        if q_text:
            cleaned.append(q_text)

    return cleaned[:3]

# ── Endpoints ────────────────────────────────────────────────────

@router.get("/challenge")
async def generate_challenge(
    x_gemini_api_key: str = Header(default=None),
    x_groq_api_key: str = Header(default=None),
    x_session_id: str = Header(...)
):
    """
    Asynchronously calls Gemini to generate logic-based questions from document context.

    Endpoint: /api/challenge [GET]

    Returns:
        dict: A JSON response containing a list of questions under the "questions" key.

    Raises:
        HTTPException: If unable to generate or parse at least 3 valid questions.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        text_path = os.path.join(base_dir, "vectorstore", x_session_id, "temp_text.txt")
        
        if not os.path.exists(text_path):
            raise HTTPException(status_code=400, detail="No document found. Please upload a file first.")
            
        with open(text_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Generate 3 logic questions based on the text
        raw_output = generate_logic_questions(full_text, api_key=x_gemini_api_key, groq_api_key=x_groq_api_key)
        print("DEBUG RAW GEMINI OUTPUT:\n", raw_output)

        # Clean and split into exactly 3 numbered questions
        questions = extract_questions_from_text(raw_output)
        print("DEBUG Parsed Questions:", questions)

        if len(questions) < 3:
            raise Exception("Less than 3 valid questions extracted from Gemini output.")

        return {"questions": questions}

    except Exception as e:
        print("ERROR generating questions:", e)
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@router.post("/evaluate")
async def evaluate_answers(
    request: ChallengeResponse,
    x_gemini_api_key: str = Header(default=None),
    x_groq_api_key: str = Header(default=None),
    x_session_id: str = Header(...)
):
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
        feedback = evaluate_user_answers(request.answers, session_id=x_session_id, api_key=x_gemini_api_key, groq_api_key=x_groq_api_key)
        return {"feedback": feedback}
    except Exception as e:
        print("ERROR evaluating answers:", e)
        raise HTTPException(status_code=500, detail=f"Error evaluating answers: {str(e)}")