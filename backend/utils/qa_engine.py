import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import pickle

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates a grounded answer using Gemini with justification.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert research assistant.
Use the below document content to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer based strictly on the provided context.
- Include a short justification (e.g. "As mentioned in paragraph 2..." or "Based on section 3...").
- Do not make up information.

Answer:
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

CHUNKS_PATH = "backend/vectorstore/chunk_texts.pkl"

def load_context() -> str:
    """
    Loads all document chunks as a single string context.
    """
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("No uploaded document found.")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return "\n\n".join(chunks[:20])  # limit to ~20 chunks for context

def generate_logic_questions() -> str:
    """
    Calls Gemini to generate 3 logic-based questions.
    Returns raw output (unparsed).
    """
    context = load_context()

    prompt = f"""
You are an AI assistant generating challenging logic-based questions from a document.

Instructions:
- Generate 3 logic/reasoning-based questions from the document.
- Ensure each question is standalone and tests comprehension and inference.
- Number them clearly: 1. ..., 2. ..., 3. ...
- Output format:
1. ...
2. ...
3. ...

Document:
\"\"\"
{context}
\"\"\"
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)

    print("\n🔍 [DEBUG] Gemini Raw Output:\n", response.text)  # Add this
    return response.text.strip()

import json
import re
import google.generativeai as genai
from dotenv import load_dotenv
import os
from utils.qa_engine import load_context  # Make sure load_context is already defined

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

def evaluate_user_answers(user_answers: list[str]) -> list[dict]:
    """
    Uses Gemini to evaluate user-submitted answers against ideal ones,
    and returns score + justification + feedback for each.
    """

    # ✅ Load the full document context (joined chunks)
    context = load_context()

    # ✅ Construct the evaluation prompt
    prompt = f"""
You are an expert tutor evaluating student answers based on a document.

Context:
{context}

User Answers:
{user_answers}

Instructions:
- For each answer, generate the ideal answer.
- Score the user answer from 1 to 5.
- Provide justification like "This is partially correct because...".

Return ONLY a valid JSON list in the format:
[
  {{
    "question": "What is X?",
    "ideal_answer": "X is ...",
    "user_answer": "User's input",
    "score": 4,
    "feedback": "Well explained but missed detail Y."
  }},
  ...
]
    """

    # ✅ Call Gemini 2.5 Pro model
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    raw_output = response.text.strip()

    # ✅ Remove markdown code block fencing (```json ... ```)
    cleaned_output = re.sub(r"```json|```", "", raw_output).strip()

    # ✅ Try to parse the cleaned string as JSON
    try:
        return json.loads(cleaned_output)
    except Exception as e:
        return [{
            "error": "Failed to parse Gemini response",
            "exception": str(e),
            "raw_output": cleaned_output
        }]