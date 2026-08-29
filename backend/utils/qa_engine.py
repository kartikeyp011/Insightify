"""
QA generation and evaluation engine with multi-mode provider support.

This module houses the core generative components of the application. It provides
functions to generate grounded answers to user questions, dynamically devise
logic/reasoning puzzles based on document context, and intelligently evaluate
user-submitted answers to those puzzles.

All LLM calls branch on the global config mode at runtime:
  - ``"external"`` → ``llm_providers.generate_text()`` (Gemini → Groq → OpenRouter)
  - ``"local"``    → ``local_llm.generate_local_text()`` (Ollama, model from config)
  - default / None → direct Gemini API call (original behaviour)

Components:
    generate_answer: Answers a question based on provided semantic chunks.
    load_context: Retrieves the entire document text dynamically from FAISS index metadata.
    generate_logic_questions: Constructs challenges based on context.
    evaluate_user_answers: Grades and provides feedback for logic challenges.

Dependencies:
    - google.generativeai: Direct Gemini LLM calls (default mode).
    - utils.llm_providers: Fallback-aware dispatcher (external mode).
    - utils.local_llm: Ollama-based local inference (local mode).
    - utils.model_config: Reads the active inference mode and model selection.
    - dotenv, os: Environment variable handling.
    - re, json: Used for cleaning and validating LLM output.
    - pickle: Reads text chunk metadata.
"""
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import json
import pickle

from utils.llm_providers import generate_text

# ── Initialization ───────────────────────────────────────────────

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "vectorstore", "chunk_texts.pkl")

# ── Core Operations ──────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[str], api_key: str = None, groq_api_key: str = None) -> str:
    """
    Generates a grounded answer with justification using the LLM fallback chain.
    """
    # Combine individual chunks into a unified string block for prompting
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

    return generate_text(prompt, api_key=api_key, groq_api_key=groq_api_key)

def load_context(session_id: str) -> str:
    """
    Loads all document chunks for a specific session as a single, concatenated string.
    """
    chunks_path = os.path.join(BASE_DIR, "vectorstore", session_id, "chunk_texts.pkl")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("No uploaded document found for this session.")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    # Cap string limit to roughly 20 chunks to avoid standard token limits
    return "\n\n".join(chunks[:20])

def generate_logic_questions(full_text: str, api_key: str = None, groq_api_key: str = None) -> str:
    """
    Generates 3 logic-based questions.
    """
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
{full_text}
\"\"\"
"""
    result = generate_text(prompt, api_key=api_key, groq_api_key=groq_api_key)
    print("\n[DEBUG] LLM Raw Output:\n", result)
    return result

def evaluate_user_answers(user_answers: list[str], session_id: str = None, api_key: str = None, groq_api_key: str = None) -> list[dict]:
    """
    Evaluates user-submitted answers against derived truth using the fallback chain.
    """
    if not session_id:
        raise ValueError("session_id is required for evaluation.")
    
    # Load the full document context to establish ground truth
    context = load_context(session_id)

    # Construct the evaluation prompt enforcing strict schema compliance
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

    raw_output = generate_text(prompt, api_key=api_key, groq_api_key=groq_api_key)

    # NOTE: Often LLMs output code blocks (e.g. ```json); these must be aggressively cleaned
    cleaned_output = re.sub(r"```json|```", "", raw_output).strip()

    # Safely digest string schema to dict; fallback on failure rather than crashing out
    try:
        return json.loads(cleaned_output)
    except Exception as e:
        return [{
            "error": "Failed to parse LLM response",
            "exception": str(e),
            "raw_output": cleaned_output
        }]