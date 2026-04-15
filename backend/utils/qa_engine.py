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

from utils.model_config import get_config
from utils.llm_providers import generate_text
from utils.local_llm import generate_local_text

# ── Initialization ───────────────────────────────────────────────

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "vectorstore", "chunk_texts.pkl")

# ── Core Operations ──────────────────────────────────────────────

def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates a grounded answer with justification using the Gemini LLM.

    This function pieces together retrieved chunks of semantic context 
    into a structured prompt, forcing the LLM to restrict its answers to 
    provided knowledge and state its sources logically.

    Args:
        question (str): The user's query about the document.
        context_chunks (list[str]): Relevant excerpts retrieved from FAISS.

    Returns:
        str: The generated response text, complete with internal justification.

    Example:
        answer = generate_answer("What is the ROI?", ["The ROI is 5%."])
        # answer => "Based on the text, the ROI is 5%."
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

    # ── Provider dispatch ──────────────────────────────────────
    cfg = get_config()
    if cfg["mode"] == "external":
        # Fallback chain: Gemini → Groq → OpenRouter
        return generate_text(prompt)
    elif cfg["mode"] == "local":
        # Route to locally-running Ollama model chosen by the user
        print(f"[LOCAL-LLM] generate_answer using local model: {cfg['llm_choice']}")
        return generate_local_text(prompt, cfg["llm_choice"])

    # Default: direct Gemini call (mode not yet set)
    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

def load_context() -> str:
    """
    Loads all document chunks as a single, concatenated string.

    This is necessary for operations that require understanding the entire document
    holistically (like generating global challenges), rather than answering targeted
    semantic queries.

    Returns:
        str: The full text context limited to approximately 20 chunks to prevent context overflow.

    Raises:
        FileNotFoundError: If the chunk metadata file isn't found.
    """
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("No uploaded document found.")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
        
    # Cap string limit to roughly 20 chunks to avoid standard token limits
    return "\n\n".join(chunks[:20])

def generate_logic_questions() -> str:
    """
    Asynchronously calls Gemini to generate 3 logic-based questions.

    This compiles the full conversational document context and prompts the LLM
    to formulate testable inference scenarios based on the data. 

    Returns:
        str: Raw, unparsed output directly from Gemini.
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
    # ── Provider dispatch ──────────────────────────────────────
    cfg = get_config()
    if cfg["mode"] == "external":
        result = generate_text(prompt)
        print("\n[DEBUG] LLM Raw Output:\n", result)
        return result
    elif cfg["mode"] == "local":
        print(f"[LOCAL-LLM] generate_logic_questions using local model: {cfg['llm_choice']}")
        result = generate_local_text(prompt, cfg["llm_choice"])
        print("\n[DEBUG] Local LLM Raw Output:\n", result)
        return result

    # Default: direct Gemini call
    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)
    print("\n[DEBUG] Gemini Raw Output:\n", response.text)
    return response.text.strip()

def evaluate_user_answers(user_answers: list[str]) -> list[dict]:
    """
    Uses Gemini to evaluate user-submitted answers against derived truth.

    This acts as an automatic grading system. It compares the user's answers against
    the source document, produces an ideal truth answer, computes a relative score,
    and returns granular feedback structure parsed from an enforced JSON response.

    Args:
        user_answers (list[str]): The plain text replies provided by the user.

    Returns:
        list[dict]: A list of objects containing question, truth, score, and feedback details.
                    If parsing fails, returns a fallback structure encapsulating the raw output.
    """
    # Load the full document context to establish ground truth
    context = load_context()

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

    # ── Provider dispatch ──────────────────────────────────────
    cfg = get_config()
    if cfg["mode"] == "external":
        raw_output = generate_text(prompt)
    elif cfg["mode"] == "local":
        print(f"[LOCAL-LLM] evaluate_user_answers using local model: {cfg['llm_choice']}")
        raw_output = generate_local_text(prompt, cfg["llm_choice"])
    else:
        # Default: direct Gemini call
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        raw_output = response.text.strip()

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