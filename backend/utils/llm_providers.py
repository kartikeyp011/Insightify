"""
LLM provider dispatcher with automatic fallback chain.

This module abstracts all LLM text generation behind a single public function
``generate_text(prompt)``. When mode is "external", it tries each provider in
the chain below and automatically falls back on any failure (exception, rate
limit, empty response):

    Gemini  →  Groq

Each provider is isolated in its own private function so failures are fully
contained. If every provider fails, a ``RuntimeError`` is raised so the
upstream caller receives a clean, descriptive error instead of a cryptic trace.

Usage (from qa_engine or any other utility):
    from utils.llm_providers import generate_text
    answer = generate_text(prompt)

Dependencies:
    - google.generativeai : Gemini (already a project dependency)
    - groq                : Groq SDK  (pip install groq)
    - requests            : OpenRouter HTTP calls (already installed)
    - dotenv, os          : Environment variable loading
"""

import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ── Provider constants ───────────────────────────────────────────

GEMINI_KEY        = os.getenv("GEMINI_KEY")
GROQ_KEY          = os.getenv("GROQ_API_KEY")
GEMINI_MODEL      = "gemini-flash-latest"
GROQ_MODEL        = "llama3-8b-8192"          # Fast, widely available on Groq

# ── Private provider implementations ────────────────────────────

def _call_gemini(prompt: str, api_key: str = None) -> str:
    """
    Calls the Gemini GenerativeModel API and returns the response text.

    Args:
        prompt (str): The full prompt to send to the model.

    Returns:
        str: Stripped response text from Gemini.

    Raises:
        Exception: Any network or API error propagates to the dispatcher.
    """
    key_to_use = api_key or GEMINI_KEY
    if not key_to_use:
        raise ValueError("GEMINI_KEY not set and no api_key provided — skipping Gemini.")

    genai.configure(api_key=key_to_use)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    text = response.text.strip()

    if not text:
        raise ValueError("Gemini returned an empty response.")

    return text


def _call_groq(prompt: str, api_key: str = None) -> str:
    """
    Calls the Groq Chat Completions API via the official groq SDK.

    Args:
        prompt (str): The full prompt to send as a user message.
        api_key (str): Optional user-provided Groq API Key.

    Returns:
        str: Stripped content from the first choice.

    Raises:
        ImportError: If the groq package is not installed.
        Exception: Any API or network error.
    """
    key_to_use = api_key or GROQ_KEY
    if not key_to_use:
        raise ValueError("GROQ_API_KEY not set and no api_key provided — skipping Groq.")

    try:
        from groq import Groq  # Lazy import so missing package only fails here
    except ImportError as exc:
        raise ImportError(
            "groq package not found. Install with: pip install groq"
        ) from exc

    client = Groq(api_key=key_to_use)
    chat_response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = chat_response.choices[0].message.content.strip()

    if not text:
        raise ValueError("Groq returned an empty response.")

    return text


# ── Public dispatcher ────────────────────────────────────────────

# Ordered chain: (provider name, callable)
_LLM_CHAIN = [
    ("Gemini",      _call_gemini),
    ("Groq",        _call_groq),
]


def generate_text(prompt: str, api_key: str = None, groq_api_key: str = None) -> str:
    """
    Generates text from the best available LLM provider.

    Iterates through the provider chain (Gemini → Groq),
    attempting each in turn. The first successful non-empty response is
    returned. If all providers fail the collective exception context is
    collected and re-raised as a RuntimeError.

    This function is the single entry point used by qa_engine for all
    LLM-dependent operations when mode is "external".

    Args:
        prompt (str): The complete prompt to pass to the LLM.

    Returns:
        str: The generated text response.

    Raises:
        RuntimeError: If every provider in the chain fails.

    Example:
        from utils.llm_providers import generate_text
        answer = generate_text("Summarise the following: ...")
    """
    errors: list[str] = []

    for name, call_fn in _LLM_CHAIN:
        try:
            print(f"[LLM] Attempting provider: {name}")
            if name == "Gemini":
                result = call_fn(prompt, api_key)
            elif name == "Groq":
                result = call_fn(prompt, groq_api_key)
            else:
                result = call_fn(prompt)
            print(f"[LLM] Success with provider: {name}")
            return result

        except Exception as exc:
            print(f"[LLM] FALLBACK — {name} failed: {exc}")
            errors.append(f"{name}: {exc}")

    # All providers exhausted
    error_msg = "All LLM providers failed. Details:\n" + "\n".join(errors)
    print(f"[LLM] CRITICAL ERROR:\n{error_msg}")
    
    # Return a graceful fallback string rather than crashing the request
    return (
        "⚠️ I'm sorry, but all AI inference providers are currently unavailable. "
        "Please check your API keys or try again later. "
        f"(Last error: {errors[-1] if errors else 'Unknown'})"
    )
