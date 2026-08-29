"""
General document summarization utility.

This module provides a singular abstraction for producing brief executive
summaries from large blobs of text content, powering initial frontend insights.

Dependencies:
    - google.generativeai: Connects to LLM endpoints.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

from utils.llm_providers import generate_text

# Initialize the environment
load_dotenv()

def generate_summary(text: str, api_key: str = None, groq_api_key: str = None) -> str:
    """
    Uses the AI fallback chain to generate a concise summary of the provided text.

    This function attempts to synthesize the main points into a 150-word bounded
    paragraph. It protects against generation-level failures by catching exceptions.

    Args:
        text (str): The complete, raw input text representing a logical document.

    Returns:
        str: A synthesized summary text chunk, or a blank string/None if the operation failed.
    """
    try:
        # Instruction prompt ensuring token limitation compliance from Gemini
        prompt = (
            "Summarize the following document in 150 words or less. "
            "Be concise but cover the key points.\n\n"
            f"{text}"
        )

        return generate_text(prompt, api_key=api_key, groq_api_key=groq_api_key)

    except Exception as e:
        # TODO(dev): Should gracefully propagate this forward as an HTTPException detail instead of console log
        print("Gemini summarization failed:", e)