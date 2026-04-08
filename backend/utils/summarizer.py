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

# Initialize the Gemini connection for operations across the summarizer module
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

def generate_summary(text: str) -> str:
    """
    Uses the Gemini LLM to generate a concise summary of the provided text.

    This function attempts to synthesize the main points into a 150-word bounded
    paragraph. It protects against generation-level failures by catching exceptions.

    Args:
        text (str): The complete, raw input text representing a logical document.

    Returns:
        str: A synthesized summary text chunk, or a blank string/None if the operation failed.

    Example:
        summary = generate_summary("Extremely long earnings report...")
        # summary => "The report highlights a Q3 growth..."
    """
    try:
        # Utilize the flash model to optimize summarization latency constraints
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")

        # Instruction prompt ensuring token limitation compliance from Gemini
        prompt = (
            "Summarize the following document in 150 words or less. "
            "Be concise but cover the key points.\n\n"
            f"{text}"
        )

        # Await sync resolution of the request
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        # TODO(dev): Should gracefully propagate this forward as an HTTPException detail instead of console log
        print("Gemini summarization failed:", e)