import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load Gemini API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

def generate_summary(text: str) -> str:
    """
    Uses Gemini LLM to generate a concise summary (≤ 150 words) from full document text.
    """

    try:
        # Initialize the Gemini model (text-only)
        model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

        # Instruction prompt for Gemini
        prompt = (
            "Summarize the following document in 150 words or less. "
            "Be concise but cover the key points.\n\n"
            f"{text}"
        )

        # Generate the summary
        response = model.generate_content(prompt)

        # Extract and return the text response
        return response.text.strip()

    except Exception as e:
        # In case Gemini fails
        print("Gemini summarization failed:", e)