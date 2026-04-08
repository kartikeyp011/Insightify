"""
Diagnostic script for testing the Gemini embedding pipeline.

This module exists independently of the main API routing paths to verify whether
environmental keys validly resolve vectors against the generative endpoint. It is
run locally.

Dependencies:
    - dotenv: Loads environmental keys.
    - google.generativeai: Makes test requests to model endpoints.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configuration setup for testing explicitly without tying to overall app setup
load_dotenv()
api_key = os.getenv("GEMINI_KEY")

# Check if environment setup exists properly to assist isolated local debugging
if not api_key:
    print("❌ GEMINI_KEY is missing from your .env file")
    exit(1)

genai.configure(api_key=api_key)

# ── Validation Invocation ──────────────────────────────────────

try:
    # Uses genai.embed_content explicitly vs instantiating structural models
    response = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content="This is a test chunk to embed.",
        task_type="retrieval_document"
    )

    # Validate output vector structures manually
    embedding = response['embedding']
    print("✅ Embedding successful. Vector size:", len(embedding))
    print("🔹 Vector preview:", embedding[:5])

except Exception as e:
    # Will fail if account hits rate limits or keys are broadly invalid
    print("❌ Gemini embedding failed:", e)