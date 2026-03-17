import os
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Step 1: Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_KEY")  # Make sure your .env uses GEMINI_KEY

if not api_key:
    print("❌ GEMINI_KEY is missing from your .env file")
    exit(1)

genai.configure(api_key=api_key)

# ✅ Step 2: Use the global genai.embed_content() function (not model.embed_content)
try:
    response = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content="This is a test chunk to embed.",
        task_type="retrieval_document"
    )

    embedding = response['embedding']
    print("✅ Embedding successful. Vector size:", len(embedding))
    print("🔹 Vector preview:", embedding[:5])

except Exception as e:
    print("❌ Gemini embedding failed:", e)