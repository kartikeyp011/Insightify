"""
Utility for querying and retrieving semantic vectors from a FAISS index,
with external provider fallback support.

This module acts as the search bridge between a user's plain-text query and
the numerical embeddings stored locally. It embeds the live query and executes
an L2 distance search across the index space.

All query embedding branches on the global config mode at runtime:
  - ``"external"`` → ``embedding_providers.embed_text()`` (Gemini → Together AI → HF)
  - ``"local"``    → ``local_embedder.generate_local_embedding()`` (sentence-transformers)
  - default / None → direct Gemini Embedding API call (original behaviour)

Components:
    get_relevant_chunks: Primary interface for document retrieval.

Dependencies:
    - numpy: Handles the raw vector layout.
    - pickle: For reading associated textual metadata linked to indices.
    - faiss: For executing the nearest neighbors lookup.
    - google.generativeai: Direct Gemini query embedding (default mode).
    - utils.embedding_providers: Fallback-aware dispatcher (external mode).
    - utils.local_embedder: sentence-transformers dispatcher (local mode).
    - utils.model_config: Reads the active inference mode and model selection.
"""
import numpy as np
import pickle
import faiss
import os
from dotenv import load_dotenv
import google.generativeai as genai

from utils.model_config import get_config
from utils.embedding_providers import embed_text
from utils.local_embedder import generate_local_embedding

# ── Initialization ───────────────────────────────────────────────

# Ensure API configurations are set up to capture live requests
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
CHUNKS_PATH = os.path.join(BASE_DIR, "vectorstore", "chunk_texts.pkl")

# ── Operations ──────────────────────────────────────────────────

def get_relevant_chunks(query: str, top_k: int = 4) -> list[str]:
    """
    Embeds the user query and retrieves conceptually similar chunks from FAISS.

    This function first ensures the FAISS index database exists. If so, it embeds
    the incoming user query string using the Gemini Embedding API, searches for the `top_k`
    nearest vectors, and maps these vector indices back to their actual underlying
    text chunk content.

    Args:
        query (str): The plain-text question being asked.
        top_k (int, optional): Maximum amount of related text chunks to return. Defaults to 4.

    Returns:
        list[str]: A list of relevant semantic text string excerpts.

    Raises:
        FileNotFoundError: If the index vectors or mapping references are entirely absent.

    Example:
        excerpts = get_relevant_chunks("What is the conclusion?", top_k=2)
        # excerpts => ["Conclusion: Context A...", "Summary: Context B..."]
    """
    # ── Database Verification ───────────────────────────────────
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store not found.")

    # ── Memory Loading ─────────────────────────────────────────
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)

    # ── Vectorization ───────────────────────────────────────────
    cfg = get_config()
    mode = cfg["mode"]
    embedding_model = cfg.get("embedding_choice")  # Only set when mode == "local"

    if mode == "external":
        # ── External mode: fallback chain (Gemini → Together AI → HF) ──
        raw_vector = embed_text(query, task="query")
        query_vector = np.array(raw_vector, dtype="float32").reshape(1, -1)
    elif mode == "local":
        # ── Local mode: sentence-transformers (model from config) ────────
        print(f"[LOCAL-EMBED] get_relevant_chunks using local model: {embedding_model}")
        raw_vector = generate_local_embedding(query, embedding_model, task="query")
        query_vector = np.array(raw_vector, dtype="float32").reshape(1, -1)
    else:
        # ── Default: direct Gemini Embedding API call ────────────────────
        response = genai.embed_content(
            model="models/gemini-embedding-2-preview",
            content=query,
            task_type="retrieval_query",
        )
        # Needs to be reshaped to conform to FAISS search expectations [1, 768]
        query_vector = np.array(response["embedding"], dtype="float32").reshape(1, -1)

    # ── Searching ──────────────────────────────────────────────
    # Distances provide score magnitude; indices locate actual string mappings
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]

    return retrieved_chunks