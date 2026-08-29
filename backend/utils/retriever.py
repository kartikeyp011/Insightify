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

from utils.embedding_providers import embed_text

# ── Initialization ───────────────────────────────────────────────

# Ensure API configurations are set up to capture live requests
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_session_paths(session_id: str):
    """Returns the faiss_index and chunk_texts.pkl paths for the given session_id."""
    session_dir = os.path.join(BASE_DIR, "vectorstore", session_id)
    return os.path.join(session_dir, "faiss_index"), os.path.join(session_dir, "chunk_texts.pkl")

# ── Operations ──────────────────────────────────────────────────

def get_relevant_chunks(query: str, session_id: str = None, top_k: int = 4, api_key: str = None) -> list[str]:
    """
    Embeds the user query and retrieves conceptually similar chunks from FAISS.
    """
    if not session_id:
        raise ValueError("session_id must be provided to retrieve embeddings.")



    Example:
        excerpts = get_relevant_chunks("What is the conclusion?", top_k=2)
        # excerpts => ["Conclusion: Context A...", "Summary: Context B..."]
    """
    # ── Database Verification ───────────────────────────────────
    index_path, meta_path = get_session_paths(session_id)
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Vector store not found for this session.")

    # ── Memory Loading ─────────────────────────────────────────
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        all_chunks = pickle.load(f)

    # ── Vectorization ───────────────────────────────────────────
    raw_vector = embed_text(query, task="query", api_key=api_key)
    query_vector = np.array(raw_vector, dtype="float32").reshape(1, -1)

    # ── Searching ──────────────────────────────────────────────
    # Distances provide score magnitude; indices locate actual string mappings
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]

    return retrieved_chunks