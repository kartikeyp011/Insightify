"""
Utility for generating and storing vector embeddings with provider fallback.

This module is responsible for taking raw text chunks, converting them into
feature vectors, and storing both the vectors and their spatial index using
FAISS for rapid similarity search.

All embedding generation branches on the global config mode at runtime:
  - ``"external"`` → ``embedding_providers.embed_text()`` (Gemini → Together AI → HF)
  - ``"local"``    → ``local_embedder.generate_local_embedding()`` (sentence-transformers)
  - default / None → direct Gemini Embedding API call (original behaviour)

Components:
    embed_and_store_chunks: Embeds text and populates the FAISS store.

Dependencies:
    - os, pickle: For file paths and saving text metadata.
    - numpy: For vector manipulation.
    - faiss: For creating the similarity search index.
    - dotenv: To manage environment secrets.
    - google.generativeai: Direct Gemini embedding (default mode).
    - utils.embedding_providers: Fallback-aware dispatcher (external mode).
    - utils.local_embedder: sentence-transformers dispatcher (local mode).
    - utils.model_config: Reads the active inference mode and model selection.
"""
import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

from utils.model_config import get_config
from utils.embedding_providers import embed_text
from utils.local_embedder import generate_local_embedding

# ── Initialization ───────────────────────────────────────────────

# Load Gemini API Key from .env file securely
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Compute absolute paths to correctly resolve vectorstore storage locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
META_PATH = os.path.join(BASE_DIR, "vectorstore", "chunk_texts.pkl")

# ── Embedding Logic ──────────────────────────────────────────────

def embed_and_store_chunks(chunks: list[str]) -> None:
    """
    Embeds a list of text chunks using Gemini and stores them in a FAISS index.

    This function calls the Gemini Embedding API for each chunk, converting the
    semantic text into a high-dimensional vector. It creates a FAISS index using
    L2 distance and stores the raw text locally so context can be retrieved via
    index alignment.

    Args:
        chunks (list[str]): The plain text chunks to be embedded.

    Returns:
        None

    Raises:
        ValueError: If Gemini completely fails and produces 0 vectors.
    """
    print(f"[INFO] Embedding {len(chunks)} chunks...")

    vectors = []
    cleaned_chunks = []

    # Resolve the active mode once before the loop to avoid redundant config reads
    cfg = get_config()
    mode = cfg["mode"]
    embedding_model = cfg.get("embedding_choice")  # Only set when mode == "local"

    if mode == "local":
        print(f"[LOCAL-EMBED] embed_and_store_chunks using local model: {embedding_model}")

    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            # Skip iterations where chunk is purely whitespace to prevent API errors
            continue

        try:
            if mode == "external":
                # ── External mode: fallback chain (Gemini → Together AI → HF) ──
                vector = embed_text(chunk, task="document")
                vectors.append(np.array(vector, dtype="float32"))
            elif mode == "local":
                # ── Local mode: sentence-transformers (model from config) ──────
                vector = generate_local_embedding(chunk, embedding_model, task="document")
                vectors.append(np.array(vector, dtype="float32"))
            else:
                # ── Default: direct Gemini Embedding API call ────────────────
                # Generate the embedding using the Gemini API explicitly designated for retrieval
                response = genai.embed_content(
                    model="models/gemini-embedding-2-preview",
                    content=chunk,
                    task_type="retrieval_document",
                )
                # Append the resulting vector parsed as float32 required by FAISS
                vectors.append(np.array(response["embedding"], dtype="float32"))

            cleaned_chunks.append(chunk)

        except Exception as e:
            # Soft fail so that one bad chunk doesn't crash the entire document pipeline
            print(f"[WARN] Chunk #{idx+1} failed → {e}")
            print("Partial content:", repr(chunk[:100]))

    # Ensure at least one successful embedding exists to create a valid index
    if not vectors:
        raise ValueError("[ERROR] Failed to embed all chunks. No vectors created.")

    # Convert the list of arrays into a single contiguous multi-dimensional NumPy array
    embeddings_array = np.array(vectors, dtype="float32")

    # Create the FAISS similarity index based on vector dimensionality (likely 768)
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)

    # Persist both the FAISS binary index and the original chunk text list
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(cleaned_chunks, f)

    print(f"[SUCCESS] FAISS index saved with {len(cleaned_chunks)} vectors.")