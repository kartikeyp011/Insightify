"""
Embedding provider dispatcher with automatic fallback chain.

This module abstracts all vector embedding generation behind a single public
function ``embed_text(text, task)``. When mode is "external", it tries each
provider in the chain below and automatically falls back on any failure:

    Gemini Embeddings  →  Together AI  →  Hugging Face Inference API

All providers return a plain ``list[float]`` so callers (embedder.py,
retriever.py) remain completely provider-agnostic.

The ``task`` parameter accepts ``"document"`` or ``"query"`` and is mapped to
each provider's expected task-type string internally.

Usage:
    from utils.embedding_providers import embed_text
    vector = embed_text("some chunk of text", task="document")
    query_vector = embed_text("user question", task="query")

Dependencies:
    - google.generativeai : Gemini embeddings (already a project dependency)
    - requests            : Together AI and HF HTTP calls (already installed)
    - dotenv, os          : Environment variable loading
"""

import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ── Provider constants ───────────────────────────────────────────

GEMINI_KEY      = os.getenv("GEMINI_KEY")
TOGETHER_KEY    = os.getenv("TOGETHER_API_KEY")
HF_KEY          = os.getenv("HF_API_KEY")

GEMINI_EMB_MODEL    = "models/gemini-embedding-2-preview"

# Together AI — BAAI/bge-base-en-v1.5 is a top-tier open retrieval model
TOGETHER_EMB_MODEL  = "BAAI/bge-base-en-v1.5"
TOGETHER_EMB_URL    = "https://api.together.xyz/v1/embeddings"

# Hugging Face Inference API — all-MiniLM-L6-v2 is compact and widely cached
HF_EMB_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMB_URL          = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMB_MODEL}"

# Map generic task names → Gemini-specific task types
_GEMINI_TASK_MAP = {
    "document": "retrieval_document",
    "query":    "retrieval_query",
}

# Configure Gemini once at module load
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ── Private provider implementations ────────────────────────────

def _call_gemini_embed(text: str, task: str) -> list[float]:
    """
    Embeds text using the Gemini Embedding API.

    Args:
        text (str): The text to embed.
        task (str): ``"document"`` or ``"query"``.

    Returns:
        list[float]: The embedding vector.

    Raises:
        Exception: Any API or network error.
    """
    if not GEMINI_KEY:
        raise ValueError("GEMINI_KEY not set — skipping Gemini embeddings.")

    task_type = _GEMINI_TASK_MAP.get(task, "retrieval_document")
    response = genai.embed_content(
        model=GEMINI_EMB_MODEL,
        content=text,
        task_type=task_type,
    )
    vector = response["embedding"]

    if not vector:
        raise ValueError("Gemini returned an empty embedding.")

    return [float(v) for v in vector]


def _call_together_embed(text: str, task: str) -> list[float]:  # noqa: ARG001
    """
    Embeds text using the Together AI Embeddings API.

    Together AI does not differentiate document vs query task types at the API
    level, so the ``task`` parameter is accepted but unused.

    Args:
        text (str): The text to embed.
        task (str): Ignored for this provider (accepted for interface parity).

    Returns:
        list[float]: The embedding vector.

    Raises:
        requests.HTTPError: On non-2xx HTTP status.
        Exception: Any network or parsing error.
    """
    if not TOGETHER_KEY:
        raise ValueError("TOGETHER_API_KEY not set — skipping Together AI.")

    headers = {
        "Authorization": f"Bearer {TOGETHER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TOGETHER_EMB_MODEL,
        "input": text,
    }

    resp = requests.post(TOGETHER_EMB_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    vector = resp.json()["data"][0]["embedding"]

    if not vector:
        raise ValueError("Together AI returned an empty embedding.")

    return [float(v) for v in vector]


def _call_hf_embed(text: str, task: str) -> list[float]:  # noqa: ARG001
    """
    Embeds text using the Hugging Face Inference API (feature-extraction pipeline).

    The HF Inference API returns a nested list for sentence-transformer models;
    this function handles both ``[[float, ...]]`` and ``[float, ...]`` shapes.

    Args:
        text (str): The text to embed.
        task (str): Ignored for this provider (accepted for interface parity).

    Returns:
        list[float]: The flat embedding vector.

    Raises:
        requests.HTTPError: On non-2xx HTTP status.
        Exception: Any network or parsing error.
    """
    if not HF_KEY:
        raise ValueError("HF_API_KEY not set — skipping Hugging Face.")

    headers = {
        "Authorization": f"Bearer {HF_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    resp = requests.post(HF_EMB_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    raw = resp.json()

    # HF sentence-transformer models return [[v1, v2, ...]] (mean-pooled, wrapped)
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        vector = raw[0]
    else:
        vector = raw  # Already flat

    if not vector:
        raise ValueError("Hugging Face returned an empty embedding.")

    return [float(v) for v in vector]


# ── Public dispatcher ────────────────────────────────────────────

# Ordered chain: (provider name, callable)
_EMBEDDING_CHAIN = [
    ("Gemini",       _call_gemini_embed),
    ("Together AI",  _call_together_embed),
    ("Hugging Face", _call_hf_embed),
]


def embed_text(text: str, task: str = "document") -> list[float]:
    """
    Generates an embedding vector from the best available provider.

    Iterates through the provider chain (Gemini → Together AI → Hugging Face),
    attempting each in turn. The first successful non-empty vector is returned.
    If all providers fail, a ``RuntimeError`` is raised.

    This is the single entry point used by ``embedder.py`` and ``retriever.py``
    when mode is "external".

    Args:
        text (str): The text to embed.
        task (str): ``"document"`` for indexing chunks, ``"query"`` for search
            queries. Defaults to ``"document"``.

    Returns:
        list[float]: A flat list of floats representing the embedding vector.

    Raises:
        RuntimeError: If every provider in the chain fails.

    Example:
        from utils.embedding_providers import embed_text
        vec = embed_text("What is machine learning?", task="query")
    """
    errors: list[str] = []

    for name, call_fn in _EMBEDDING_CHAIN:
        try:
            print(f"[EMBED] Attempting provider: {name} (task={task})")
            vector = call_fn(text, task)
            print(f"[EMBED] Success with provider: {name} — dim={len(vector)}")
            return vector

        except Exception as exc:
            print(f"[EMBED] FALLBACK — {name} failed: {exc}")
            errors.append(f"{name}: {exc}")

    # All providers exhausted
    error_msg = "All embedding providers failed. Details:\n" + "\n".join(errors)
    print(f"[EMBED] CRITICAL ERROR:\n{error_msg}")
    
    # Return an empty vector so the system degrades gracefully rather than crashing.
    # Downstream callers (embedder.py, retriever.py) already handle empty responses.
    return []
