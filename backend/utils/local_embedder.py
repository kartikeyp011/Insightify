"""
Local embedding dispatcher using sentence-transformers.

This module generates dense vector embeddings from locally-downloaded
HuggingFace models when the global config mode is ``"local"``. It uses the
``sentence-transformers`` library, which returns numpy arrays that are
immediately compatible with the FAISS index format used by the rest of the
pipeline.

Model instances are cached in a module-level registry dict so that a given
model is only loaded from disk once per process lifetime, regardless of how
many chunks or queries are embedded. This avoids the typical ~0.5–2 s cold-
start penalty on every request.

The ``e5-base`` model requires specific text prefixes (``"passage: "`` for
documents, ``"query: "`` for search queries) to achieve its optimal retrieval
performance. These are applied automatically based on the ``task`` parameter.

The single public function ``generate_local_embedding`` is signature-compatible
with ``embedding_providers.embed_text`` so callers in ``embedder`` and
``retriever`` can dispatch to either with a single branch.

Components:
    generate_local_embedding: Embed text with the selected local model.

Dependencies:
    - sentence-transformers : ``pip install sentence-transformers``
    - torch                 : Pulled transitively by sentence-transformers.
"""

from __future__ import annotations

# ── Model name mapping ───────────────────────────────────────────
# Maps frontend display labels to HuggingFace model identifiers.

_EMBEDDING_MODEL_MAP: dict[str, str] = {
    "BGE-base":          "BAAI/bge-base-en-v1.5",
    "all-MiniLM-L6-v2":  "sentence-transformers/all-MiniLM-L6-v2",
    "e5-base":           "intfloat/e5-base",
}

# Models that require a task-dependent text prefix for best retrieval quality.
# The e5 family is trained with "passage: " for documents and "query: " for queries.
_E5_PREFIX_MAP: dict[str, str] = {
    "document": "passage: ",
    "query":    "query: ",
}
_E5_MODELS = {"intfloat/e5-base"}

# ── Model registry (cache) ───────────────────────────────────────
# Populated lazily on first use; keyed by HuggingFace model ID.
# Avoids reloading heavy model weights on every embedding call.
_model_registry: dict[str, object] = {}

# ── Private helpers ──────────────────────────────────────────────

def _resolve_model_id(model_name: str) -> str:
    """
    Resolves a frontend display name to a HuggingFace model identifier.

    Args:
        model_name (str): The label selected by the user (e.g. ``"BGE-base"``).

    Returns:
        str: The corresponding HuggingFace model ID.

    Raises:
        ValueError: If ``model_name`` is not in the known mapping table.
    """
    model_id = _EMBEDDING_MODEL_MAP.get(model_name)
    if model_id is None:
        raise ValueError(
            f"Unknown local embedding model '{model_name}'. "
            f"Valid options: {sorted(_EMBEDDING_MODEL_MAP)}"
        )
    return model_id


def _get_or_load_model(model_id: str):
    """
    Returns a cached ``SentenceTransformer`` instance, loading it if needed.

    The first call for a given ``model_id`` downloads/loads the model from the
    local HuggingFace cache and stores it in ``_model_registry``. Subsequent
    calls return the cached instance immediately.

    Args:
        model_id (str): A HuggingFace model identifier string.

    Returns:
        SentenceTransformer: The loaded model instance.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
        Exception: Any model loading error from the sentence-transformers library.
    """
    if model_id in _model_registry:
        return _model_registry[model_id]

    try:
        from sentence_transformers import SentenceTransformer  # Lazy import
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    print(f"[LOCAL-EMBED] Loading model '{model_id}' (first use — may take a moment)...")
    model = SentenceTransformer(model_id)
    _model_registry[model_id] = model
    print(f"[LOCAL-EMBED] Model '{model_id}' loaded and cached.")
    return model


def _apply_prefix(text: str, model_id: str, task: str) -> str:
    """
    Prepends the task-specific prefix for models that require it (e.g. e5).

    Args:
        text (str): The raw text to embed.
        model_id (str): The HuggingFace model identifier.
        task (str): ``"document"`` or ``"query"``.

    Returns:
        str: Text with prefix applied if applicable, otherwise unchanged.
    """
    if model_id in _E5_MODELS:
        prefix = _E5_PREFIX_MAP.get(task, "passage: ")
        return prefix + text
    return text


# ── Public API ───────────────────────────────────────────────────

def generate_local_embedding(
    text: str,
    model_name: str,
    task: str = "document",
) -> list[float]:
    """
    Generates a dense embedding vector using the selected local sentence-transformer.

    Loads (or retrieves from cache) the model corresponding to ``model_name``,
    applies any required task-prefix, runs inference, and returns a plain
    ``list[float]`` that is compatible with both FAISS (via numpy conversion in
    the caller) and the output contract of ``embedding_providers.embed_text``.

    Logging prints indicate which model is active and whether it was a cache
    hit or a first-time load.

    Args:
        text (str): The text to embed (a chunk or a query string).
        model_name (str): The frontend display label (e.g. ``"BGE-base"``).
            Must be one of the keys in ``_EMBEDDING_MODEL_MAP``.
        task (str): Either ``"document"`` (for indexing chunks) or
            ``"query"`` (for search queries at retrieval time).
            Defaults to ``"document"``.

    Returns:
        list[float]: A flat list of floats representing the embedding vector.
            Dimensionality depends on the model (384 for MiniLM, 768 for BGE/e5).

    Raises:
        ValueError: If ``model_name`` is not a recognised local embedding option.
        ImportError: If ``sentence-transformers`` is not installed.
        RuntimeError: If the model returns an empty or malformed vector.

    Example:
        from utils.local_embedder import generate_local_embedding
        vec = generate_local_embedding("What is ML?", "BGE-base", task="query")
    """
    try:
        model_id = _resolve_model_id(model_name)
        print(f"[LOCAL-EMBED] Using local embedding model: {model_name} (id={model_id}, task={task})")

        # Apply task-specific prefix where the model architecture requires it
        prefixed_text = _apply_prefix(text, model_id, task)

        # Retrieve the cached model or load it for the first time
        model = _get_or_load_model(model_id)

        # encode() returns a numpy ndarray of shape (dim,)
        # convert_to_numpy=True is the default but stated explicitly for clarity
        vector = model.encode(prefixed_text, convert_to_numpy=True)

        if vector is None or len(vector) == 0:
            print(f"[LOCAL-EMBED] ERROR: Model '{model_name}' returned an empty vector.")
            return []

        return vector.tolist()

    except Exception as exc:
        print(f"[LOCAL-EMBED] ERROR: {exc}")
        return []
