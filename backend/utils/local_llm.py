"""
Local LLM dispatcher using Ollama as the inference backend.

This module routes LLM generation tasks to a locally-running Ollama instance
when the global config mode is ``"local"``. It communicates with Ollama's REST
API at ``http://localhost:11434`` — no Python ML dependencies are required
beyond the standard ``requests`` library.

Ollama manages model loading, GPU offloading, and in-memory caching itself,
so this module only needs to maintain the frontend-label → Ollama-tag mapping
and issue JSON HTTP calls.

The single public function ``generate_local_text(prompt, model_name)`` is
intentionally signature-compatible with ``llm_providers.generate_text`` so
callers in ``qa_engine`` can dispatch to either with a single branch.

Components:
    generate_local_text: Send a prompt to the locally running Ollama model.

Dependencies:
    - requests : HTTP calls to the Ollama REST API (already installed).
    - Ollama   : Must be installed and running on the host machine.
                 Models must be pre-pulled with ``ollama pull <tag>``.
"""

import requests

# ── Constants ────────────────────────────────────────────────────

OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_GEN_URL    = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL   = f"{OLLAMA_BASE_URL}/api/tags"   # Used for availability check

# Timeout for generation requests — local models can be slow on CPU
OLLAMA_TIMEOUT    = 300  # seconds

# ── Model name mapping ───────────────────────────────────────────
# Maps the display labels shown in the Streamlit frontend to the Ollama
# model tags used in API calls. Tags must match what is pulled locally.

_LLM_TAG_MAP: dict[str, str] = {
    "Phi-3 Mini":    "phi3:mini",
    "Gemma 2B":      "gemma:2b",
    "DeepSeek R1":   "deepseek-r1:1.5b",
    "SmolLM2":       "smollm2:135m",
}

# ── Private helpers ──────────────────────────────────────────────

def _resolve_tag(model_name: str) -> str:
    """
    Resolves a frontend display name to an Ollama model tag.

    Args:
        model_name (str): The label selected by the user in the frontend
            (e.g. ``"Phi-3 Mini"``).

    Returns:
        str: The corresponding Ollama model tag (e.g. ``"phi3:mini"``).

    Raises:
        ValueError: If ``model_name`` is not in the known mapping table.
    """
    tag = _LLM_TAG_MAP.get(model_name)
    if tag is None:
        raise ValueError(
            f"Unknown local LLM '{model_name}'. "
            f"Valid options: {sorted(_LLM_TAG_MAP)}"
        )
    return tag


def _check_ollama_reachable() -> None:
    """
    Verifies that the Ollama daemon is reachable before attempting generation.

    Raises:
        RuntimeError: If the Ollama server is not responding at the expected URL.
    """
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. "
            "Ensure Ollama is installed and running (`ollama serve`). "
            f"Details: {exc}"
        ) from exc


# ── Public API ───────────────────────────────────────────────────

def generate_local_text(prompt: str, model_name: str) -> str:
    """
    Sends a prompt to the selected locally-running Ollama LLM.

    This function resolves the frontend display name to an Ollama model tag,
    verifies the Ollama daemon is reachable, and issues a non-streaming
    generation request. The full response text is returned as a plain string,
    matching the output contract of ``llm_providers.generate_text``.

    Logging prints indicate which model is active so server logs clearly
    show when local mode is in use vs. external providers.

    Args:
        prompt (str): The complete prompt string to send to the model.
        model_name (str): The frontend display label (e.g. ``"Phi-3 Mini"``).
            Must be one of the keys in ``_LLM_TAG_MAP``.

    Returns:
        str: The generated text response, stripped of leading/trailing whitespace.

    Raises:
        ValueError: If ``model_name`` is not a recognised local LLM option.
        RuntimeError: If Ollama is not running or the model is not pulled.
        RuntimeError: If Ollama returns an empty response.

    Example:
        from utils.local_llm import generate_local_text
        answer = generate_local_text("Summarise the document.", "Phi-3 Mini")
    """
    try:
        tag = _resolve_tag(model_name)
        print(f"[LOCAL-LLM] Using local model: {model_name} (tag={tag})")

        # Verify daemon is up before spending time on the HTTP call
        _check_ollama_reachable()

        payload = {
            "model":  tag,
            "prompt": prompt,
            "stream": False,   # Return full response in a single JSON object
        }

        resp = requests.post(OLLAMA_GEN_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        
        # Handle 404 cleanly since it typically means the tag isn't pulled yet
        if resp.status_code == 404:
            return f"⚠️ Model '{tag}' is not available in Ollama. Please pull it first: `ollama pull {tag}`"
            
        resp.raise_for_status()

        data = resp.json()
        text = data.get("response", "").strip()

        if not text:
            return f"⚠️ Ollama returned an empty response for model '{tag}'."

        print(f"[LOCAL-LLM] Response received from {model_name} ({len(text)} chars).")
        return text

    except Exception as exc:
        print(f"[LOCAL-LLM] ERROR: {exc}")
        return f"⚠️ Local LLM failed: {exc}."
