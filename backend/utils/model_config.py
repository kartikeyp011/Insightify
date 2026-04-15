"""
Global configuration manager for model and embedding selection.

This module maintains a single, process-wide configuration dictionary that
records which inference mode (local vs. external) the user has chosen, along
with the specific LLM and embedding model names. Because FastAPI runs in a
single process, a module-level dict is a safe and lightweight source of truth
for these settings.

The config is intended to be written once per request cycle (via the frontend's
/api/config endpoint, to be wired later) and read lazily by qa_engine,
retriever and embedder whenever they need to branch on the active model.

Components:
    set_config: Overwrites the active configuration.
    get_config: Returns a read-only copy of the current configuration.
    reset_config: Restores all fields to their default (None) values.

Dependencies:
    - threading: Provides a reentrant lock for concurrent-safe reads/writes.
"""

import threading

# ── Internal State ───────────────────────────────────────────────

# The single source of truth for the currently active model configuration.
# Fields:
#   mode            – "local" | "external" | None
#   llm_choice      – LLM display name chosen by the user, or None
#   embedding_choice – Embedding model display name chosen by the user, or None
_config: dict = {
    "mode": None,
    "llm_choice": None,
    "embedding_choice": None,
}

# Reentrant lock so that concurrent FastAPI requests don't race on reads/writes
_lock = threading.RLock()

# ── Valid Options ────────────────────────────────────────────────
# These mirror the options shown in the Streamlit frontend exactly.
# Keeping them here gives other modules a single authoritative source.

VALID_MODES = {"local", "external"}

VALID_LLM_CHOICES = {
    "Phi-3 Mini",
    "Gemma 2B",
    "DeepSeek R1",
    "SmolLM2",
}

VALID_EMBEDDING_CHOICES = {
    "BGE-base",
    "all-MiniLM-L6-v2",
    "e5-base",
}

# ── Public API ───────────────────────────────────────────────────

def set_config(
    mode: str,
    llm_choice: str | None = None,
    embedding_choice: str | None = None,
) -> None:
    """
    Overwrites the active model configuration with the supplied values.

    This is the sole write path into the config. It validates the supplied
    arguments against the allowed option sets and raises a ValueError for
    unrecognised values so that callers catch mismatches early rather than
    silently propagating bad state into model calls.

    For external mode, llm_choice and embedding_choice should be None (or
    simply omitted), since the backend will route to an external API without
    local model selection.

    Args:
        mode (str): Inference mode — must be "local" or "external".
        llm_choice (str | None): The LLM model name selected by the user.
            Required when mode is "local", ignored for "external".
        embedding_choice (str | None): The embedding model name selected by
            the user. Required when mode is "local", ignored for "external".

    Raises:
        ValueError: If mode is not one of the accepted values.
        ValueError: If mode is "local" and llm_choice / embedding_choice are
            not among the accepted options.

    Example:
        set_config("local", "Phi-3 Mini", "BGE-base")
        set_config("external")
    """
    # ── Validate mode ──────────────────────────────────────────
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}"
        )

    # ── Validate local-mode sub-selections ────────────────────
    if mode == "local":
        if llm_choice not in VALID_LLM_CHOICES:
            raise ValueError(
                f"Invalid llm_choice '{llm_choice}'. "
                f"Must be one of: {sorted(VALID_LLM_CHOICES)}"
            )
        if embedding_choice not in VALID_EMBEDDING_CHOICES:
            raise ValueError(
                f"Invalid embedding_choice '{embedding_choice}'. "
                f"Must be one of: {sorted(VALID_EMBEDDING_CHOICES)}"
            )

    # ── Persist atomically ────────────────────────────────────
    with _lock:
        _config["mode"] = mode
        # For external mode, explicitly clear any stale local selections
        _config["llm_choice"] = llm_choice if mode == "local" else None
        _config["embedding_choice"] = embedding_choice if mode == "local" else None

    print(
        f"[CONFIG] mode={_config['mode']} | "
        f"llm={_config['llm_choice']} | "
        f"embedding={_config['embedding_choice']}"
    )


def get_config() -> dict:
    """
    Returns a shallow copy of the current model configuration.

    Callers receive their own dict so they cannot accidentally mutate the
    shared state. The copy is taken inside the lock to guarantee a
    consistent snapshot even if another thread is mid-write.

    Returns:
        dict: A copy of the config with keys:
            - "mode"             (str | None)
            - "llm_choice"       (str | None)
            - "embedding_choice" (str | None)

    Example:
        cfg = get_config()
        if cfg["mode"] == "local":
            load_local_model(cfg["llm_choice"])
    """
    with _lock:
        # Return a copy so callers cannot mutate shared state
        return dict(_config)


def reset_config() -> None:
    """
    Resets all configuration fields to None.

    Useful in tests or when a new document is uploaded and the previous
    model selection should no longer be considered active.

    Returns:
        None

    Example:
        reset_config()
        assert get_config() == {"mode": None, "llm_choice": None, "embedding_choice": None}
    """
    with _lock:
        _config["mode"] = None
        _config["llm_choice"] = None
        _config["embedding_choice"] = None

    print("[CONFIG] Reset to defaults.")
