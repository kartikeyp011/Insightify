"""
Router for handling document uploads and ingestion.

This module provides the API endpoint to ingest PDF or TXT files, extract
and chunk the text, generate embeddings via the Gemini Embedding API, and
store them in a FAISS vector database. It also accepts optional model
configuration parameters (mode, llm_choice, embedding_choice) and persists
them via the model_config manager so downstream components can adapt their
behaviour accordingly.

Components:
    upload_file: API endpoint to handle the file upload and processing pipeline.

Dependencies:
    - fastapi: Used for web endpoints, file handling, and HTTP exceptions.
    - utils.parser: To extract plain text from file contents.
    - utils.summarizer: To briefly summarize the parsed text.
    - utils.chunker: To divide text into semantic parts.
    - utils.embedder: To convert text into vector representations.
    - utils.model_config: To persist the user's model/embedding selection.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Import utility functions for parsing, summarizing, chunking, and embedding
from utils.parser import extract_text_from_file
from utils.summarizer import generate_summary
from utils.chunker import split_text_into_chunks
from utils.embedder import embed_and_store_chunks
from utils.model_config import set_config

# Initialize FastAPI router
router = APIRouter()

# ── Endpoints ────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form(default=""),
    llm_choice: str = Form(default=""),
    embedding_choice: str = Form(default=""),
    chunking_strategy: str = Form(default="Large Chunking (1200, overlap 200)"),
):
    """
    Asynchronously handles file uploads (PDF/TXT), processes text, and populates vectorstore.

    This endpoint reads the file content, extracts the raw string representation,
    generates a short summary, splits everything into chunks, computes embeddings
    using Gemini, and stores the vectors in a local FAISS index for retrieval.

    It also accepts optional model configuration form fields (mode, llm_choice,
    embedding_choice). When present and valid, these are persisted via set_config()
    so that downstream components (qa_engine, retriever, embedder) can branch on
    the active model. Fields are silently ignored if absent or empty to preserve
    backward compatibility with older clients.

    Args:
        file (UploadFile): The uploaded file, constrained to .pdf or .txt formats.
        mode (str): Optional. Inference mode — "local" or "external".
        llm_choice (str): Optional. LLM model name (required when mode is "local").
        embedding_choice (str): Optional. Embedding model name (required when mode is "local").

    Returns:
        JSONResponse: An object containing a short text summary under the "summary" key,
            plus a "config" sub-object echoing back the persisted model configuration.

    Raises:
        HTTPException:
            - 400 if the incoming file is not PDF or TXT.
            - 422 if text extraction yields no content.
            - 500 if an unexpected error occurs during processing.
    """
    filename = file.filename

    # ── Model Configuration ──────────────────────────────────────
    # Persist the user's model/embedding selection when valid values are supplied.
    # Fields arrive as empty strings when the frontend sends no selection, so we
    # treat "" the same as absent to preserve full backward compatibility.
    if mode:  # non-empty string means the user explicitly chose something
        try:
            set_config(
                mode=mode,
                llm_choice=llm_choice or None,
                embedding_choice=embedding_choice or None,
            )
        except ValueError as exc:
            # Invalid option values — surface as a 400 rather than crashing the
            # whole upload. The file processing pipeline still runs below.
            raise HTTPException(status_code=400, detail=f"Invalid model config: {exc}")

    # Step 1: Check if file is a PDF or TXT to prevent unhandled media types
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    try:
        # Step 2: Read file content into memory asynchronously
        contents = await file.read()

        # Step 3: Extract clean text from file bytes based on its format
        text = extract_text_from_file(contents, filename)

    except Exception as e:
        # Surface fitz/IO errors with a clear message instead of masking them
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    # Step 4: Validate extracted text to ensure subsequent steps don't fail on empty strings
    # This check lives OUTSIDE the broad except so the 422 is never swallowed as a 500
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the PDF. The file may be scanned/image-based with no selectable text."
        )

    try:
        # Step 5: Generate a short API-driven summary for the frontend
        # NOTE: This uses Gemini behind the scenes.
        summary = generate_summary(text)

        # Step 6: Save raw text to a file (optional but useful for debugging full context)
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vectorstore_path = os.path.join(base_dir, "vectorstore")
        os.makedirs(vectorstore_path, exist_ok=True)
        with open(os.path.join(vectorstore_path, "temp_text.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Step 7: Split the text into overlapping chunks to maintain semantic context
        chunks = split_text_into_chunks(text, strategy=chunking_strategy)

        # Step 8: Use Gemini Embedding API to create vector embeddings & save to FAISS
        embed_and_store_chunks(chunks)

        # Step 9: Return generated summary plus the active config as response
        from utils.model_config import get_config
        return JSONResponse(
            content={"summary": summary, "config": get_config()},
            status_code=200,
        )

    except Exception as e:
        # If anything goes wrong in the summarization/embedding pipeline, surface it cleanly
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
