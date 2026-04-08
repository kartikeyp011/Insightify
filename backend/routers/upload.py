"""
Router for handling document uploads and ingestion.

This module provides the API endpoint to ingest PDF or TXT files, extract
and chunk the text, generate embeddings via the Gemini Embedding API, and
store them in a FAISS vector database.

Components:
    upload_file: API endpoint to handle the file upload and processing pipeline.

Dependencies:
    - fastapi: Used for web endpoints, file handling, and HTTP exceptions.
    - utils.parser: To extract plain text from file contents.
    - utils.summarizer: To briefly summarize the parsed text.
    - utils.chunker: To divide text into semantic parts.
    - utils.embedder: To convert text into vector representations.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import utility functions for parsing, summarizing, chunking, and embedding
from utils.parser import extract_text_from_file
from utils.summarizer import generate_summary
from utils.chunker import split_text_into_chunks
from utils.embedder import embed_and_store_chunks

# Initialize FastAPI router
router = APIRouter()

# ── Endpoints ────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Asynchronously handles file uploads (PDF/TXT), processes text, and populates vectorstore.

    This endpoint reads the file content, extracts the raw string representation,
    generates a short summary, splits everything into chunks, computes embeddings
    using Gemini, and stores the vectors in a local FAISS index for retrieval.

    Args:
        file (UploadFile): The uploaded file, constrained to .pdf or .txt formats.

    Returns:
        JSONResponse: An object containing a short text summary under the "summary" key.

    Raises:
        HTTPException:
            - 400 if the incoming file is not PDF or TXT.
            - 422 if text extraction yields no content.
            - 500 if an unexpected error occurs during processing.
    """
    filename = file.filename

    # Step 1: Check if file is a PDF or TXT to prevent unhandled media types
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    try:
        # Step 2: Read file content into memory asynchronously
        contents = await file.read()

        # Step 3: Extract clean text from file bytes based on its format
        text = extract_text_from_file(contents, filename)

        # Step 4: Validate extracted text to ensure subsequent steps don't fail on empty strings
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

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
        chunks = split_text_into_chunks(text)

        # Step 8: Use Gemini Embedding API to create vector embeddings & save to FAISS
        embed_and_store_chunks(chunks)

        # Step 9: Return generated summary as response
        return JSONResponse(content={"summary": summary}, status_code=200)

    except Exception as e:
        # If anything goes wrong inside the pipeline, return a cleanly surfaced error
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")