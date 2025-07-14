from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Import utility functions for parsing, summarizing, chunking, and embedding
from backend.utils.parser import extract_text_from_file
from backend.utils.summarizer import generate_summary
from backend.utils.chunker import split_text_into_chunks
from backend.utils.embedder import embed_and_store_chunks

# Initialize FastAPI router
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    POST /upload

    This endpoint accepts a PDF or TXT file:
    1. Parses and extracts clean text.
    2. Generates a short summary (≤150 words).
    3. Splits the text into chunks for embedding.
    4. Uses Gemini API to create vector embeddings.
    5. Stores vectors in a FAISS database for later search.
    6. Returns the summary in JSON response.
    """

    filename = file.filename

    # ✅ Step 1: Check if file is a PDF or TXT
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    try:
        # ✅ Step 2: Read file content into memory
        contents = await file.read()

        # ✅ Step 3: Extract clean text from file
        text = extract_text_from_file(contents, filename)

        # ✅ Step 4: Validate extracted text
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=422, detail="Could not extract text from file.")

        # ✅ Step 5: Generate a short summary (placeholder for Gemini summary)
        summary = generate_summary(text)

        # ✅ Step 6: Save raw text to a file (optional: useful for debug or later use)
        with open("backend/vectorstore/temp_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

        # ✅ Step 7: Split the text into overlapping chunks
        chunks = split_text_into_chunks(text)

        # ✅ Step 8: Use Gemini Embedding API to create vector embeddings & save to FAISS
        embed_and_store_chunks(chunks)

        # ✅ Step 9: Return summary as response
        return JSONResponse(content={"summary": summary}, status_code=200)

    except Exception as e:
        # ✅ If anything goes wrong, return a 500 Internal Server Error with details
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")