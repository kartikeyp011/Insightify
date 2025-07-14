import fitz  # PyMuPDF library for reading PDFs

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Reads and extracts text from uploaded PDF or TXT file content.
    - file_bytes: raw bytes of the file
    - filename: name (to check extension)
    Returns: clean text as a string
    """

    if filename.endswith(".txt"):
        # Convert bytes to string (UTF-8 decoding)
        return file_bytes.decode("utf-8")

    elif filename.endswith(".pdf"):
        text = ""
        # Read PDF directly from memory using PyMuPDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                # Get plain text from each page
                text += page.get_text()
        return text

    else:
        # Not supported (should not reach here due to earlier check)
        return ""
