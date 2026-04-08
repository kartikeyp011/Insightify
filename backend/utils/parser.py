"""
Utility for parsing raw document files into plain text.

This module provides helper utilities to safely extract unstructured text 
from various file formats (e.g., PDF, TXT) uploaded by users.

Dependencies:
    - fitz (PyMuPDF): Crucial for high-performance extraction of text from PDFs.
"""
import fitz  # PyMuPDF library for reading PDFs

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Reads and extracts clean text from uploaded PDF or TXT file contents.

    This function handles decoding raw text files and parsing through PDF
    pages dynamically, extracting all readable plaintext.

    Args:
        file_bytes (bytes): The raw byte stream of the uploaded file.
        filename (str): The filename used to deduce the file type/extension.

    Returns:
        str: The fully extracted, clean plain text from the file.

    Example:
        raw_text = extract_text_from_file(b"Hello world", "test.txt")
        # raw_text => "Hello world"
    """
    if filename.endswith(".txt"):
        # Convert bytes directly to string using UTF-8 decoding for standard text files
        return file_bytes.decode("utf-8")

    elif filename.endswith(".pdf"):
        text = ""
        # Open the PDF directly from the in-memory byte stream using PyMuPDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                # Accumulate the plain text extracted from each sequential page
                text += page.get_text()
        return text

    else:
        # Fallback for unsupported formats (ideally intercepted by earlier validation)
        return ""
