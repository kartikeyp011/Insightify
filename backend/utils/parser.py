"""
Utility for parsing raw document files into plain text.

This module provides helper utilities to safely extract unstructured text
from various file formats (e.g., PDF, TXT) uploaded by users.

PDF extraction strategy (two-stage):
    1. Native text extraction via fitz (fast; works on text-based PDFs).
    2. OCR fallback via pytesseract (for scanned/image-based PDFs where
       fitz returns no selectable text). Pages are rendered to images using
       fitz pixmaps — no Poppler dependency required on Windows.

Dependencies:
    - fitz (PyMuPDF): High-performance PDF parsing and page rendering.
    - pytesseract: Python wrapper around the Tesseract-OCR engine.
    - PIL (Pillow): Used to convert fitz pixmap bytes into images for OCR.
"""
import io
import os

import fitz        # PyMuPDF — PDF parsing and page rendering
import pytesseract # Tesseract OCR wrapper
from PIL import Image  # Required to pass rendered pages to pytesseract

# ── Tesseract binary configuration (Windows) ──────────────────────────────
# pytesseract needs to know where the Tesseract executable lives.
# The default Windows installer path is used here; update if installed elsewhere.
_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH
# If not found at the hardcoded path, pytesseract will try the system PATH.
# If Tesseract is not installed at all, OCR will raise a clear error at runtime.

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
        # ── Stage 1: Native text extraction (fast path) ──────────────────
        # Works for any digitally created PDF (Word exports, browser saves, etc.)
        native_text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                native_text += page.get_text()

        if native_text.strip():
            # Text found — return immediately without OCR overhead
            return native_text

        # ── Stage 2: OCR fallback (for scanned / image-based PDFs) ───────
        # Render each page as a high-resolution image using fitz, then run
        # Tesseract OCR on it. No Poppler dependency needed on Windows.
        ocr_text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                # 2× zoom → ~144 DPI; better accuracy than default 72 DPI
                matrix = fitz.Matrix(2, 2)
                pixmap = page.get_pixmap(matrix=matrix)

                # Convert raw PNG bytes from the pixmap into a PIL Image
                img = Image.open(io.BytesIO(pixmap.tobytes("png")))

                # Run Tesseract and accumulate text from each page
                ocr_text += pytesseract.image_to_string(img)

        return ocr_text

    else:
        # Fallback for unsupported formats (ideally intercepted by earlier validation)
        return ""
