"""
Utility for chunking large text documents into manageable semantic pieces.

This module provides functions to split large bodies of text into smaller, overlapping
chunks suitable for generating vector embeddings.

Dependencies:
    - langchain.text_splitter: Used for robust text tokenization and chunking.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Breaks a large string of text into smaller, overlapping chunks.

    This function uses LangChain's RecursiveCharacterTextSplitter to ensure chunks
    are split on natural boundaries (like paragraphs or sentences) while maintaining
    a specified overlap to prevent loss of context across boundaries.

    Args:
        text (str): The large body of text to be split.
        chunk_size (int, optional): The maximum number of characters per chunk. Defaults to 800.
        overlap (int, optional): The number of characters to overlap between sequential chunks. Defaults to 150.

    Returns:
        list[str]: A list of text chunks.

    Example:
        chunks = split_text_into_chunks("A very long text...", chunk_size=500, overlap=50)
        # chunks => ["A very long...", "long text..."]
    """
    # Initialize the LangChain splitter with configured boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)