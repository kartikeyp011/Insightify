"""
Utility for chunking large text documents into manageable semantic pieces.

This module provides functions to split large bodies of text into smaller, overlapping
chunks suitable for generating vector embeddings.

Dependencies:
    - langchain.text_splitter: Used for robust text tokenization and chunking.
"""
def split_text_into_chunks(text: str, strategy: str = "Large Chunking (1200, overlap 200)") -> list[str]:
    """
    Breaks a large string of text into smaller chunks based on the specified strategy.
    
    Supported Strategies:
    - Large Chunking (1200, overlap 200)
    - Sentence Based
    - Token Based
    - Paragraph Based
    - Semantic Chunking
    """
    if strategy == "Large Chunking (1200, overlap 200)":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        return splitter.split_text(text)

    elif strategy == "Sentence Based":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        # Split on sentence terminals first, then fallback to spaces if huge
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
            chunk_size=400,
            chunk_overlap=50
        )
        return splitter.split_text(text)

    elif strategy == "Token Based":
        from langchain_text_splitters import TokenTextSplitter
        # Uses tiktoken underneath (default cl100k_base or gpt2)
        splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=20)
        return splitter.split_text(text)

    elif strategy == "Paragraph Based":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=800,
            chunk_overlap=100
        )
        return splitter.split_text(text)

    elif strategy == "Semantic Chunking":
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Initialize an explicit embedding model for sentence boundaries
        print("[CHUNKER] Initializing HuggingFace Embeddings for Semantic Chunking...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        # Split semantically by grouping sentences with high cosine similarity
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        return splitter.split_text(text)

    else:
        # Fallback
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=75)
        return splitter.split_text(text)