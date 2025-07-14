from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size=800, overlap=150):
    """
    Breaks large text into overlapping chunks.
    - chunk_size: max characters per chunk
    - overlap: number of chars to repeat across chunks
    Returns: list of chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)