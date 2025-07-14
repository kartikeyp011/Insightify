import numpy as np
import pickle
import faiss
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load key and configure
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

INDEX_PATH = "vectorstore/faiss_index"
CHUNKS_PATH = "vectorstore/chunk_texts.pkl"

def get_relevant_chunks(query: str, top_k: int = 4) -> list[str]:
    """
    Embeds the user query and retrieves top_k similar chunks from FAISS.
    """
    # Load FAISS index and chunks
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store not found.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        all_chunks = pickle.load(f)

    # Embed the query
    response = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_vector = np.array(response['embedding'], dtype='float32').reshape(1, -1)

    # Search top_k similar chunks
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]

    return retrieved_chunks