import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ✅ Load Gemini API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Paths to store FAISS index and chunk metadata
INDEX_PATH = "backend/vectorstore/faiss_index"
META_PATH = "backend/vectorstore/chunk_texts.pkl"

def embed_and_store_chunks(chunks: list[str]):
    """
    Embeds a list of text chunks using Gemini and stores them in a FAISS index.
    Also saves chunk metadata for future reference.
    """

    print(f"[INFO] Embedding {len(chunks)} chunks using Gemini...")

    vectors = []
    cleaned_chunks = []

    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue  # Skip empty chunks

        try:
            # ✅ Use genai.embed_content (not model.embed_content)
            response = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )

            # ✅ Append the embedding vector
            vectors.append(np.array(response['embedding'], dtype='float32'))
            cleaned_chunks.append(chunk)

        except Exception as e:
            print(f"[WARN] Chunk #{idx+1} failed → {e}")
            print("Partial content:", repr(chunk[:100]))

    # ✅ Ensure at least one successful embedding
    if not vectors:
        raise ValueError("[ERROR] Gemini failed to embed all chunks. No vectors created.")

    # ✅ Convert to NumPy array
    embeddings_array = np.array(vectors, dtype='float32')

    # ✅ Create FAISS index and add vectors
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)

    # ✅ Save FAISS index and chunk metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(cleaned_chunks, f)

    print(f"[SUCCESS] FAISS index saved with {len(cleaned_chunks)} vectors.")