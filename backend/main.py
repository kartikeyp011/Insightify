"""
Entry point for the Smart Research Assistant FastAPI application.

This module initializes the FastAPI server, configures CORS middleware,
and mounts the API routers for uploading, asking questions, and challenges.

Components:
    read_root: Returns a simple health check message.

Dependencies:
    - fastapi: Used for the web framework.
    - routers: Local module containing API endpoint definitions.
"""
import sys
import os
# Ensure the current directory is in the path to allow absolute imports within the backend.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, ask, challenge

# ── App Initialization ─────────────────────────────────────────

# Initialize FastAPI app with metadata for the auto-generated docs.
app = FastAPI(
    title="Smart Research Assistant API",
    description="Backend service for document-based Q&A and reasoning",
    version="1.0.0"
)

# ── Middleware ──────────────────────────────────────────────────

# Configure CORS: Allow frontend (Streamlit or others) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # NOTE: You can restrict this to localhost or frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routing ─────────────────────────────────────────────────────

# Mount API routes under /api prefix for better organization.
app.include_router(upload.router, prefix="/api")
app.include_router(ask.router, prefix="/api")
app.include_router(challenge.router, prefix="/api")

# ── Root Route ──────────────────────────────────────────────────

@app.get("/")
def read_root() -> dict:
    """
    Returns a simple health check message verifying the API is running.

    Returns:
        dict: A dictionary containing a status message.
    """
    return {"msg": "Smart Assistant API is running"}
