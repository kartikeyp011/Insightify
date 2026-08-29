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
from contextlib import asynccontextmanager
import asyncio
import time
import shutil

# ── Background Cleanup Task ─────────────────────────────────────

async def cleanup_old_sessions():
    """Background task to delete session folders older than 6 hours."""
    vectorstore_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")
    expiration_seconds = 6 * 3600
    while True:
        try:
            if os.path.exists(vectorstore_path):
                now = time.time()
                for session_dir in os.listdir(vectorstore_path):
                    dir_path = os.path.join(vectorstore_path, session_dir)
                    if os.path.isdir(dir_path):
                        mtime = os.path.getmtime(dir_path)
                        if (now - mtime) > expiration_seconds:
                            print(f"[CLEANUP] Deleting expired session directory: {session_dir}")
                            shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            print(f"[CLEANUP] Error during cleanup: {e}")
        await asyncio.sleep(15 * 60) # Run every 15 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: spawn the background cleanup task
    task = asyncio.create_task(cleanup_old_sessions())
    yield
    # Shutdown: cancel the task
    task.cancel()

# ── App Initialization ─────────────────────────────────────────

# Initialize FastAPI app with metadata for the auto-generated docs.
app = FastAPI(
    title="Smart Research Assistant API",
    description="Backend service for document-based Q&A and reasoning",
    version="1.0.0",
    lifespan=lifespan
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
