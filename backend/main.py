from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import upload, ask, challenge

# ✅ Initialize FastAPI app
app = FastAPI(
    title="Smart Research Assistant API",
    description="Backend service for document-based Q&A and reasoning",
    version="1.0.0"
)

# ✅ CORS: Allow frontend (Streamlit or others) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to localhost or frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount API routes under /api
app.include_router(upload.router, prefix="/api")
app.include_router(ask.router, prefix="/api")
app.include_router(challenge.router, prefix="/api")

# ✅ Root route (optional)
@app.get("/")
def read_root():
    return {"msg": "Smart Assistant API is running"}
