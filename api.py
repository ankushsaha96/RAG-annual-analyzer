import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing logic
from main import load_rag_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="RAG Annual Result Analyzer API",
    description="API for querying the RAG pipeline",
    version="1.0.0"
)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. In prod, configure to frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reference to pipeline to keep it in memory
RAG_PIPELINE = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class QueryRequest(BaseModel):
    query: str
    embeddings_path: Optional[str] = None

class ChunkResponse(BaseModel):
    text: str
    page_number: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: List[ChunkResponse]
    answer_json: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    global RAG_PIPELINE
    logger.info("Initializing RAG Pipeline on startup...")
    # Load the pipeline; this may block, but we only do it once
    try:
        # Pass None so it finds the default embeddings path from config
        RAG_PIPELINE = load_rag_pipeline(embeddings_path=None, verbose=True)
        logger.info("RAG Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")
        # Not raising here so the server can still start and show status.

@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "pipeline_loaded": RAG_PIPELINE is not None
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    global RAG_PIPELINE
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    if RAG_PIPELINE is None:
        try:
            logger.info("Pipeline not loaded. Loading now...")
            RAG_PIPELINE = load_rag_pipeline(embeddings_path=request.embeddings_path, verbose=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")

    try:
        # Run query
        result = RAG_PIPELINE.query(request.query)
        
        # Map chunks
        mapped_chunks = []
        for chunk in result.get("retrieved_chunks", []):
            mapped_chunks.append(ChunkResponse(
                text=chunk.get("text", chunk.get("sentence_chunk", "No text provided")),
                page_number=chunk.get("page_number", -1),
                score=float(chunk.get("score", 0.0))
            ))
            
        return QueryResponse(
            answer=result["answer"],
            chunks=mapped_chunks,
            answer_json=result.get("answer_json")
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
