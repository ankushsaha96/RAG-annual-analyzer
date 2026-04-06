import os
import asyncio
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.qdrant_service import QdrantService

# Initialize FastAPI app
app = FastAPI(
    title="FinSight AI API",
    description="AI-powered annual report analysis via Qdrant VectorDB",
    version="2.0.0"
)

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
QDRANT_SERVICE: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """Lazy-init the Qdrant service."""
    global QDRANT_SERVICE
    if QDRANT_SERVICE is None:
        QDRANT_SERVICE = QdrantService()
    return QDRANT_SERVICE


# ── Request / Response Models ─────────────────────────────

class CheckEmbeddingsRequest(BaseModel):
    company_name: str
    year: int

class CheckEmbeddingsResponse(BaseModel):
    exists: bool

class CreateEmbeddingsRequest(BaseModel):
    company_name: str
    year: int

class CreateEmbeddingsResponse(BaseModel):
    status: str
    message: str
    chunks: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    company_name: str
    year: int

class ChunkResponse(BaseModel):
    text: str
    page_number: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: List[ChunkResponse]
    answer_json: Optional[Dict[str, Any]] = None


# ── Endpoints ─────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {"status": "online"}


@app.post("/api/check-embeddings", response_model=CheckEmbeddingsResponse)
async def check_embeddings(request: CheckEmbeddingsRequest):
    """Check if embeddings for a company+year exist in Qdrant."""
    try:
        svc = get_qdrant_service()
        exists = await asyncio.to_thread(svc.check_embeddings_exist, request.company_name, request.year)
        return CheckEmbeddingsResponse(exists=exists)
    except Exception as e:
        logger.error(f"Error checking embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/create-embeddings", response_model=CreateEmbeddingsResponse)
async def create_embeddings(request: CreateEmbeddingsRequest):
    """Fetch annual report and create embeddings in Qdrant."""
    try:
        svc = get_qdrant_service()
        result = await asyncio.to_thread(svc.create_embeddings, request.company_name, request.year)
        return CreateEmbeddingsResponse(**result)
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """Query the RAG pipeline filtered by company+year."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        svc = get_qdrant_service()

        # 1. Encode the query
        query_embedding = svc.encode_query(request.query)

        # 2. Retrieve from Qdrant
        retrieved = svc.query_points(
            query_embedding=query_embedding,
            company_name=request.company_name,
            year=request.year,
            limit=10,
        )

        if not retrieved:
            return QueryResponse(
                answer="No relevant information found for this company and year.",
                chunks=[],
                answer_json=None,
            )

        # 3. Format context for LLM
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            text = chunk.get("text", "")
            page = chunk.get("page_number", "?")
            context_parts.append(f"{i}. [Page {page}] {text}")
        context = "\n".join(context_parts)

        # 4. Generate answer via Groq
        from src.config import get_config
        from groq import Groq
        import json

        config = get_config()
        groq_client = Groq(api_key=config.rag.groq_api_key)

        prompt = f"""You must answer strictly using the provided context.
This context is from the annual report of {request.company_name} for FY {request.year - 1}-{str(request.year)[-2:]}.

Context:
{context}

Question:
{request.query}

Rules:
	•	Every part of the answer MUST be directly supported by the source
	•	Do NOT infer or assume relationships (e.g., revenue ≠ margin unless explicitly stated)
	•	If any part of the question is not explicitly answered → say "Not explicitly stated in context"
	•	Include numerical values if present (%, basis points, etc.)
	•	Ensure answer is complete (cause + effect if applicable)
	•	IMPORTANT: Always cite page numbers inline using the exact format (Page X) wherever you reference data. For example: "Revenue was ₹100 crore (Page 22)."
	•	You may cite multiple pages if the answer spans across them

Return JSON:
{{"answer": "…", "confidence": "high/medium/low", "source": ""}}"""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.rag.llm_model,
        )
        answer = chat_completion.choices[0].message.content

        # Parse JSON
        answer_json = None
        try:
            answer_json = json.loads(answer)
        except json.JSONDecodeError:
            pass

        # Map chunks
        mapped_chunks = [
            ChunkResponse(
                text=c.get("text", ""),
                page_number=c.get("page_number", -1),
                score=float(c.get("score", 0.0)),
            )
            for c in retrieved
        ]

        return QueryResponse(
            answer=answer,
            chunks=mapped_chunks,
            answer_json=answer_json,
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
