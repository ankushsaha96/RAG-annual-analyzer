"""RAG (Retrieval Augmented Generation) module for answering queries."""

# ===== MEMORY AND SAFETY FIXES FOR macOS =====
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import json
import logging
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import faiss
from groq import Groq

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based retriever for semantic similarity search."""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Initialize FAISS retriever with embeddings.
        
        Args:
            embeddings: Normalized embeddings array of shape (n, embedding_dim).
                       Must be float32 and L2 normalized for cosine similarity.
        
        Raises:
            ValueError: If embeddings is empty or invalid shape.
        """
        if embeddings.size == 0:
            raise ValueError("embeddings cannot be empty")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
        
        try:
            # IndexFlatIP uses inner product (= cosine similarity for normalized vectors)
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index created with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k similar embeddings.
        
        Args:
            query_embedding: Query embedding of shape (1, embedding_dim).
                            Must be float32 and L2 normalized.
            k: Number of top results to return.
        
        Returns:
            Tuple of (scores, indices) where:
                - scores: Array of similarity scores (higher is better)
                - indices: Array of indices into the embedding collection
                
        Raises:
            ValueError: If query_embedding is invalid.
        """
        if query_embedding.size == 0:
            raise ValueError("query_embedding cannot be empty")
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim != 2:
            query_embedding = query_embedding.reshape(1, -1)
        
        try:
            scores, indices = self.index.search(query_embedding, k)
            logger.debug(f"Search returned {len(indices[0])} results")
            return scores, indices
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            raise


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        embedding_model: Any,
        groq_api_key: str,
        llm_model: str = "llama-3.3-70b-versatile",
        tokenizer_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        top_k: int = 8
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            chunks: List of text chunks with metadata.
            embeddings: Pre-computed embeddings for chunks (must be L2 normalized).
            embedding_model: SentenceTransformer model for encoding queries.
            groq_api_key: API key for Groq service.
            llm_model: Name of the LLM to use via Groq.
            tokenizer_model: Name of the tokenizer model.
            top_k: Number of top chunks to retrieve.
            
        Raises:
            ValueError: If inputs are invalid.
        """
        if not chunks:
            raise ValueError("chunks cannot be empty")
        if embeddings.size == 0:
            raise ValueError("embeddings cannot be empty")
        if not groq_api_key:
            raise ValueError("groq_api_key cannot be empty")
        
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.tokenizer = None
        
        try:
            self.retriever = FAISSRetriever(embeddings)
            self.groq_client = Groq(api_key=groq_api_key)
            self.llm_model = llm_model
            
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Query string.
            k: Number of chunks to retrieve (defaults to self.top_k).
        
        Returns:
            List of retrieved chunks sorted by relevance.
            
        Raises:
            ValueError: If query is empty.
        """
        if not query:
            raise ValueError("query cannot be empty")
        
        k = k or self.top_k
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query]).astype(np.float32)
            
            # Normalize
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / (norm + 1e-8)
            
            # Search
            scores, indices = self.retriever.search(query_embedding, k)
            
            # Return chunks with scores
            retrieved_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                chunk = self.chunks[idx].copy()
                chunk["retrieval_score"] = float(score)
                retrieved_chunks.append(chunk)
            
            logger.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            chunks: List of retrieved chunks.
        
        Returns:
            Formatted context string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("sentence_chunk", "")
            page = chunk.get("page_number", "?")
            context_parts.append(f"{i}. [Page {page}] {text}")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create formatted prompt for LLM.
        
        Args:
            query: User query.
            context: Retrieved context.
        
        Returns:
            Formatted prompt string.
        """
        base_prompt = """You must answer strictly using the provided context.

Context:
{context}

Question:
{query}

Rules:
	•	Every part of the answer MUST be directly supported by the source
	•	Do NOT infer or assume relationships (e.g., revenue ≠ margin unless explicitly stated)
	•	If any part of the question is not explicitly answered → say "Not explicitly stated in context"
	•	Include numerical values if present (%, basis points, etc.)
	•	Ensure answer is complete (cause + effect if applicable)
	•	In the source mention which part of the context you are referring to

Return JSON:
{{"answer": "…", "confidence": "high/medium/low", "source": ""}}"""
        
        base_prompt = base_prompt.format(context=context, query=query)
        return base_prompt
    
    def generate(self, query: str, retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate answer to query using retrieved context.
        
        Args:
            query: Query string.
            retrieved_chunks: Pre-retrieved chunks (if None, will retrieve).
        
        Returns:
            Generated answer from LLM.
            
        Raises:
            ValueError: If query is empty.
            RuntimeError: If LLM generation fails.
        """
        if not query:
            raise ValueError("query cannot be empty")
        
        try:
            # Retrieve if not already done
            if retrieved_chunks is None:
                retrieved_chunks = self.retrieve(query)
            
            if not retrieved_chunks:
                logger.warning("No chunks retrieved for query")
                return "No relevant information found in the document."
            
            # Format context
            context = self.format_context(retrieved_chunks)
            
            # Create prompt
            prompt = self.create_prompt(query, context)
            
            # Generate answer
            logger.info(f"Generating answer for query: {query[:100]}...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
            )
            
            answer = chat_completion.choices[0].message.content
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}")
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        End-to-end query: retrieve and generate.
        
        Args:
            query: Query string.
        
        Returns:
            Dictionary with:
                - query: Original query
                - answer: Generated answer
                - retrieved_chunks: List of retrieved chunks
                - answer_json: Parsed JSON response if valid
        """
        if not query:
            raise ValueError("query cannot be empty")
        
        try:
            # Retrieve
            retrieved_chunks = self.retrieve(query)
            
            # Generate
            answer = self.generate(query, retrieved_chunks)
            
            # Try to parse as JSON
            answer_json = None
            try:
                answer_json = json.loads(answer)
            except json.JSONDecodeError:
                logger.debug("Answer is not valid JSON")
            
            result = {
                "query": query,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "answer_json": answer_json
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            raise
