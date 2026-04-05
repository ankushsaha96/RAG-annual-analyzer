"""Configuration management for RAG Annual Result Analyzer."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class PDFConfig:
    """Configuration for PDF processing."""
    
    pdf_path: str = "Data/TCS-annual-report-2024-2025.pdf"
    page_offset: int = 4  # Adjust page numbers for documents starting on specific page
    

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    
    sentence_chunk_size: int = 6  # Number of sentences per chunk
    large_chunk_size: int = 100  # Alternative larger chunk size


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    
    model_name: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v1"
    nvidia_api_key: Optional[str] = None  # Set via environment variable NVIDIA_API_KEY
    device: str = "cpu"  # No longer used by NVIDIA API but kept for backward compatibility

    def __post_init__(self):
        """Load NVIDIA API key from environment if not provided."""
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not self.nvidia_api_key:
                logger = logging.getLogger(__name__)
                logger.warning("NVIDIA_API_KEY not provided. Embedding generation might fail.")


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval Augmented Generation)."""
    
    groq_api_key: Optional[str] = None  # Set via environment variable GROQ_API_KEY
    llm_model: str = "llama-3.3-70b-versatile"
    top_k_results: int = 8
    tokenizer_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    def __post_init__(self):
        """Load Groq API key from environment if not provided."""
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not provided. Set it via environment variable "
                    "or pass it to RAGConfig directly."
                )


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    
    embeddings_output_path: str = "Data/embedding.csv"
    index_output_path: str = "Data/faiss_index.faiss"


@dataclass
class QdrantConfig:
    """Configuration for Qdrant Vector Database."""
    
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection_name: str = "annual_reports"
    vector_size: int = 2048
    
    def __post_init__(self):
        """Load Qdrant credentials from environment if not provided."""
        if not self.url:
            self.url = os.getenv("QDRANT_URL")
        if not self.api_key:
            self.api_key = os.getenv("QDRANT_API_KEY")
        if not self.url or not self.api_key:
            logger = logging.getLogger(__name__)
            logger.warning("QDRANT_URL or QDRANT_API_KEY not set. Qdrant features will be unavailable.")


@dataclass
class Config:
    """Main configuration class combining all configs."""
    
    pdf: PDFConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    rag: RAGConfig
    storage: StorageConfig
    qdrant: QdrantConfig
    
    def __init__(self):
        """Initialize configuration with defaults."""
        self.pdf = PDFConfig()
        self.chunking = ChunkingConfig()
        self.embedding = EmbeddingConfig()
        self.rag = RAGConfig()
        self.storage = StorageConfig()
        self.qdrant = QdrantConfig()


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()
