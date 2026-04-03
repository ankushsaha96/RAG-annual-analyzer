"""Configuration management for RAG Annual Result Analyzer."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
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
    
    model_name: str = "all-mpnet-base-v2"
    device: str = "cpu"  # Set to "cuda" for GPU


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
class Config:
    """Main configuration class combining all configs."""
    
    pdf: PDFConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    rag: RAGConfig
    storage: StorageConfig
    
    def __init__(self):
        """Initialize configuration with defaults."""
        self.pdf = PDFConfig()
        self.chunking = ChunkingConfig()
        self.embedding = EmbeddingConfig()
        self.rag = RAGConfig()
        self.storage = StorageConfig()


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()
