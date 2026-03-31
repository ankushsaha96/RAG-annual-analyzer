"""RAG Annual Result Analyzer package."""

__version__ = "1.0.0"
__author__ = "RAG Team"

from src.config import get_config, Config
from src.pdf_extractor import open_and_read_pdf, create_chunks_from_sentences
from src.embedding import EmbeddingGenerator, load_chunks_from_csv, save_chunks_to_csv
from src.rag import RAGPipeline, FAISSRetriever

__all__ = [
    "get_config",
    "Config",
    "open_and_read_pdf",
    "create_chunks_from_sentences",
    "EmbeddingGenerator",
    "load_chunks_from_csv",
    "save_chunks_to_csv",
    "RAGPipeline",
    "FAISSRetriever",
]
