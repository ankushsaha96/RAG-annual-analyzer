"""FinSight AI package."""

__version__ = "1.0.0"
__author__ = "RAG Team"

# Lazy imports — heavy modules (torch, faiss) are only loaded when accessed.
# This avoids crashing on import when only the Qdrant-based flow is needed.

def __getattr__(name):
    """Lazy import for backward compatibility."""
    _lazy_map = {
        "get_config": ("src.config", "get_config"),
        "Config": ("src.config", "Config"),
        "open_and_read_pdf": ("src.pdf_extractor", "open_and_read_pdf"),
        "create_chunks_from_sentences": ("src.pdf_extractor", "create_chunks_from_sentences"),
        "EmbeddingGenerator": ("src.embedding", "EmbeddingGenerator"),
        "load_chunks_from_csv": ("src.embedding", "load_chunks_from_csv"),
        "save_chunks_to_csv": ("src.embedding", "save_chunks_to_csv"),
        "RAGPipeline": ("src.rag", "RAGPipeline"),
        "FAISSRetriever": ("src.rag", "FAISSRetriever"),
    }
    if name in _lazy_map:
        module_path, attr = _lazy_map[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'src' has no attribute {name!r}")

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

