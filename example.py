"""
Example: Basic embedding creation and querying.

This example shows how to use the RAG system programmatically.
"""

from src.config import get_config
from src.pdf_extractor import (
    open_and_read_pdf,
    add_sentences_to_pages,
    create_chunks_from_sentences,
)
from src.embedding import (
    EmbeddingGenerator,
    save_chunks_to_csv,
    normalize_embeddings,
)
from src.rag import RAGPipeline
import numpy as np


def example_create_embeddings():
    """Example: Create embeddings from PDF."""
    print("Creating embeddings from PDF...")
    
    config = get_config()
    
    # Extract PDF text
    pages = open_and_read_pdf(config.pdf.pdf_path, config.pdf.page_offset)
    print(f"Extracted {len(pages)} pages")
    
    # Add sentence splits
    add_sentences_to_pages(pages)
    
    # Create chunks
    chunks = create_chunks_from_sentences(pages, config.chunking.sentence_chunk_size)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    embedder = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device
    )
    chunks = embedder.encode_chunks(chunks)
    
    # Save
    save_chunks_to_csv(chunks, config.storage.embeddings_output_path)
    print(f"Saved to {config.storage.embeddings_output_path}")


def example_query():
    """Example: Query the RAG system."""
    print("\nQuerying RAG system...")
    
    from src.embedding import load_chunks_from_csv
    from sentence_transformers import SentenceTransformer
    
    config = get_config()
    
    # Load embeddings
    chunks, embeddings = load_chunks_from_csv(config.storage.embeddings_output_path)
    embeddings = normalize_embeddings(embeddings).astype(np.float32)
    
    # Create embedding model
    embedding_model = SentenceTransformer(config.embedding.model_name)
    
    # Create RAG pipeline
    pipeline = RAGPipeline(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        groq_api_key=config.rag.groq_api_key,
        llm_model=config.rag.llm_model,
        tokenizer_model=config.rag.tokenizer_model,
        top_k=config.rag.top_k_results,
    )
    
    # Ask a question
    query = "What was the revenue growth?"
    result = pipeline.query(query)
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")


if __name__ == "__main__":
    # Uncomment to run examples
    
    # Create embeddings first (if not already done)
    # example_create_embeddings()
    
    # Then query
    # example_query()
    
    print("See code above for examples. Uncomment to run.")
