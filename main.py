"""Main CLI for RAG Annual Result Analyzer."""

# ===== MEMORY AND SAFETY FIXES FOR macOS =====
# These environment variables must be set BEFORE importing torch/faiss
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import faiss

from src.config import get_config
from src.pdf_extractor import (
    open_and_read_pdf,
    add_sentences_to_pages,
    create_chunks_from_sentences,
)
from src.embedding import (
    EmbeddingGenerator,
    save_chunks_to_csv,
    load_chunks_from_csv,
    normalize_embeddings,
    embeddings_to_tensor,
)
from src.rag import RAGPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set logging level based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)


def create_embeddings(config_override: Optional[dict] = None) -> str:
    """
    Create embeddings for PDF document.
    
    Steps:
    1. Extract text from PDF
    2. Split into chunks
    3. Generate embeddings
    4. Save to CSV
    
    Args:
        config_override: Optional configuration overrides.
        
    Returns:
        Path to saved embeddings CSV file.
    """
    logger.info("Starting embedding creation pipeline...")
    
    config = get_config()
    if config_override:
        for key, value in config_override.items():
            if hasattr(config.pdf, key):
                setattr(config.pdf, key, value)
    
    # Step 1: Extract PDF text
    logger.info(f"Extracting text from PDF: {config.pdf.pdf_path}")
    pages_and_texts = open_and_read_pdf(
        config.pdf.pdf_path,
        page_offset=config.pdf.page_offset
    )
    
    if not pages_and_texts:
        raise ValueError("No text extracted from PDF")
    
    logger.info(f"Extracted text from {len(pages_and_texts)} pages")
    
    # Step 2: Add sentence splits
    logger.info("Splitting text into sentences...")
    add_sentences_to_pages(pages_and_texts)
    
    # Step 3: Create chunks
    logger.info(f"Creating chunks ({config.chunking.sentence_chunk_size} sentences per chunk)...")
    pages_and_chunks = create_chunks_from_sentences(
        pages_and_texts,
        chunk_size=config.chunking.sentence_chunk_size
    )
    
    if not pages_and_chunks:
        raise ValueError("No chunks created")
    
    logger.info(f"Created {len(pages_and_chunks)} chunks")
    
    # Step 4: Generate embeddings
    logger.info("Generating embeddings...")
    embedding_gen = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device
    )
    
    pages_and_chunks = embedding_gen.encode_chunks(pages_and_chunks)
    
    # Step 5: Save to CSV
    logger.info(f"Saving embeddings to: {config.storage.embeddings_output_path}")
    save_chunks_to_csv(pages_and_chunks, config.storage.embeddings_output_path)
    
    logger.info("✓ Embedding creation completed")
    return config.storage.embeddings_output_path


def load_rag_pipeline(
    embeddings_path: Optional[str] = None,
    verbose: bool = False
) -> RAGPipeline:
    """
    Load or create RAG pipeline.
    
    Args:
        embeddings_path: Path to embeddings CSV. If None, creates new embeddings.
        verbose: Enable verbose logging.
        
    Returns:
        Initialized RAGPipeline.
    """
    setup_logging(verbose)
    config = get_config()
    
    # Load or create embeddings
    if embeddings_path is None:
        embeddings_path = config.storage.embeddings_output_path
    
    embeddings_file = Path(embeddings_path)
    
    if not embeddings_file.exists():
        logger.info(f"Embeddings file not found at {embeddings_path}")
        logger.info("Creating new embeddings...")
        create_embeddings()
    else:
        logger.info(f"Loading embeddings from {embeddings_path}")
    
    # Load chunks and embeddings
    chunks, embeddings = load_chunks_from_csv(embeddings_path)
    
    # Normalize embeddings for cosine similarity
    embeddings = normalize_embeddings(embeddings)
    embeddings = embeddings.astype(np.float32)
    
    # Create embedding model for query encoding
    embedding_model = EmbeddingGenerator(
        model_name=config.embedding.model_name,
        device=config.embedding.device
    )
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        groq_api_key=config.rag.groq_api_key,
        llm_model=config.rag.llm_model,
        tokenizer_model=config.rag.tokenizer_model,
        top_k=config.rag.top_k_results
    )
    
    logger.info("✓ RAG pipeline loaded successfully")
    return rag_pipeline


def query_rag(query: str, embeddings_path: Optional[str] = None, verbose: bool = False) -> dict:
    """
    Query the RAG pipeline.
    
    Args:
        query: Query string.
        embeddings_path: Path to embeddings CSV.
        verbose: Enable verbose logging.
        
    Returns:
        Query result dictionary.
    """
    setup_logging(verbose)
    
    rag_pipeline = load_rag_pipeline(embeddings_path, verbose)
    result = rag_pipeline.query(query)
    
    return result


def interactive_mode(embeddings_path: Optional[str] = None, verbose: bool = False):
    """
    Interactive query mode.
    
    Args:
        embeddings_path: Path to embeddings CSV.
        verbose: Enable verbose logging.
    """
    setup_logging(verbose)
    
    logger.info("Loading RAG pipeline...")
    rag_pipeline = load_rag_pipeline(embeddings_path, verbose)
    
    logger.info("\n" + "="*70)
    logger.info("RAG Annual Result Analyzer - Interactive Mode")
    logger.info("="*70)
    logger.info("Type 'exit' or 'quit' to exit\n")
    
    while True:
        try:
            query = input("Query> ").strip()
            
            if query.lower() in ["exit", "quit"]:
                logger.info("Exiting...")
                break
            
            if not query:
                continue
            
            logger.info("\nProcessing query...")
            result = rag_pipeline.query(query)
            
            logger.info("\n" + "="*70)
            logger.info("ANSWER:")
            logger.info("="*70)
            print(result["answer"])
            
            if result["answer_json"]:
                logger.info("\nPARSED JSON:")
                print(json.dumps(result["answer_json"], indent=2))
            
            logger.info(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
            logger.info("="*70 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Annual Result Analyzer - Extract embeddings and query documents"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Embedding command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Create embeddings from PDF"
    )
    embed_parser.add_argument(
        "--pdf-path",
        type=str,
        help="Path to PDF file"
    )
    embed_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG pipeline"
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Query string"
    )
    query_parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings CSV file"
    )
    query_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    query_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive query mode"
    )
    interactive_parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings CSV file"
    )
    interactive_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "embed":
            setup_logging(args.verbose)
            config_override = {}
            if args.pdf_path:
                config_override["pdf_path"] = args.pdf_path
            
            output_path = create_embeddings(config_override)
            logger.info(f"\nEmbeddings saved to: {output_path}")
        
        elif args.command == "query":
            result = query_rag(args.query, args.embeddings, args.verbose)
            
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                logger.info("\n" + "="*70)
                logger.info("ANSWER:")
                logger.info("="*70)
                print(result["answer"])
                
                if result["answer_json"]:
                    logger.info("\nPARSED JSON:")
                    print(json.dumps(result["answer_json"], indent=2))
                
                logger.info(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
                logger.info("="*70)
        
        elif args.command == "interactive":
            interactive_mode(args.embeddings, args.verbose)
        
        else:
            parser.print_help()
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
