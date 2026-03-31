"""Embedding generation and management module."""

import logging
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and manage embeddings for text chunks."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cpu"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
            device: Device to load model on ("cpu" or "cuda").
            
        Raises:
            ValueError: If model_name is empty.
            RuntimeError: If model cannot be loaded.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        self.model_name = model_name
        self.device = device
        
        try:
            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            self.model = SentenceTransformer(
                model_name_or_path=model_name,
                device=device
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of text strings.
        
        Returns:
            Numpy array of embeddings with shape (n, embedding_dim).
            
        Raises:
            ValueError: If texts is empty or invalid.
            RuntimeError: If encoding fails.
        """
        if isinstance(texts, str):
            if not texts:
                raise ValueError("Text cannot be empty")
            texts = [texts]
        elif isinstance(texts, list):
            if not texts:
                raise ValueError("Text list cannot be empty")
        else:
            raise ValueError("texts must be a string or list of strings")
        
        try:
            embeddings = self.model.encode(texts)
            logger.debug(f"Encoded {len(texts)} text(s) into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise RuntimeError(f"Failed to encode texts: {e}")
    
    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Encode sentence chunks and add embeddings to each chunk.
        
        Modifies chunks in-place by adding "embedding" key to each chunk.
        
        Args:
            chunks: List of chunk dictionaries (from pdf_extractor.create_chunks_from_sentences).
        
        Returns:
            The same chunks list with embeddings added.
            
        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("chunks cannot be empty")
        
        # Extract texts from chunks
        texts = [chunk["sentence_chunk"] for chunk in chunks]
        
        try:
            logger.info(f"Encoding {len(chunks)} chunks...")
            embeddings = self.encode(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            logger.info(f"Successfully encoded {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            raise


def save_chunks_to_csv(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save chunks with embeddings to CSV file.
    
    Args:
        chunks: List of chunk dictionaries with embeddings.
        output_path: Path to save the CSV file.
        
    Raises:
        ValueError: If chunks is empty.
        IOError: If file cannot be written.
    """
    if not chunks:
        raise ValueError("chunks cannot be empty")
    
    try:
        df = pd.DataFrame(chunks)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        logger.error(f"Error saving chunks to CSV: {e}")
        raise IOError(f"Failed to save chunks to CSV: {e}")


def load_chunks_from_csv(csv_path: str) -> tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load chunks and embeddings from CSV file.
    
    Converts embedding strings back to numpy arrays.
    
    Args:
        csv_path: Path to the CSV file with chunks and embeddings.
    
    Returns:
        Tuple of (chunks_list, embeddings_array) where:
            - chunks_list: List of chunk dictionaries
            - embeddings_array: Numpy array of embeddings (n, embedding_dim)
            
    Raises:
        FileNotFoundError: If CSV file not found.
        ValueError: If CSV format is invalid.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} chunks from {csv_path}")
        
        # Convert embedding column from string back to numpy array
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ")
            )
        else:
            raise ValueError("CSV missing 'embedding' column")
        
        # Convert to list of dicts
        chunks = df.to_dict(orient="records")
        
        # Extract embeddings as numpy array
        embeddings = np.array(df["embedding"].tolist())
        
        logger.debug(f"Embeddings shape: {embeddings.shape}")
        return chunks, embeddings
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading chunks from CSV: {e}")
        raise ValueError(f"Failed to load chunks from CSV: {e}")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length (L2 normalization).
    
    Used for cosine similarity search in FAISS.
    
    Args:
        embeddings: Embeddings array of shape (n, embedding_dim).
    
    Returns:
        L2 normalized embeddings.
        
    Raises:
        ValueError: If embeddings is empty or has wrong shape.
    """
    if embeddings.size == 0:
        raise ValueError("embeddings cannot be empty")
    
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    
    try:
        # L2 normalization
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norm + 1e-8)  # Add small epsilon to avoid division by zero
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing embeddings: {e}")
        raise


def embeddings_to_tensor(
    embeddings: np.ndarray,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Convert numpy embeddings to torch tensor.
    
    Args:
        embeddings: Numpy array of embeddings.
        device: Device to move tensor to ("cpu" or "cuda").
    
    Returns:
        Torch tensor on specified device (float32).
        
    Raises:
        ValueError: If embeddings is empty.
    """
    if embeddings.size == 0:
        raise ValueError("embeddings cannot be empty")
    
    try:
        # Convert to float32 for efficiency
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        logger.debug(f"Converted embeddings to tensor on {device}: shape {tensor.shape}")
        return tensor
    except Exception as e:
        logger.error(f"Error converting embeddings to tensor: {e}")
        raise
