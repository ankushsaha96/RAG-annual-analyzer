#!/usr/bin/env python
"""Test RAG pipeline creation."""

import sys
import warnings
warnings.filterwarnings('ignore')

try:
    print("Step 1: Check embeddings file...", flush=True)
    from pathlib import Path
    from src.config import get_config
    config = get_config()
    
    embeddings_path = Path(config.storage.embeddings_output_path)
    if not embeddings_path.exists():
        print(f"  ⚠ Embeddings not found at {embeddings_path}")
        print("  Run: python main.py embed")
        sys.exit(1)
    print(f"  ✓ Found embeddings file\n", flush=True)
    
    print("Step 2: Load embeddings CSV...", flush=True)
    from src.embedding import load_chunks_from_csv, normalize_embeddings
    import numpy as np
    chunks, embeddings = load_chunks_from_csv(str(embeddings_path))
    print(f"  ✓ Loaded {len(chunks)} chunks, shape: {embeddings.shape}\n", flush=True)
    
    print("Step 3: Normalize embeddings...", flush=True)
    embeddings = normalize_embeddings(embeddings)
    embeddings = embeddings.astype(np.float32)
    print(f"  ✓ Normalized\n", flush=True)
    
    print("Step 4: Load embedding model...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(config.embedding.model_name)
    print("  ✓ Model loaded\n", flush=True)
    
    print("Step 5: Create FAISS retriever...", flush=True)
    from src.rag import FAISSRetriever
    retriever = FAISSRetriever(embeddings)
    print("  ✓ FAISS index created\n", flush=True)
    
    print("Step 6: Initialize RAG pipeline...", flush=True)
    import os
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("  ⚠ GROQ_API_KEY not set - skipping pipeline init")
        print("  Run: export GROQ_API_KEY='your-key'\n")
    else:
        from src.rag import RAGPipeline
        pipeline = RAGPipeline(
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            groq_api_key=groq_key,
            top_k=8
        )
        print("  ✓ RAG pipeline created\n", flush=True)
        
        print("Step 7: Test retrieve...", flush=True)
        test_query = "What is the strategy?"
        results = pipeline.retrieve(test_query)
        print(f"  ✓ Retrieved {len(results)} chunks\n", flush=True)
    
    print("✓ All RAG pipeline tests passed!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
