# Development Guide

This guide explains the architecture and how to extend the RAG system.

## Architecture Overview

```
PDF Document
    ↓
[pdf_extractor.py] Extract & Split Text
    ↓
Text Chunks (with metadata)
    ↓
[embedding.py] Generate Embeddings
    ↓
Embeddings + Chunks (saved to CSV)
    ↓
[rag.py] RAG Pipeline
    ├─ FAISSRetriever: Semantic search
    └─ RAGPipeline: Retrieve + Generate
    ↓
User Query
    ↓
Answer (from LLM)
```

## Module Design

### 1. **config.py** - Configuration Management
- Centralized configuration using dataclasses
- Easy to override settings
- Loads API keys from environment variables

**Key Classes:**
- `PDFConfig`: PDF processing settings
- `ChunkingConfig`: Text chunking parameters
- `EmbeddingConfig`: Model and device settings
- `RAGConfig`: LLM and retrieval settings
- `StorageConfig`: Output paths
- `Config`: Main configuration aggregator

**Example:**
```python
from src.config import get_config
config = get_config()
config.embedding.device = "cuda"  # Use GPU
```

### 2. **pdf_extractor.py** - PDF Processing
Handles all PDF-related operations with proper error handling.

**Key Functions:**
- `open_and_read_pdf()`: Extract text from PDF with statistics
- `add_sentences_to_pages()`: Split text into sentences
- `create_chunks_from_sentences()`: Group sentences into chunks
- `split_list()`: Utility for splitting lists

**Features:**
- Character and word count statistics
- Token estimation (1 token ≈ 4 characters)
- Configurable page offset for different document layouts
- Comprehensive error handling and logging

### 3. **embedding.py** - Embedding Generation
Manages embedding creation and persistence.

**Key Classes:**
- `EmbeddingGenerator`: Wraps SentenceTransformer with error handling
  - `.encode()`: Encode texts into embeddings
  - `.encode_chunks()`: Batch encode chunks

**Key Functions:**
- `save_chunks_to_csv()`: Persist chunks with embeddings
- `load_chunks_from_csv()`: Load from disk (handles string→array conversion)
- `normalize_embeddings()`: L2 normalization for cosine similarity
- `embeddings_to_tensor()`: Convert to PyTorch tensors

**Features:**
- Automatic model downloading
- Device management (CPU/GPU)
- Proper error handling for missing models
- Efficient batch processing

### 4. **rag.py** - RAG Pipeline
Core retrieval and generation logic.

**Key Classes:**

#### FAISSRetriever
- Wraps FAISS index for efficient similarity search
- Uses inner product (= cosine similarity for normalized vectors)
- `.search()`: Find top-k similar embeddings

#### RAGPipeline
- Orchestrates retrieval and generation
- Methods:
  - `.retrieve()`: Find relevant chunks
  - `.format_context()`: Format chunks for LLM
  - `.create_prompt()`: Create well-structured prompts
  - `.generate()`: Get LLM response
  - `.query()`: End-to-end pipeline

**Features:**
- Semantic similarity search
- Context-aware prompt engineering
- JSON response parsing
- Comprehensive error handling
- Logging at each step

### 5. **main.py** - CLI Application
Command-line interface with multiple subcommands.

**Commands:**
```bash
python main.py embed           # Create embeddings
python main.py query TEXT      # Single query
python main.py interactive     # Interactive mode
```

**Features:**
- Verbose logging with `-v` flag
- JSON output support
- Automatic embedding creation
- Error handling with exit codes

## Design Patterns Used

### 1. **Dependency Injection**
Components receive dependencies rather than creating them:
```python
pipeline = RAGPipeline(
    chunks=chunks,
    embeddings=embeddings,
    embedding_model=embedding_model,  # Injected
    groq_api_key=api_key
)
```

### 2. **Configuration Objects**
Centralized configuration using dataclasses:
```python
@dataclass
class Config:
    pdf: PDFConfig
    embedding: EmbeddingConfig
    rag: RAGConfig
```

### 3. **Error Handling**
Explicit error handling with informative messages:
```python
try:
    result = pdf_extractor.open_and_read_pdf(path)
except FileNotFoundError:
    logger.error(f"PDF file not found: {path}")
    raise
```

### 4. **Logging**
Comprehensive logging for debugging:
```python
logger.info("Processing started")
logger.debug("Detailed info")
logger.error("Error occurred")
```

### 5. **Type Hints**
Full type annotations for better IDE support:
```python
def query(self, query: str) -> Dict[str, Any]:
    """..."""
```

## Extending the System

### Add Support for New File Formats

1. Create new extractor module: `src/docx_extractor.py`
2. Implement similar interface:
```python
def extract_text(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from DOCX file."""
    # Implementation
    return pages_and_texts
```
3. Update config to support new format
4. Add CLI option for new format

### Add Different Embedding Models

The system already supports any SentenceTransformer model:

```python
config.embedding.model_name = "all-MiniLM-L6-v2"  # Faster
config.embedding.model_name = "all-mpnet-base-v2"  # Better quality
```

### Add Custom LLM Provider

Replace Groq with another provider:

```python
# In src/rag.py RAGPipeline.__init__()
# Replace:
# self.groq_client = Groq(api_key=groq_api_key)

# With your provider:
from openai import OpenAI
self.llm_client = OpenAI(api_key=api_key)
```

### Add Caching

Cache embeddings to avoid recomputation:

```python
import hashlib

def get_embeddings_cached(chunks, cache_dir="cache"):
    cache_key = hashlib.md5(
        str([c["sentence_chunk"] for c in chunks]).encode()
    ).hexdigest()
    
    cache_file = f"{cache_dir}/{cache_key}.npy"
    
    if os.path.exists(cache_file):
        return np.load(cache_file)
    
    # Generate new embeddings
    embeddings = embedder.encode_chunks(chunks)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_file, embeddings)
    return embeddings
```

### Add Result Re-ranking

Improve retrieval results by re-ranking:

```python
class RAGPipeline:
    def retrieve_and_rerank(self, query: str, k: int = 20, top_k: int = 5):
        # Get more candidates
        chunks = self.retrieve(query, k=k)
        
        # Re-rank using cross-encoder or LLM
        # ... ranking logic ...
        
        # Return top-k after reranking
        return chunks[:top_k]
```

### Add Query Expansion

Improve retrieval with query variations:

```python
def expand_query(query: str) -> List[str]:
    """Generate query variations for better retrieval."""
    return [
        query,
        query.replace("?" , ""),
        f"Explain: {query}",
        f"What is: {query}"
    ]
```

## Testing

### Unit Test Example

```python
# tests/test_pdf_extractor.py
import pytest
from src.pdf_extractor import split_list

def test_split_list():
    result = split_list([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]
    
def test_split_list_empty():
    with pytest.raises(ValueError):
        split_list([], 2)
```

### Integration Test Example

```python
# tests/test_rag_pipeline.py
def test_rag_pipeline_query(sample_chunks, sample_embeddings):
    pipeline = RAGPipeline(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        # ... other args ...
    )
    
    result = pipeline.query("test query")
    
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0
```

## Performance Optimization

### 1. GPU Support
```python
config.embedding.device = "cuda"  # Enable GPU
# Install: pip install faiss-gpu
```

### 2. Smaller Models
```python
# Faster but lower quality
config.embedding.model_name = "all-MiniLM-L6-v2"
```

### 3. Reduce Top-K
```python
# Retrieve fewer chunks
config.rag.top_k_results = 5  # Instead of 8
```

### 4. Batch Processing
```python
# Process multiple queries at once
queries = ["Q1", "Q2", "Q3"]
results = [pipeline.query(q) for q in queries]
```

## Monitoring and Debugging

### Enable Debug Logging
```bash
python main.py query "test" -v  # Verbose mode
```

### Log Inspection
Logs include:
- Model loading times
- Embedding computation time
- Retrieval scores
- LLM generation status

### Profile Performance
```python
import time

start = time.time()
result = pipeline.query("question")
elapsed = time.time() - start
print(f"Query took {elapsed:.2f}s")
```

## Production Deployment

### Environment Setup
```bash
# Create .env file
GROQ_API_KEY=your-key

# Install dependencies
pip install -r requiremets.txt

# Pre-compute embeddings
python main.py embed
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requiremets.txt .
RUN pip install -r requiremets.txt
COPY . .
ENV GROQ_API_KEY=${GROQ_API_KEY}
CMD ["python", "main.py", "interactive"]
```

### Error Handling for Production
- Implement retry logic for API calls
- Add fallback LLM providers
- Cache results for frequently asked questions
- Set query timeouts

## Contributing Guidelines

1. Write type hints for all functions
2. Add docstrings with examples
3. Include error handling
4. Add logging statements
5. Write tests for new features
6. Update README if needed

## Common Issues

### OOM (Out of Memory)
- Use smaller embedding model
- Reduce chunk size
- Use CPU instead of GPU

### Slow Inference
- Use GPU
- Reduce top_k_results
- Use faster LLM model

### Poor Answer Quality
- Increase top_k_results
- Try better embedding model
- Adjust chunk size
- Improve prompt engineering
