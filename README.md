# RAG Annual Result Analyzer

A production-ready Retrieval Augmented Generation (RAG) system for analyzing annual reports and answering questions based on the document content.

## Features

- 📄 **PDF Extraction**: Efficiently extracts and processes text from PDF documents
- 🔗 **Semantic Search**: Uses FAISS for fast similarity-based document retrieval
- 🤖 **LLM Integration**: Leverages Groq's LLM API for generating contextual answers
- ⚙️ **Configuration Management**: Easy configuration for all components
- 📊 **Logging & Error Handling**: Comprehensive logging and error handling throughout
- 🎯 **Type Hints**: Full type annotations for better code quality
- 🖥️ **CLI Interface**: Command-line interface for easy interaction

## Project Structure

```
.
├── main.py                      # CLI application entry point
├── src/
│   ├── config.py               # Configuration management
│   ├── pdf_extractor.py        # PDF extraction and text processing
│   ├── embedding.py            # Embedding generation and management
│   ├── rag.py                  # RAG pipeline (retrieval + generation)
│   ├── embedder.py             # Legacy embedder (for compatibility)
│   └── tokenizer.py            # Legacy tokenizer (for compatibility)
├── Data/
│   ├── TCS-annual-report-2024-2025.pdf  # Input PDF document
│   └── embedding.csv           # Generated embeddings (created after first run)
├── Notebooks/
│   ├── Part1_embedding.ipynb   # Legacy notebook (reference)
│   └── Part2_result_generator.ipynb    # Legacy notebook (reference)
├── requiremets.txt             # Python dependencies
└── README.md                   # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requiremets.txt
```

### 2. Set Up Groq API Key

Get your API key from [Groq Console](https://console.groq.com/):

```bash
export GROQ_API_KEY="your-api-key-here"
```

Or on Windows:
```cmd
set GROQ_API_KEY=your-api-key-here
```

## Usage

### Quick Start: Create Embeddings

Generate embeddings from the PDF document:

```bash
python main.py embed
```

Options:
- `--pdf-path <path>`: Specify custom PDF path
- `-v, --verbose`: Enable verbose logging

### Query the RAG System

Ask a question about the document:

```bash
python main.py query "What was the revenue growth?"
```

Options:
- `--embeddings <path>`: Path to embeddings CSV file
- `--json`: Output as JSON
- `-v, --verbose`: Enable verbose logging

### Interactive Mode

For interactive querying:

```bash
python main.py interactive
```

This starts an interactive session where you can ask multiple questions. Type `exit` or `quit` to exit.

## Configuration

Edit `src/config.py` to customize:

- **PDFConfig**: PDF file path and page offset
- **ChunkingConfig**: Chunk size and splitting parameters
- **EmbeddingConfig**: Embedding model and device (CPU/GPU)
- **RAGConfig**: LLM model and retrieval parameters
- **StorageConfig**: Output file paths

Example:
```python
from src.config import Config

config = Config()
config.pdf.pdf_path = "path/to/your/document.pdf"
config.embedding.device = "cuda"  # Use GPU
config.rag.top_k_results = 10  # Retrieve top 10 chunks
```

## API Usage

Use the production modules programmatically:

### Extract and Embed PDF

```python
from src.pdf_extractor import open_and_read_pdf, create_chunks_from_sentences
from src.embedding import EmbeddingGenerator, save_chunks_to_csv

# Extract text
pages = open_and_read_pdf("document.pdf")

# Create chunks
chunks = create_chunks_from_sentences(pages, chunk_size=6)

# Generate embeddings
embedder = EmbeddingGenerator(device="cpu")
chunks = embedder.encode_chunks(chunks)

# Save
save_chunks_to_csv(chunks, "embeddings.csv")
```

### Query with RAG

```python
from src.embedding import load_chunks_from_csv, normalize_embeddings
from src.rag import RAGPipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Load data
chunks, embeddings = load_chunks_from_csv("embeddings.csv")
embeddings = normalize_embeddings(embeddings).astype(np.float32)

# Create pipeline
embedding_model = SentenceTransformer("all-mpnet-base-v2")
pipeline = RAGPipeline(
    chunks=chunks,
    embeddings=embeddings,
    embedding_model=embedding_model,
    groq_api_key="your-api-key",
    top_k=8
)

# Query
result = pipeline.query("Your question here?")
print(result["answer"])
```

## Module Documentation

### `config.py`
Configuration dataclasses for all components. Loads `GROQ_API_KEY` from environment.

### `pdf_extractor.py`
- `open_and_read_pdf()`: Extract text from PDF
- `create_chunks_from_sentences()`: Split text into chunks
- `add_sentences_to_pages()`: Split text into sentences

### `embedding.py`
- `EmbeddingGenerator`: Generate embeddings using SentenceTransformer
- `save_chunks_to_csv()`: Save chunks with embeddings
- `load_chunks_from_csv()`: Load chunks from CSV
- `normalize_embeddings()`: L2 normalization for cosine similarity

### `rag.py`
- `FAISSRetriever`: Semantic similarity search using FAISS
- `RAGPipeline`: End-to-end RAG pipeline (retrieve → generate)
  - `retrieve()`: Get top-k relevant chunks
  - `generate()`: Generate answer using LLM
  - `query()`: Full pipeline (retrieve + generate)

## Performance Tips

1. **GPU Support**: Install `faiss-gpu` instead of `faiss-cpu` for faster retrieval
   ```bash
   pip install faiss-gpu
   ```

2. **Larger Models**: Use better embedding models for improved quality
   ```python
   config.embedding.model_name = "all-mpnet-base-v2"  # Current default
   # Or try: "all-MiniLM-L6-v2" (faster), "all-mpnet-base-v2" (better quality)
   ```

3. **Chunk Size**: Adjust chunk size based on your document
   - Smaller chunks: More precise retrieval, more chunks to process
   - Larger chunks: Faster processing, less precise

## Troubleshooting

### ModuleNotFoundError
Make sure you're in the project root directory and dependencies are installed:
```bash
pip install -r requiremets.txt
```

### GROQ_API_KEY not found
Set the environment variable:
```bash
export GROQ_API_KEY="your-key"
```

### Out of Memory
- Reduce chunk size
- Use smaller embedding model
- Use CPU instead of GPU via config

### Slow Inference
- Check device selection (GPU recommended)
- Reduce `top_k_results` for faster retrieval
- Use faster LLM model

## Converting from Notebooks

The original notebooks have been converted to production modules:

**Part 1 (Embedding)** → `src/pdf_extractor.py` + `src/embedding.py`
- PDF extraction
- Text chunking
- Embedding generation

**Part 2 (Result Generator)** → `src/rag.py`
- FAISS indexing
- Query retrieval
- LLM-based generation

Benefits of modular approach:
✅ Reusable components
✅ Better error handling
✅ Type safety
✅ Comprehensive logging
✅ Easy testing
✅ Production-ready

## Testing

Example test queries for TCS Annual Report:

```bash
python main.py query "Which vertical contributed the highest revenue?"
python main.py query "What is the company's strategy for AI and cloud?"
python main.py query "How did margins change compared to last year?"
```

## License

This project uses libraries with different licenses:
- PyMuPDF: AGPL-3.0 (consider commercial implications)
- SentenceTransformers: Apache 2.0
- Transformers: Apache 2.0
- FAISS: MIT
- Groq Python SDK: Apache 2.0

## Contributing

Improvements welcome! Areas for enhancement:
- Add support for multiple documents
- Implement caching for embeddings
- Add batch query support
- Implement result ranking and filtering
- Add UI interface

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs with `-v` flag for debugging
3. Verify API key and dependencies are correct
