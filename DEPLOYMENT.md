# Production Deployment Guide

This guide provides instructions for deploying the RAG Annual Result Analyzer in production environments.

## Pre-Deployment Checklist

- [ ] Dependencies installed: `pip install -r requiremets.txt`
- [ ] API keys configured: `GROQ_API_KEY` environment variable set
- [ ] Embeddings pre-computed: `python main.py embed`
- [ ] Test query executed: `python main.py query "test"`
- [ ] Error handling verified with `-v` flag
- [ ] Performance tested and acceptable
- [ ] Logs checked for any warnings

## Deployment Options

### Option 1: Direct Python (Simplest)

**For small deployments or quick setup:**

```bash
# 1. Clone/setup project
cd /path/to/rag-analyzer

# 2. Install dependencies
pip install -r requiremets.txt

# 3. Pre-compute embeddings (one-time)
python main.py embed

# 4. Use interactively or in scripts
python main.py query "Your question?"
```

**Pros:** Simple, no overhead
**Cons:** Requires Python environment, manual scaling

### Option 2: Docker Container (Recommended)

**For reliable, reproducible deployments:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requiremets.txt .
RUN pip install --no-cache-dir -r requiremets.txt

# Copy application
COPY . .

# Pre-compute embeddings (optional, can mount as volume)
# RUN python main.py embed

ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV PYTHONUNBUFFERED=1

# Run interactive mode
CMD ["python", "main.py", "interactive"]
```

**Build and run:**
```bash
# Build image
docker build -t rag-analyzer:1.0 .

# Run container with API key
docker run -e GROQ_API_KEY="your-key" rag-analyzer:1.0

# Or with persistent embeddings volume
docker run \
  -e GROQ_API_KEY="your-key" \
  -v rag_data:/app/Data \
  rag-analyzer:1.0
```

**Pros:** Reproducible, isolated, scalable
**Cons:** Requires Docker

### Option 3: Web API (Flask/FastAPI)

**For REST API access:**

Create `api.py`:
```python
from flask import Flask, request, jsonify
from main import load_rag_pipeline
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Load pipeline once at startup
pipeline = load_rag_pipeline()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query")
    
    if not query_text:
        return jsonify({"error": "query required"}), 400
    
    try:
        result = pipeline.query(query_text)
        return jsonify({
            "query": result["query"],
            "answer": result["answer"],
            "chunks_retrieved": len(result["retrieved_chunks"])
        })
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
```

**Run API:**
```bash
export GROQ_API_KEY="your-key"
python api.py
```

**Test API:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is TCS strategy?"}'
```

### Option 4: AWS Lambda

**For serverless deployments:**

```python
# lambda_handler.py
import json
import os
from main import load_rag_pipeline

# Load pipeline at module level (reuses across invocations)
pipeline = load_rag_pipeline()

def lambda_handler(event, context):
    try:
        query = event.get("query")
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "query required"})
            }
        
        result = pipeline.query(query)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "query": result["query"],
                "answer": result["answer"],
                "confidence": result.get("answer_json", {}).get("confidence")
            })
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
```

**Deploy to AWS Lambda:**
```bash
# Create deployment package
pip install -t package -r requiremets.txt
cd package && zip -r ../deployment.zip . && cd ..
zip deployment.zip lambda_handler.py main.py src/

# Create Lambda function
aws lambda create-function \
  --function-name rag-analyzer \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://deployment.zip \
  --environment Variables={GROQ_API_KEY=your-key} \
  --timeout 60
```

## Environment Configuration

### Environment Variables

```bash
# Required
export GROQ_API_KEY="your-groq-api-key"

# Optional (defaults shown)
export RAG_PDF_PATH="Data/TCS-annual-report-2024-2025.pdf"
export RAG_EMBEDDINGS_PATH="Data/embedding.csv"
export RAG_TOP_K="8"
export RAG_DEVICE="cpu"  # or "cuda"
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Using .env File

Create `.env`:
```
GROQ_API_KEY=your-key-here
RAG_DEVICE=cpu
LOG_LEVEL=INFO
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Performance Optimization

### 1. GPU Acceleration

```bash
# Install GPU version
pip install faiss-gpu torch-cuda

# Use in code
config.embedding.device = "cuda"
```

**Expected speedup:** 5-10x for embeddings, 2-3x for queries

### 2. Model Optimization

```python
# Faster model for speed
config.embedding.model_name = "all-MiniLM-L6-v2"

# Better model for quality
config.embedding.model_name = "all-mpnet-base-v2"
```

### 3. Result Caching

```python
import functools
import hashlib

@functools.lru_cache(maxsize=1000)
def cached_query(query_hash):
    # Query logic
    pass

def query_cached(query, pipeline):
    q_hash = hashlib.md5(query.encode()).hexdigest()
    return cached_query(q_hash)
```

### 4. Batch Processing

```python
queries = ["Q1", "Q2", "Q3"]
results = []

for query in queries:
    result = pipeline.query(query)
    results.append(result)

# Process all at once instead of sequential
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

### Metrics to Monitor

1. **Query Performance**
   - Response time
   - Token usage
   - Error rate

2. **System Health**
   - Memory usage
   - CPU/GPU utilization
   - Model loading time

3. **Data Quality**
   - Retrieval score distribution
   - Query coverage
   - Answer quality (if available)

### Example Monitoring Script

```python
#!/usr/bin/env python
"""Monitor system performance."""

import time
import psutil
import logging

def monitor_query(pipeline, query):
    """Monitor metrics for a single query."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        result = pipeline.query(query)
        elapsed = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        logging.info({
            "query": query[:50],
            "response_time_s": elapsed,
            "memory_delta_mb": end_memory - start_memory,
            "chunks_retrieved": len(result["retrieved_chunks"]),
            "status": "success"
        })
        
        return result
    except Exception as e:
        logging.error({
            "query": query[:50],
            "response_time_s": time.time() - start_time,
            "status": "error",
            "error": str(e)
        })
        raise
```

## Security Considerations

### API Key Management

```python
# BAD: Hard-coded key
API_KEY = "sk-..."

# GOOD: Environment variable
import os
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set")

# BETTER: AWS Secrets Manager
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='rag-analyzer/groq-key')
API_KEY = secret['SecretString']
```

### Input Validation

```python
def query(self, query: str) -> Dict[str, Any]:
    # Validate input
    if not isinstance(query, str):
        raise ValueError("query must be string")
    if len(query) > 1000:
        raise ValueError("query too long")
    if not query.strip():
        raise ValueError("query cannot be empty")
    
    # Process
    return self._process_query(query)
```

### Rate Limiting

```python
from functools import wraps
import time

def rate_limit(max_per_minute=60):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(max_per_minute=30)
def query(self, query: str):
    return self._query(query)
```

## Disaster Recovery

### Backup Embeddings

```bash
# Regular backup
cp Data/embedding.csv Data/embedding.csv.backup.$(date +%Y%m%d)

# Or use version control
git add Data/embedding.csv
git commit -m "Update embeddings"
```

### Re-create Embeddings

```bash
# If embeddings are lost
python main.py embed

# Verify integrity
python -c "
import pandas as pd
df = pd.read_csv('Data/embedding.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
"
```

### Health Checks

```python
def health_check():
    """Verify system is operational."""
    try:
        # Check embeddings file
        import os
        if not os.path.exists("Data/embedding.csv"):
            return False, "Embeddings not found"
        
        # Check API key
        import os
        if not os.getenv("GROQ_API_KEY"):
            return False, "API key not set"
        
        # Test quick query
        pipeline = load_rag_pipeline()
        result = pipeline.query("test")
        if not result.get("answer"):
            return False, "Query failed"
        
        return True, "Healthy"
    except Exception as e:
        return False, str(e)

# Use in deployment
if __name__ == "__main__":
    healthy, msg = health_check()
    print(f"Health: {msg}")
    exit(0 if healthy else 1)
```

## Load Testing

```python
#!/usr/bin/env python
"""Load test the RAG system."""

import time
import concurrent.futures
import statistics

def load_test(num_queries=10, num_workers=4):
    from main import load_rag_pipeline
    
    queries = [
        "What is the revenue?",
        "What is the strategy?",
        "What are the challenges?",
    ] * (num_queries // 3)
    
    pipeline = load_rag_pipeline()
    times = []
    errors = 0
    
    def query(q):
        try:
            start = time.time()
            pipeline.query(q)
            return time.time() - start
        except Exception as e:
            nonlocal errors
            errors += 1
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(query, queries)
        times = [t for t in results if t is not None]
    
    print(f"Total queries: {num_queries}")
    print(f"Errors: {errors}")
    print(f"Average time: {statistics.mean(times):.2f}s")
    print(f"Median time: {statistics.median(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")
    print(f"Min time: {min(times):.2f}s")

if __name__ == "__main__":
    load_test(num_queries=20, num_workers=4)
```

## Cost Optimization

### LLM API Costs

```python
# Monitor token usage
from groq import Groq

client = Groq(api_key="...")
response = client.chat.completions.create(...)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

### Reduce Costs

1. **Use faster model:** `llama-3.3-70b-versatile` vs `gpt-4`
2. **Reduce top_k:** Retrieve fewer chunks
3. **Shorter prompts:** More efficient context formatting
4. **Batch queries:** Process multiple queries together

## Rollback Plan

```bash
# Keep git history
git log --oneline

# Rollback to previous version
git checkout abc123 -- main.py src/

# Or restore from backup
cp main.py.backup main.py
```

## Troubleshooting Checklist

| Issue | Solution |
|-------|----------|
| "API key not found" | Check `GROQ_API_KEY` environment variable |
| "Module not found" | Run `pip install -r requiremets.txt` |
| "Embeddings not found" | Run `python main.py embed` |
| "Out of memory" | Use smaller model or reduce chunk size |
| "Slow queries" | Use GPU or reduce top_k results |
| "API rate limit" | Add retry logic and rate limiting |
| "Poor answer quality" | Increase top_k or try better model |

## Maintenance

### Regular Tasks

- **Weekly:** Check logs for errors
- **Weekly:** Test queries with new documents if added
- **Monthly:** Review performance metrics
- **Monthly:** Update dependencies: `pip list --outdated`
- **Quarterly:** Update models to latest versions
- **Annually:** Security audit and compliance check

## Support and Escalation

1. Check logs: `python main.py query "test" -v`
2. Review DEVELOPMENT.md for architecture
3. Review MIGRATION.md for common issues
4. Check Groq API status page
5. Verify all dependencies are installed correctly
