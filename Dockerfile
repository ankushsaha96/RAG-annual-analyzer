# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for macOS/Linux compatibility
ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Set work directory
WORKDIR /app

# Install system dependencies (e.g., for building sentence-transformers or parsing PDFs if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the API port
EXPOSE 8000

# Start FastAPI server, bind to Render's dynamic $PORT (fallback to 8000)
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
