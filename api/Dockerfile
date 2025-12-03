# API Dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download BGE-M3 and reranker models (cached in volume)
RUN python3 -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
    BGEM3FlagModel('BAAI/bge-m3', use_fp16=True); \
    FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"

# Create directories
RUN mkdir -p /app/data /app/data/uploads /app/.cache/huggingface

# Copy application code
COPY ./app /app/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]