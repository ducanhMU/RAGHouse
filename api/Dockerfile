# ============================================
# OPTIMIZED DOCKERFILE
# ============================================
# Dùng image PyTorch có sẵn CUDA
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
# Note: requirements.txt shouldn't include 'torch'
RUN pip install --no-cache-dir -r requirements.txt

# Set environment for HF cache *only* (optional)
ENV HF_HOME=/app/cache/huggingface
ENV HF_HUB_CACHE=/app/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/cache/huggingface/datasets
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Create directories for data + cache (optional)
RUN mkdir -p /app/data /app/data/uploads /app/cache/huggingface

# Copy application code last
COPY ./app /app/app

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
