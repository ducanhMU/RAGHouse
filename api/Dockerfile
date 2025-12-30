# ============================================
# OPTIMIZED DOCKERFILE
# ============================================
# Thay vì dùng nvidia/cuda (phải tự cài python & torch rất lâu), 
# ta dùng image chính chủ của PyTorch đã có sẵn mọi thứ.
# Image này dựa trên Ubuntu, có sẵn Python 3.10+, CUDA 12.1 và PyTorch 2.2+
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential & git: cần thiết để pip build một số thư viện python extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (đã có sẵn nhưng nên update)
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements TRƯỚC (để tận dụng Docker Cache)
COPY requirements.txt .

# Install Python dependencies
# Lưu ý: requirements.txt đã bỏ 'torch' để dùng bản có sẵn của image
RUN pip install --no-cache-dir -r requirements.txt

# Download AI Models (BGE-M3 & Reranker)
# Bước này sẽ được cache lại, không phải tải lại trừ khi bạn sửa dòng này
RUN python -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
    print('Downloading BGE-M3...'); \
    BGEM3FlagModel('BAAI/bge-m3', use_fp16=True); \
    print('Downloading Reranker...'); \
    FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"

# Create directories
RUN mkdir -p /app/data /app/data/uploads /app/.cache/huggingface

# Copy application code SAU CÙNG
# (Để khi bạn sửa code, Docker không phải chạy lại các bước cài thư viện ở trên)
COPY ./app /app/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]