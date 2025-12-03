# 📊 RAG Financial Assistant

A production-ready Retrieval-Augmented Generation (RAG) system designed for financial document analysis with hybrid search, GPU acceleration, and infinite conversation memory.

## 🌟 Key Features

- **Hybrid Search**: Combines dense (BGE-M3) and sparse (lexical) embeddings for optimal retrieval
- **Cross-Encoder Reranking**: BGE-reranker-v2-m3 for fine-grained relevance scoring
- **Infinite Context**: 3-3 memory mechanism (summaries + checkpoints) for unlimited conversation history
- **GPU Acceleration**: CUDA-optimized embedding and reranking
- **Dual LLM Support**: Gemini 2.0 Flash (primary) + Llama 3.2 3B (fallback)
- **Real-Time Streaming**: Server-Sent Events (SSE) for responsive chat
- **Production-Ready**: Docker Compose orchestration with health checks and monitoring

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│   PostgreSQL   │
│     UI      │     │   Gateway    │     │   (Metadata)   │
└─────────────┘     └──────────────┘     └────────────────┘
                           │
                           ├────────▶ Milvus (Vectors)
                           │
                           ├────────▶ BGE-M3 (Embeddings)
                           │
                           ├────────▶ BGE-Reranker (Rerank)
                           │
                           └────────▶ Gemini / Ollama (LLM)
```

---

## 📋 Prerequisites

### Required:
- **Docker** 20.10+
- **Docker Compose** 2.0+
- **NVIDIA GPU** with CUDA support
- **nvidia-container-toolkit**

### Optional:
- **Gemini API Key** (recommended for production)
- 16GB+ VRAM (for full model stack)

---

## 🚀 Quick Start

### 1. Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
make install-nvidia-toolkit

# Or manually:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Gemini API key (optional but recommended)
nano .env

# Example:
# GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Launch System

```bash
# Build and start all services
make quickstart

# Or step by step:
make build
make up
make pull-models  # Download Llama 3.2 3B
```

### 4. Access UI

Open browser to: **http://localhost:8501**

---

## 📁 Project Structure

```
rag/
├── api/                      # FastAPI Backend
│   ├── app/
│   │   ├── database.py       # PostgreSQL ORM models
│   │   ├── ingest.py         # Document ingestion & embedding
│   │   ├── rag.py            # Hybrid search & generation
│   │   ├── main.py           # FastAPI app entry point
│   │   └── __init__.py
│   ├── data/                 # Preloaded documents (auto-ingest)
│   ├── Dockerfile
│   └── requirements.txt
├── ui/                       # Streamlit Frontend
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml        # Service orchestration
├── Makefile                  # Utility commands
├── .env.example              # Environment template
├── .gitignore
└── README.md
```

---

## 🔌 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **UI** | http://localhost:8501 | Streamlit chat interface |
| **API** | http://localhost:8000 | FastAPI backend |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Attu** | http://localhost:3000 | Milvus vector DB UI |
| **MinIO Console** | http://localhost:9001 | Object storage UI |
| **PostgreSQL** | localhost:5433 | Database (user: rag_user, pass: rag_password) |
| **Ollama** | localhost:11435 | Local LLM API |

---

## 🛠️ Makefile Commands

```bash
make help           # Show all available commands
make build          # Build Docker images
make up             # Start all services
make down           # Stop all services
make restart        # Restart services
make logs           # View all logs
make logs-api       # View API logs only
make logs-ui        # View UI logs only
make clean          # Remove containers and volumes (⚠️ deletes data)
make pull-models    # Download Ollama models
make health         # Check system health
make test-db        # Test database connection
make shell-api      # Open bash in API container
make backup-db      # Backup PostgreSQL database
make restore-db     # Restore database (FILE=path/to/backup.sql)
```

---

## 📊 System Requirements

### Minimum (CPU Mode):
- 16GB RAM
- 50GB disk space
- No GPU required (set `DEVICE=cpu` in `.env`)

### Recommended (GPU Mode):
- 32GB RAM
- 16GB VRAM (NVIDIA GPU)
- 100GB disk space
- CUDA 12.1+

### VRAM Allocation (GPU Mode):
| Component | Model | VRAM |
|-----------|-------|------|
| Embedding | BGE-M3 | ~1.5 GB |
| Reranker | BGE-reranker-v2-m3 | ~1.5 GB |
| LLM (Fallback) | Llama 3.2 3B | ~2.5 GB |
| **Total** | | **~5.5 GB** |

---

## 📚 API Endpoints

### Health & System
- `GET /health` - Overall health check
- `GET /health/db` - PostgreSQL status
- `GET /health/vector-db` - Milvus status
- `GET /stats/system` - System statistics
- `GET /stats/milvus` - Vector DB stats
- `GET /system/services` - Service URLs
- `GET /features` - Enabled features

### File Management
- `POST /files/upload` - Upload document
- `GET /files` - List all files
- `GET /files/status` - Processing status
- `GET /files/{id}` - File details
- `DELETE /files/{id}` - Delete file

### Chat
- `POST /sessions` - Create session
- `GET /sessions` - List sessions
- `DELETE /sessions/{id}` - Delete session
- `GET /sessions/{id}/history` - Chat history
- `POST /chat` - Send message (streaming)

---

## 🧪 Testing

### Check System Health
```bash
# Via Makefile
make health

# Or via curl
curl http://localhost:8000/health | jq
```

### Test Database
```bash
make test-db
```

### Upload Test Document
```bash
curl -X POST "http://localhost:8000/files/upload" \
  -F "file=@test_document.pdf"
```

### Send Chat Query
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "message": "What is the revenue for Q3?",
    "use_rag": true,
    "top_k": 7
  }'
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_db

# Milvus
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_hybrid_collection

# GPU
DEVICE=cuda  # or 'cpu'
CUDA_VISIBLE_DEVICES=0

# LLM
GEMINI_API_KEY=your_key_here  # Get from: https://ai.google.dev/
OLLAMA_URL=http://ollama:11434
```

### Hybrid Search Parameters
Edit `api/app/rag.py`:
```python
# Adjust weights
alpha = 0.2  # Importance score weight
beta = 0.8   # Reranker weight
gamma = 0.2  # Importance in final ranking

# Fusion weights
WeightedRanker(0.7, 0.3)  # 0.7 dense + 0.3 sparse
```

---

## 🔍 Monitoring

### View Logs
```bash
# All services
make logs

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
docker-compose logs -f milvus
```

### Check Container Status
```bash
docker-compose ps
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## 🛡️ Troubleshooting

### Issue: API not starting
```bash
# Check logs
make logs-api

# Common fixes:
# 1. Ensure GPU is available
nvidia-smi

# 2. Check if Milvus is healthy
curl http://localhost:9091/healthz

# 3. Verify database connection
make test-db
```

### Issue: Models not loading
```bash
# Check HuggingFace cache
docker exec -it rag_api ls -la /app/.cache/huggingface

# Re-download models
docker exec -it rag_api python3 -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

### Issue: Out of memory
```bash
# Reduce batch size in api/app/rag.py:
batch_size=4  # Default: 16

# Or use CPU mode:
# Edit .env: DEVICE=cpu
```

### Issue: Milvus connection failed
```bash
# Restart Milvus stack
docker-compose restart etcd minio milvus

# Wait for health check
docker-compose ps
```

---

## 📦 Backup & Restore

### Backup Database
```bash
make backup-db
# Creates: backups/rag_db_YYYYMMDD_HHMMSS.sql
```

### Restore Database
```bash
make restore-db FILE=backups/rag_db_20240101_120000.sql
```

### Backup Volumes
```bash
# Backup all volumes
docker run --rm \
  -v rag_postgres_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres_data.tar.gz -C /data .
```

---

## 🚧 Roadmap

- [ ] Multi-GPU support
- [ ] Advanced chunk importance scoring
- [ ] Automatic summary generation with LLM
- [ ] Export chat history to PDF
- [ ] REST API authentication
- [ ] Kubernetes deployment manifests
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

---

