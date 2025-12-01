# 🚀 RAG V2 ULTIMATE

**Production-Ready Retrieval-Augmented Generation System**

A complete AI assistant with hybrid search, infinite context memory, and GPU acceleration.

## ✨ Key Features

- **🔍 Hybrid Search**: Combines semantic (Milvus) + keyword (PostgreSQL FTS) search with RRF fusion
- **🧠 Infinite Context**: Smart 3-3 memory architecture (Summary + Checkpoint system)
- **⚡ GPU-Accelerated**: Batch embedding, reranking, and optional local LLM
- **🌐 Dual LLM**: Gemini 2.0 Flash (primary) + Llama 3.2 (fallback)
- **📚 Document Intelligence**: PDF ingestion with deduplication
- **💾 Enterprise Database**: PostgreSQL with FTS + Milvus vector DB
- **🎨 Modern UI**: Streamlit interface with real-time status

## 🏗️ Architecture

```
┌─────────────┐
│ Streamlit UI│
└──────┬──────┘
       │
┌──────▼───────┐     ┌──────────────┐
│  FastAPI     │────▶│  PostgreSQL  │
│  Gateway     │     │  (FTS + Chat)│
└──────┬───────┘     └──────────────┘
       │
       ├────────────▶┌──────────────┐
       │             │   Milvus     │
       │             │  (Vectors)   │
       │             └──────────────┘
       │
       └────────────▶┌──────────────┐
                     │   Ollama     │
                     │ (Embedding + │
                     │   Fallback)  │
                     └──────────────┘
```

## 📋 Prerequisites

- **Docker** & **Docker Compose**
- **NVIDIA GPU** with CUDA support (12GB+ VRAM recommended)
- **nvidia-container-toolkit** installed
- **Google Gemini API Key** (free tier available)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

The script will:
- ✅ Check prerequisites (Docker, GPU, nvidia-toolkit)
- ✅ Create directory structure
- ✅ Generate `.env` file with secure passwords
- ✅ Verify all required files are present
- ✅ Build and start all services
- ✅ Wait for services to be healthy
- ✅ Display access information

### Option 2: Manual Setup

```bash
# 1. Create environment file
cp .env.example .env

# 2. Edit .env and add your Gemini API key
nano .env
# Set: GOOGLE_API_KEY=your_actual_key_here

# 3. Start all services
docker-compose up -d

# 4. View logs
docker-compose logs -f api

# 5. Stop system
docker-compose down
```

### 3. Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Main application |
| **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| **FastAPI Health** | http://localhost:8000/health | System status JSON |
| **Milvus Attu** | http://localhost:3000 | Vector DB admin |
| **MinIO Console** | http://localhost:9001 | Object storage (admin/admin) |

### 4. Initial Setup Verification

```bash
# Wait ~2 minutes for all services to initialize, then:

# Check system health
curl http://localhost:8000/health

# Expected output:
# {
#   "postgres": "ok",
#   "milvus": "ok",
#   "models": "ok",
#   "internet": "ok"
# }

# Check service status
docker-compose ps

# All services should show "Up" or "Up (healthy)"
```

## 📊 Database Schema

### Chat System (Infinite Context)

```sql
chat_sessions
├── id (UUID)
├── title
├── created_at
└── updated_at

chat_events (Append-Only Event Store)
├── id (UUID)
├── session_id (FK)
├── sequence_num (Ordered)
├── role (USER/ASSISTANT/SYSTEM)
├── content
├── event_type (NORMAL/SUMMARY/CHECKPOINT)
├── visibility (VISIBLE/HIDDEN)
└── model_used
```

### Knowledge Base

```sql
file_registry
├── id (UUID)
├── file_hash (MD5 - Unique)
├── filename
├── status (PENDING/PROCESSING/COMPLETED/FAILED)
└── meta_info (JSONB)

document_chunks
├── id (UUID)
├── file_id (FK)
├── content (TEXT)
├── search_vector (TSVECTOR - Auto-generated)
├── chunk_index
└── page_number
```

## 🎯 Usage Examples

### Chat with Documents

1. Open http://localhost:8501
2. Upload PDF files in sidebar
3. Wait for processing (green checkmark)
4. Ask questions in chat

### API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Create session
session = requests.post("http://localhost:8000/sessions", 
                       json={"title": "My Chat"}).json()

# Send message
message = requests.post(
    f"http://localhost:8000/sessions/{session['session_id']}/message",
    json={"content": "What is the revenue?", "use_rag": True}
).json()

print(message['reply'])
```

## 🔧 Configuration

### Key Environment Variables

```bash
# LLM Selection
GOOGLE_API_KEY=sk-...        # Gemini (primary)
OLLAMA_MODEL=llama3.2:3b     # Fallback

# Performance Tuning
EMBEDDING_BATCH_SIZE=64      # GPU batch size
MAX_WORKERS=4                # Concurrent workers

# RAG Settings
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
```

## 📈 Performance Tips

### For 16GB VRAM GPU

- `EMBEDDING_BATCH_SIZE=64` (optimal)
- Models loaded: nomic-embed (0.5GB) + reranker (1.5GB) + llama3.2 (2.5GB) = ~4.5GB
- Leaves 11GB for KV cache and batching

### For 12GB VRAM GPU

- `EMBEDDING_BATCH_SIZE=32`
- Still highly performant

### For CPU-Only

- Set `EMBEDDING_BATCH_SIZE=8`
- Expect slower ingestion (5-10x)
- Chat speed unaffected (uses Gemini API)

## 🐛 Troubleshooting

### `init-db.sql` Not Found Error

```bash
# Verify file location (must be in project root)
ls -la init-db.sql

# Should be alongside docker-compose.yml, not in api/ or ui/

# Correct structure:
# rag-v2-ultimate/
# ├── docker-compose.yml
# ├── init-db.sql          ← HERE
# ├── api/
# └── ui/
```

### Services Won't Start

```bash
# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Check service logs
docker-compose logs postgres
docker-compose logs milvus
docker-compose logs api

# Recreate volumes (⚠️ deletes data)
docker-compose down -v
docker-compose up -d
```

### Database Connection Errors

The API has built-in retry logic (10 attempts, 3s intervals). If you see:

```
🔌 Connecting to PostgreSQL (Attempt 1/10)...
```

This is **normal** during startup. Wait ~30 seconds for PostgreSQL to initialize.

If connection fails after 10 attempts:

```bash
# Check PostgreSQL
docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT 1"

# Check if init-db.sql was loaded
docker exec -it rag_postgres psql -U rag_user -d rag_db -c "\dt"

# Should show: chat_sessions, chat_events, file_registry, document_chunks
```

### Milvus Connection Errors

```bash
# Check Milvus health
curl http://localhost:9091/healthz

# Check etcd (Milvus dependency)
docker-compose logs etcd

# Restart Milvus cluster
docker-compose restart etcd minio milvus
sleep 30
docker-compose restart api
```

### Volume Permission Issues

```bash
# Linux: Fix ownership
sudo chown -R $USER:$USER .

# Check volume mounts
docker volume ls | grep rag

# Remove and recreate (⚠️ deletes data)
docker-compose down -v
docker volume prune
docker-compose up -d
```

### GPU Not Detected

```bash
# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### "Module Not Found" Errors in API

```bash
# Rebuild API container
docker-compose build --no-cache api
docker-compose up -d api

# Check if requirements installed
docker exec -it rag_api pip list
```

## 📁 Project Structure

```
rag-v2-ultimate/
├── docker-compose.yml       # ⭐ Service orchestration
├── init-db.sql             # ⭐ PostgreSQL schema (auto-loaded)
├── .env                    # ⭐ Create from .env.example
├── .env.example            # Environment template
├── setup.sh                # 🚀 Automated setup script
├── README.md               # This file
├── PROJECT_STRUCTURE.md    # Detailed file descriptions
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py         # FastAPI gateway
│       ├── database.py     # PostgreSQL operations
│       ├── rag_core.py     # RAG engine
│       └── ingest.py       # Document processing
│
└── ui/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py              # Streamlit interface
```

**Important Notes:**

1. **`init-db.sql` location**: Must be in **project root** (same level as `docker-compose.yml`)
2. **Volumes**: Automatically created by Docker (see `docker-compose.yml`)
3. **`.env` file**: Must be created from `.env.example` before first run

**Verify files exist:**

```bash
# Check critical files
ls -la init-db.sql docker-compose.yml .env

# Should show all three files
```

## 🔐 Security Notes

- PostgreSQL exposed on 5433 (change in production)
- Use strong passwords in `.env`
- Restrict API access with firewall rules
- Enable SSL/TLS for production deployment

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please open issues for bugs/features.

## 🙏 Acknowledgments

- Milvus for vector search
- Ollama for local LLM inference
- Google Gemini for fast cloud inference
- PostgreSQL for FTS capabilities

---

**Built with ❤️ for production RAG systems**