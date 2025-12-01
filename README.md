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

### 1. Clone & Configure

```bash
git clone <your-repo>
cd rag-v2-ultimate

# Copy environment file
cp .env.example .env

# Edit .env and add your Gemini API key
nano .env
# Set: GOOGLE_API_KEY=your_actual_key_here
```

### 2. Launch System

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop system
docker-compose down
```

### 3. Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Main application |
| **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| **Milvus Attu** | http://localhost:3000 | Vector DB admin |
| **MinIO Console** | http://localhost:9001 | Object storage (admin/admin) |

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

### Services Won't Start

```bash
# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Recreate volumes
docker-compose down -v
docker-compose up -d
```

### Milvus Connection Errors

```bash
# Check Milvus health
curl http://localhost:9091/healthz

# Restart Milvus cluster
docker-compose restart etcd minio milvus
```

### PostgreSQL Connection Issues

```bash
# Check PostgreSQL
docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT 1"

# View logs
docker-compose logs postgres
```

## 📁 Project Structure

```
rag-v2-ultimate/
├── docker-compose.yml       # Service orchestration
├── init-db.sql             # Database schema
├── .env.example            # Configuration template
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py         # FastAPI gateway
│       ├── database.py     # PostgreSQL operations
│       ├── rag_core.py     # RAG engine
│       └── ingest.py       # Document processing
└── ui/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py              # Streamlit interface
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