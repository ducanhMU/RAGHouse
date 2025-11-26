# RAG Financial Assistant V1

A production-ready Retrieval-Augmented Generation (RAG) system with hierarchical memory management, hybrid LLM inference, and persistent conversation storage.

## 🌟 Features

### Core Capabilities
- **Hybrid RAG**: Combines vector search (Milvus) with hierarchical memory
- **Dual LLM Support**: Primary (Google Gemini) with fallback (Ollama)
- **Hierarchical Memory**: Implements the 3-5 Rule for conversation management
  - Every 3 turns → Short-term summary
  - Every 5 summaries → Long-term checkpoint
- **Persistent Storage**: PostgreSQL for conversations, file tracking, and memory
- **Streaming Responses**: Real-time SSE streaming for better UX
- **Idempotent Uploads**: MD5 hash-based deduplication

### Technical Stack
- **Backend**: FastAPI + LangChain
- **Frontend**: Streamlit
- **Database**: PostgreSQL 15
- **Vector DB**: Milvus 2.6
- **LLM**: Google Gemini 1.5 Flash + Ollama (Llama 3.2)
- **Orchestration**: Docker Compose

## 📁 Project Structure

```
rag-system/
├── api/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── rag_core.py          # RAG engine & memory management
│   │   ├── database.py          # SQLAlchemy models
│   │   └── ingest.py            # Document processing
│   ├── Dockerfile
│   └── requirements.txt
├── ui/
│   ├── app.py                   # Streamlit interface
│   ├── Dockerfile
│   └── requirements.txt
├── volumes/                     # Persistent data
│   ├── postgres/
│   ├── milvus/
│   ├── minio/
│   └── etcd/
├── docker-compose.yml
├── .env.example
├── Makefile
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU for Ollama
- (Optional) Google AI API key

### 1. Clone & Setup

```bash
git clone <repository-url>
cd rag-system

# Create environment file
cp .env.example .env

# Edit .env and add your Google API key (optional)
nano .env
```

### 2. Build & Start

```bash
# Option A: Using Make (recommended)
make dev-setup
make build
make up

# Option B: Using Docker Compose directly
docker-compose up -d --build
```

### 3. Wait for Services

```bash
# Check health
make health

# Or manually check
docker-compose ps
```

Services will be available at:
- **UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Milvus Admin**: http://localhost:3000
- **MinIO Console**: http://localhost:9001

## 📖 Usage

### Web Interface

1. Open http://localhost:8501
2. Upload documents via sidebar
3. Start chatting!

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Upload Document
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

#### Chat (Streaming)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is in the uploaded document?",
    "session_id": null
  }'
```

#### List Sessions
```bash
curl http://localhost:8000/sessions
```

#### Get Session History
```bash
curl http://localhost:8000/sessions/{session_id}/history
```

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db

# Google AI (Optional - for Gemini)
GOOGLE_API_KEY=your_api_key_here

# Ollama
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text

# Application
MAX_FILE_SIZE=52428800  # 50MB
```

### LLM Configuration

**Primary LLM (Gemini)**:
- Set `GOOGLE_API_KEY` in `.env`
- Free tier: 15 requests/minute
- Get key: https://makersuite.google.com/app/apikey

**Fallback LLM (Ollama)**:
- Runs locally, no API key needed
- Default model: `llama3.2:3b`
- Pull models: `make pull-models`

## 🏗️ Architecture

### System Overview

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  PostgreSQL  │
│     UI      │     │     API     │     │   (Memory)   │
└─────────────┘     └─────────────┘     └──────────────┘
                           │
                           ├────▶ ┌──────────────┐
                           │      │    Milvus    │
                           │      │  (Vectors)   │
                           │      └──────────────┘
                           │
                           └────▶ ┌──────────────┐
                                  │    Ollama    │
                                  │  / Gemini    │
                                  └──────────────┘
```

### Memory Management (3-5 Rule)

```
┌─────────────────────────────────────────────────────┐
│                    CONVERSATION                      │
├─────────────────────────────────────────────────────┤
│  [User] What is quantum computing?                  │
│  [AI] Quantum computing uses quantum mechanics...   │
│  [User] How does it differ from classical?          │
│  [AI] Classical computers use bits...               │
│  [User] What are its applications?                  │
│  [AI] Applications include cryptography...          │
├─────────────────────────────────────────────────────┤
│  3 Turns (6 messages) → SUMMARY_3 Created           │
│  "Discussion about quantum computing basics..."     │
└─────────────────────────────────────────────────────┘

After 5 summaries → CHECKPOINT_5 created
"Comprehensive conversation about quantum technology..."
```

### Database Schema

**FileRegistry**: Tracks uploaded documents
- `id`: UUID
- `file_hash`: MD5 (for deduplication)
- `filename`, `file_path`
- `status`: PENDING → PROCESSING → COMPLETED/FAILED

**ChatSession**: Conversation containers
- `id`: UUID
- `title`: First message preview
- `created_at`, `updated_at`

**ChatEvent**: Append-only log
- `session_id`: FK to ChatSession
- `sequence_num`: Ordering
- `role`: user/assistant/system
- `content`: Message text
- `event_type`: NORMAL/SUMMARY_3/CHECKPOINT_5
- `visibility`: VISIBLE/HIDDEN

## 🛠️ Development

### Make Commands

```bash
make help          # Show all commands
make build         # Build images
make up            # Start services
make down          # Stop services
make logs          # View all logs
make logs-api      # View API logs only
make health        # Check service health
make clean         # Remove all data (WARNING: destructive)
make shell-api     # Open API container shell
make shell-postgres # Open PostgreSQL shell
```

### Adding New Ollama Models

```bash
# Enter Ollama container
make shell-ollama

# Pull a model
ollama pull mistral

# List models
ollama list

# Update .env
OLLAMA_MODEL=mistral
```

### Database Migrations

```bash
# Access PostgreSQL
make shell-postgres

# Or use external tool
psql -h localhost -U rag_user -d rag_db
```

## 🔍 Monitoring

### Health Checks

```bash
# Quick health check
make health

# Detailed status
curl http://localhost:8000/health | jq .
```

### Logs

```bash
# All services
make logs

# Specific service
docker-compose logs -f api

# Follow errors only
docker-compose logs -f api | grep ERROR
```

### Resource Usage

```bash
make monitor
# or
docker stats
```

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check logs
make logs-api

# Restart specific service
docker-compose restart api

# Complete restart
make restart
```

### Database Connection Issues

```bash
# Check PostgreSQL
make shell-postgres

# Test connection
docker exec rag_postgres pg_isready -U rag_user
```

### Ollama Model Issues

```bash
# Check models
make list-models

# Re-pull models
make pull-models

# Check Ollama logs
make logs-ollama
```

### Milvus Issues

```bash
# Check Milvus health
curl http://localhost:9091/healthz

# Access Attu (Milvus UI)
# Open http://localhost:3000
```

### Out of Memory

```bash
# Check usage
make monitor

# Increase Docker memory in Docker Desktop settings
# Minimum: 8GB recommended
```

## 📊 Performance

### Benchmarks (Local Testing)

- **Document Upload**: ~5-10s for 100-page PDF
- **Embedding Generation**: ~1s per 1000 tokens
- **Query Response**: ~2-5s (Gemini), ~5-10s (Ollama)
- **Concurrent Users**: Tested up to 10 simultaneous

### Optimization Tips

1. **Enable GPU for Ollama**:
   ```yaml
   # In docker-compose.yml
   ollama:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: all
               capabilities: [gpu]
   ```

2. **Increase Workers**:
   ```bash
   # In docker-compose.yml, api service
   command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

3. **Cache Embeddings**:
   - Add Redis for caching
   - Implement embedding cache in `ingest.py`

## 🔒 Security

### Current Status
- ⚠️ No authentication (development mode)
- ⚠️ CORS allows all origins
- ✅ Input validation via Pydantic
- ✅ SQL injection protection (ORM)
- ✅ File type/size validation

### Production Checklist
- [ ] Add authentication (JWT tokens)
- [ ] Restrict CORS origins
- [ ] Enable HTTPS
- [ ] Add rate limiting
- [ ] Implement audit logging
- [ ] Use secrets management (Vault)
- [ ] Enable database encryption
- [ ] Add API key management

## 📦 Backup & Restore

### Backup Database

```bash
make backup-db
# Creates backups/backup_YYYYMMDD_HHMMSS.sql
```

### Restore Database

```bash
make restore-db FILE=backups/backup_20240101_120000.sql
```

### Backup Volumes

```bash
# Stop services
make down

# Backup volumes directory
tar -czf rag-volumes-backup.tar.gz volumes/

# Restart
make up
```

## 🚀 Production Deployment

### 1. Prepare Environment

```bash
# Update .env for production
POSTGRES_PASSWORD=<strong-password>
GOOGLE_API_KEY=<your-key>

# Set resource limits in docker-compose.yml
```

### 2. Build Production Images

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
```

### 3. Deploy

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 4. Configure Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `make test-api`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file

## 🙋 Support

- Issues: GitHub Issues
- Documentation: /docs folder
- Email: support@example.com

## 🗺️ Roadmap

- [ ] Multi-user support with authentication
- [ ] Advanced citation tracking
- [ ] Support for more file types (DOCX, XLSX, PPTX)
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Mobile app
- [ ] Voice interface
- [ ] Multi-language support

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built with ❤️ by the RAG Team**