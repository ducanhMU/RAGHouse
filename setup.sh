#!/bin/bash

# =========================================
# RAG V2 ULTIMATE - AUTOMATED SETUP SCRIPT
# =========================================

set -e

echo "🚀 RAG V2 Ultimate - Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${YELLOW}⚠️  Warning: This script is optimized for Linux. Some features may not work on other systems.${NC}"
fi

# =========================================
# 1. CHECK PREREQUISITES
# =========================================

echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose found${NC}"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # Check nvidia-container-toolkit
    if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ nvidia-container-toolkit configured${NC}"
    else
        echo -e "${YELLOW}⚠️  nvidia-container-toolkit not found or not configured${NC}"
        echo "Install it with:"
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
        read -p "Continue without GPU support? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. System will run in CPU-only mode (slower).${NC}"
fi

echo ""

# =========================================
# 2. CREATE DIRECTORY STRUCTURE
# =========================================

echo "📁 Creating directory structure..."

# Create main directories
mkdir -p api/app
mkdir -p ui
mkdir -p volumes/{postgres,etcd,minio,milvus}

echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# =========================================
# 3. CREATE .env FILE
# =========================================

if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    
    read -p "Enter your Google Gemini API Key (or press Enter to skip): " GEMINI_KEY
    
    cat > .env << EOF
# =========================================
# RAG V2 ULTIMATE - ENVIRONMENT VARIABLES
# =========================================

# PostgreSQL Configuration
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password_$(openssl rand -hex 8)
POSTGRES_DB=rag_db

# Milvus Configuration
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=rag_collection_v2_hnsw

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BATCH_SIZE=32

# Google Gemini API
GOOGLE_API_KEY=${GEMINI_KEY}

# RAG Configuration
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2
EMBEDDING_BATCH_SIZE=64

# Application Configuration
MAX_WORKERS=4
DATA_PATH=/app/data
LOG_LEVEL=INFO
EOF
    
    echo -e "${GREEN}✅ .env file created${NC}"
    
    if [ -z "$GEMINI_KEY" ]; then
        echo -e "${YELLOW}⚠️  No Gemini API key provided. System will use Ollama as primary LLM (slower).${NC}"
        echo "   Get a free key at: https://aistudio.google.com/app/apikey"
    fi
else
    echo -e "${YELLOW}⚠️  .env file already exists. Skipping creation.${NC}"
fi

echo ""

# =========================================
# 4. VERIFY FILE STRUCTURE
# =========================================

echo "🔍 Verifying file structure..."

REQUIRED_FILES=(
    "docker-compose.yml"
    "init-db.sql"
    "api/Dockerfile"
    "api/requirements.txt"
    "api/app/__init__.py"
    "api/app/main.py"
    "api/app/database.py"
    "api/app/rag_core.py"
    "api/app/ingest.py"
    "ui/Dockerfile"
    "ui/requirements.txt"
    "ui/app.py"
)

MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All required files present${NC}"
else
    echo -e "${RED}❌ Missing files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Please ensure all files are in place before continuing."
    exit 1
fi

echo ""

# =========================================
# 5. BUILD AND START SERVICES
# =========================================

echo "🏗️  Building and starting services..."
echo ""

# Pull images first
echo "📥 Pulling base images..."
docker-compose pull

# Build custom images
echo "🔨 Building custom images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo -e "${GREEN}✅ Services started!${NC}"
echo ""

# =========================================
# 6. WAIT FOR SERVICES TO BE HEALTHY
# =========================================

echo "⏳ Waiting for services to be ready..."
echo ""

wait_for_service() {
    local service=$1
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep $service | grep -q "healthy\|Up"; then
            echo -e "${GREEN}✅ $service is ready${NC}"
            return 0
        fi
        echo "   Waiting for $service... ($attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service failed to start${NC}"
    return 1
}

wait_for_service "postgres"
wait_for_service "milvus"
wait_for_service "ollama"
wait_for_service "api"
wait_for_service "ui"

echo ""

# =========================================
# 7. VERIFY SYSTEM HEALTH
# =========================================

echo "🏥 Checking system health..."
sleep 5

HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo "{}")

if echo $HEALTH_RESPONSE | grep -q '"postgres":"ok"'; then
    echo -e "${GREEN}✅ PostgreSQL: Online${NC}"
else
    echo -e "${RED}❌ PostgreSQL: Offline${NC}"
fi

if echo $HEALTH_RESPONSE | grep -q '"milvus":"ok"'; then
    echo -e "${GREEN}✅ Milvus: Online${NC}"
else
    echo -e "${RED}❌ Milvus: Offline${NC}"
fi

if echo $HEALTH_RESPONSE | grep -q '"models":"ok"'; then
    echo -e "${GREEN}✅ AI Models: Ready${NC}"
else
    echo -e "${YELLOW}⚠️  AI Models: Loading...${NC}"
fi

echo ""

# =========================================
# 8. DISPLAY ACCESS INFORMATION
# =========================================

echo "=========================================="
echo "🎉 RAG V2 Ultimate is ready!"
echo "=========================================="
echo ""
echo "📊 Access Points:"
echo ""
echo "  🎨 Streamlit UI:     http://localhost:8501"
echo "  📡 FastAPI Docs:     http://localhost:8000/docs"
echo "  🔍 Milvus Attu:      http://localhost:3000"
echo "  💾 MinIO Console:    http://localhost:9001"
echo "     (user: minioadmin / pass: minioadmin)"
echo ""
echo "📝 Useful Commands:"
echo ""
echo "  View logs:           docker-compose logs -f"
echo "  View API logs:       docker-compose logs -f api"
echo "  Stop system:         docker-compose down"
echo "  Restart system:      docker-compose restart"
echo "  Clean everything:    docker-compose down -v"
echo ""
echo "🔧 Troubleshooting:"
echo ""
echo "  Check service status: docker-compose ps"
echo "  Check health:         curl http://localhost:8000/health"
echo "  Rebuild containers:   docker-compose up -d --build"
echo ""
echo "=========================================="
echo ""

# =========================================
# 9. OPTIONAL: OPEN BROWSER
# =========================================

read -p "Open Streamlit UI in browser? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8501
    elif command -v open &> /dev/null; then
        open http://localhost:8501
    else
        echo "Please open http://localhost:8501 in your browser"
    fi
fi

echo ""
echo -e "${GREEN}Setup complete! Happy chatting! 🚀${NC}"