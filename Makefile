# RAG V2 Ultimate - Makefile
# Convenient commands for development and deployment

.PHONY: help setup start stop restart logs clean build health backup restore

# Default target
help:
	@echo "🚀 RAG V2 Ultimate - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup & Deployment:"
	@echo "  make setup      - Automated setup (first time)"
	@echo "  make start      - Start all services"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make build      - Rebuild containers"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs       - View all logs (live)"
	@echo "  make logs-api   - View API logs"
	@echo "  make logs-ui    - View UI logs"
	@echo "  make ps         - Show service status"
	@echo "  make health     - Check system health"
	@echo ""
	@echo "Database:"
	@echo "  make db         - Connect to PostgreSQL"
	@echo "  make backup     - Backup all volumes"
	@echo "  make restore    - Restore from backup"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Stop and remove containers"
	@echo "  make clean-all  - Remove everything (⚠️ data loss)"
	@echo "  make rebuild    - Clean rebuild all containers"
	@echo ""

# Setup (first time)
setup:
	@echo "🚀 Running automated setup..."
	@chmod +x setup.sh
	@./setup.sh

# Start services
start:
	@echo "🚀 Starting RAG V2 Ultimate..."
	@docker-compose up -d
	@echo "✅ Services started!"
	@echo "   UI: http://localhost:8501"
	@echo "   API: http://localhost:8000/docs"

# Stop services
stop:
	@echo "🛑 Stopping services..."
	@docker-compose down
	@echo "✅ Services stopped"

# Restart services
restart:
	@echo "🔄 Restarting services..."
	@docker-compose restart
	@echo "✅ Services restarted"

# View logs
logs:
	@docker-compose logs -f

logs-api:
	@docker-compose logs -f api

logs-ui:
	@docker-compose logs -f ui

logs-db:
	@docker-compose logs -f postgres

logs-milvus:
	@docker-compose logs -f milvus

# Service status
ps:
	@docker-compose ps

# Health check
health:
	@echo "🏥 Checking system health..."
	@curl -s http://localhost:8000/health | jq || echo "❌ API not responding"
	@echo ""
	@docker-compose ps

# Build containers
build:
	@echo "🔨 Building containers..."
	@docker-compose build
	@echo "✅ Build complete"

rebuild:
	@echo "🔨 Rebuilding containers (no cache)..."
	@docker-compose build --no-cache
	@docker-compose up -d
	@echo "✅ Rebuild complete"

# Database access
db:
	@docker exec -it rag_postgres psql -U rag_user -d rag_db

db-stats:
	@docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT * FROM v_file_stats;"

db-sessions:
	@docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT * FROM v_session_stats ORDER BY updated_at DESC LIMIT 10;"

# Backup
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker run --rm \
		-v rag-v2-ultimate_postgres_data:/data \
		-v $(PWD)/backups:/backup \
		ubuntu tar czf /backup/postgres_$(shell date +%Y%m%d_%H%M%S).tar.gz /data
	@echo "✅ Backup created in ./backups/"

# Restore (use latest backup)
restore:
	@echo "⚠️  This will restore from the latest backup"
	@read -p "Continue? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		LATEST=$$(ls -t backups/postgres_*.tar.gz | head -1); \
		echo "📦 Restoring from $$LATEST"; \
		docker-compose down; \
		docker run --rm \
			-v rag-v2-ultimate_postgres_data:/data \
			-v $(PWD)/backups:/backup \
			ubuntu tar xzf /backup/$$(basename $$LATEST) -C /; \
		docker-compose up -d; \
		echo "✅ Restore complete"; \
	fi

# Clean (preserve volumes)
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down
	@echo "✅ Cleanup complete (volumes preserved)"

# Clean all (including volumes - ⚠️ DATA LOSS)
clean-all:
	@echo "⚠️  WARNING: This will delete ALL data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v --rmi all; \
		docker volume prune -f; \
		echo "✅ Complete cleanup done"; \
	fi

# Development helpers
dev-api:
	@docker-compose restart api
	@docker-compose logs -f api

dev-ui:
	@docker-compose restart ui
	@docker-compose logs -f ui

# GPU check
gpu:
	@echo "🎮 Checking GPU access..."
	@docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Quick tests
test-health:
	@curl -s http://localhost:8000/health | jq

test-upload:
	@echo "Testing file upload endpoint..."
	@curl -X POST "http://localhost:8000/files/upload" \
		-F "file=@test.pdf" 2>/dev/null | jq || echo "❌ Test failed (ensure test.pdf exists)"

# Install dependencies (for local development)
install:
	@echo "📦 Installing local dependencies..."
	@pip install -r api/requirements.txt
	@pip install -r ui/requirements.txt
	@echo "✅ Dependencies installed"

# Format code
format:
	@echo "🎨 Formatting code..."
	@black api/app/*.py
	@isort api/app/*.py
	@echo "✅ Code formatted"

# Run tests (if test suite exists)
test:
	@echo "🧪 Running tests..."
	@docker exec -it rag_api pytest tests/ -v

# Show environment
env:
	@echo "Environment variables:"
	@cat .env 2>/dev/null || echo "❌ .env file not found. Run 'make setup' first."

# Update containers
update:
	@echo "📥 Pulling latest images..."
	@docker-compose pull
	@echo "🔨 Rebuilding custom images..."
	@docker-compose build
	@echo "🔄 Restarting services..."
	@docker-compose up -d
	@echo "✅ Update complete"