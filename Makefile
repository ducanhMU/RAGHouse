# Makefile for RAG System

.PHONY: help build up down restart logs clean test health

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@make health

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs from all services
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

logs-ui: ## View UI logs
	docker-compose logs -f ui

logs-ollama: ## View Ollama logs
	docker-compose logs -f ollama

clean: ## Stop and remove all containers, networks, volumes
	docker-compose down -v
	rm -rf volumes/
	@echo "All data cleaned. You will need to re-upload documents."

clean-soft: ## Stop and remove containers but keep volumes
	docker-compose down

health: ## Check health of all services
	@echo "=== Checking Service Health ==="
	@echo "\n1. PostgreSQL:"
	@docker exec rag_postgres pg_isready -U rag_user || echo "  ❌ Not ready"
	@echo "\n2. Milvus:"
	@curl -s http://localhost:9091/healthz > /dev/null && echo "  ✅ Healthy" || echo "  ❌ Not ready"
	@echo "\n3. Ollama:"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "  ✅ Healthy" || echo "  ❌ Not ready"
	@echo "\n4. API:"
	@curl -s http://localhost:8000/health | jq . || echo "  ❌ Not ready"
	@echo "\n5. UI:"
	@curl -s http://localhost:8501/_stcore/health > /dev/null && echo "  ✅ Healthy" || echo "  ❌ Not ready"

pull-models: ## Pull required Ollama models
	@echo "Pulling Ollama models..."
	docker exec rag_ollama ollama pull nomic-embed-text
	docker exec rag_ollama ollama pull llama3.2:3b
	@echo "Models pulled successfully!"

list-models: ## List installed Ollama models
	docker exec rag_ollama ollama list

shell-api: ## Open shell in API container
	docker exec -it rag_api /bin/bash

shell-postgres: ## Open psql in PostgreSQL
	docker exec -it rag_postgres psql -U rag_user -d rag_db

shell-ollama: ## Open shell in Ollama container
	docker exec -it rag_ollama /bin/bash

backup-db: ## Backup PostgreSQL database
	@mkdir -p backups
	docker exec rag_postgres pg_dump -U rag_user rag_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backed up to backups/"

restore-db: ## Restore PostgreSQL database (use: make restore-db FILE=backups/backup.sql)
	@if [ -z "$(FILE)" ]; then echo "Usage: make restore-db FILE=backups/backup.sql"; exit 1; fi
	docker exec -i rag_postgres psql -U rag_user -d rag_db < $(FILE)
	@echo "Database restored from $(FILE)"

dev-setup: ## Initial development setup
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file"; fi
	@mkdir -p volumes/postgres volumes/milvus volumes/etcd volumes/minio api/data api/logs
	@echo "Development environment ready!"
	@echo "Run 'make up' to start all services"

test-api: ## Test API endpoints
	@echo "Testing API health..."
	curl -s http://localhost:8000/health | jq .
	@echo "\nTesting sessions endpoint..."
	curl -s http://localhost:8000/sessions | jq .
	@echo "\nTesting files endpoint..."
	curl -s http://localhost:8000/files | jq .

monitor: ## Show resource usage
	docker stats rag_api rag_ui rag_postgres rag_milvus rag_ollama

prune: ## Remove unused Docker resources
	docker system prune -f
	docker volume prune -f

install-hooks: ## Install git hooks
	@echo "Installing git hooks..."
	@echo "#!/bin/bash\nmake test-api" > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "Git hooks installed!"

# Production commands
prod-build: ## Build for production
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up: ## Start production services
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-down: ## Stop production services
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down