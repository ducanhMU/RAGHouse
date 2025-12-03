# ============================================
# Makefile for RAG Financial Assistant
# ============================================
# Complete utility commands for development, deployment, and maintenance
# Version: 1.0.0
# ============================================

# ============================================
# Variables & Configuration
# ============================================

# Project name
PROJECT_NAME := rag
COMPOSE_PROJECT_NAME := rag

# Docker Compose files
COMPOSE_FILE := docker-compose.yml
COMPOSE_DEV_FILE := docker-compose.dev.yml

# Container names
CONTAINER_API := rag_api
CONTAINER_UI := rag_ui
CONTAINER_POSTGRES := rag_postgres
CONTAINER_MILVUS := rag_milvus
CONTAINER_OLLAMA := rag_ollama
CONTAINER_ETCD := rag_etcd
CONTAINER_MINIO := rag_minio
CONTAINER_ATTU := rag_attu

# Backup directory
BACKUP_DIR := backups

# Color codes for pretty output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
BOLD := \033[1m
NC := \033[0m # No Color

# Timestamp for backups
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Python command
PYTHON := python3

# Docker compose command
DC := docker-compose -f $(COMPOSE_FILE)
DC_DEV := docker-compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE)

# ============================================
# Default Target
# ============================================

.DEFAULT_GOAL := help

# ============================================
# Phony Targets
# ============================================

.PHONY: help build up down restart logs clean \
        pull-models test-db health shell-api shell-ui \
        backup-db restore-db reset-db quickstart \
        install-nvidia check-gpu check-deps \
        dev test lint format \
        clean-all prune watch-logs monitor-gpu \
        logs-api logs-ui logs-milvus logs-postgres logs-ollama \
        status stats services features \
        list-models remove-models db-shell \
        clean-cache validate-compose export-env \
        update-images disk-usage network-inspect volume-inspect \
        benchmark stress-test

# ============================================
# Help & Documentation
# ============================================

help:
	@echo ""
	@echo "$(BLUE)$(BOLD)╔═══════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)$(BOLD)║                                                                           ║$(NC)"
	@echo "$(BLUE)$(BOLD)║               RAG FINANCIAL ASSISTANT - COMMAND REFERENCE                ║$(NC)"
	@echo "$(BLUE)$(BOLD)║                         Version 1.0.0                                     ║$(NC)"
	@echo "$(BLUE)$(BOLD)║                                                                           ║$(NC)"
	@echo "$(BLUE)$(BOLD)╚═══════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🚀 QUICK START COMMANDS$(NC)"
	@echo "$(CYAN)  make quickstart          $(WHITE)Complete setup: build + start + pull models$(NC)"
	@echo "$(CYAN)  make up                  $(WHITE)Start all services$(NC)"
	@echo "$(CYAN)  make down                $(WHITE)Stop all services$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🏗️  BUILD & DEPLOYMENT$(NC)"
	@echo "$(CYAN)  make build               $(WHITE)Build all Docker images$(NC)"
	@echo "$(CYAN)  make rebuild             $(WHITE)Rebuild images (no cache) and restart$(NC)"
	@echo "$(CYAN)  make restart             $(WHITE)Restart all running services$(NC)"
	@echo "$(CYAN)  make dev                 $(WHITE)Start in development mode (hot reload)$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)📋 LOGS & MONITORING$(NC)"
	@echo "$(CYAN)  make logs                $(WHITE)View logs from all services (follow mode)$(NC)"
	@echo "$(CYAN)  make logs-api            $(WHITE)View API logs only$(NC)"
	@echo "$(CYAN)  make logs-ui             $(WHITE)View UI logs only$(NC)"
	@echo "$(CYAN)  make logs-milvus         $(WHITE)View Milvus logs only$(NC)"
	@echo "$(CYAN)  make logs-postgres       $(WHITE)View PostgreSQL logs only$(NC)"
	@echo "$(CYAN)  make logs-ollama         $(WHITE)View Ollama logs only$(NC)"
	@echo "$(CYAN)  make watch-logs          $(WHITE)Watch logs with auto-refresh$(NC)"
	@echo "$(CYAN)  make monitor-gpu         $(WHITE)Monitor GPU usage in real-time$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔍 HEALTH & STATUS$(NC)"
	@echo "$(CYAN)  make health              $(WHITE)Check overall system health$(NC)"
	@echo "$(CYAN)  make status              $(WHITE)Show all container status$(NC)"
	@echo "$(CYAN)  make stats               $(WHITE)Display system statistics$(NC)"
	@echo "$(CYAN)  make services            $(WHITE)List all service URLs$(NC)"
	@echo "$(CYAN)  make features            $(WHITE)Show enabled features$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🤖 MODEL MANAGEMENT$(NC)"
	@echo "$(CYAN)  make pull-models         $(WHITE)Download Ollama models (Llama 3.2 3B)$(NC)"
	@echo "$(CYAN)  make list-models         $(WHITE)List all downloaded Ollama models$(NC)"
	@echo "$(CYAN)  make remove-models       $(WHITE)Remove all Ollama models$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)💾 DATABASE OPERATIONS$(NC)"
	@echo "$(CYAN)  make test-db             $(WHITE)Test PostgreSQL connection$(NC)"
	@echo "$(CYAN)  make db-shell            $(WHITE)Open PostgreSQL interactive shell$(NC)"
	@echo "$(CYAN)  make backup-db           $(WHITE)Backup database to ./backups/$(NC)"
	@echo "$(CYAN)  make restore-db          $(WHITE)Restore database (FILE=backup.sql)$(NC)"
	@echo "$(CYAN)  make reset-db            $(WHITE)Reset database (⚠️  deletes all data!)$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🐚 SHELL ACCESS$(NC)"
	@echo "$(CYAN)  make shell-api           $(WHITE)Open bash shell in API container$(NC)"
	@echo "$(CYAN)  make shell-ui            $(WHITE)Open bash shell in UI container$(NC)"
	@echo "$(CYAN)  make shell-milvus        $(WHITE)Open bash shell in Milvus container$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🧹 CLEANUP$(NC)"
	@echo "$(CYAN)  make clean               $(WHITE)Stop and remove all containers$(NC)"
	@echo "$(CYAN)  make clean-all           $(WHITE)Remove containers + volumes (⚠️  DATA LOSS!)$(NC)"
	@echo "$(CYAN)  make clean-cache         $(WHITE)Clean Docker build cache$(NC)"
	@echo "$(CYAN)  make prune               $(WHITE)Prune all unused Docker resources$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔧 DEVELOPMENT$(NC)"
	@echo "$(CYAN)  make test                $(WHITE)Run all tests$(NC)"
	@echo "$(CYAN)  make lint                $(WHITE)Run code linters$(NC)"
	@echo "$(CYAN)  make format              $(WHITE)Format code with Black$(NC)"
	@echo "$(CYAN)  make validate-compose    $(WHITE)Validate docker-compose.yml$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)📦 SYSTEM SETUP$(NC)"
	@echo "$(CYAN)  make check-deps          $(WHITE)Check required system dependencies$(NC)"
	@echo "$(CYAN)  make check-gpu           $(WHITE)Check GPU availability$(NC)"
	@echo "$(CYAN)  make install-nvidia      $(WHITE)Install NVIDIA Container Toolkit$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔄 MAINTENANCE$(NC)"
	@echo "$(CYAN)  make update-images       $(WHITE)Pull latest Docker images$(NC)"
	@echo "$(CYAN)  make disk-usage          $(WHITE)Show Docker disk usage$(NC)"
	@echo "$(CYAN)  make network-inspect     $(WHITE)Inspect Docker network$(NC)"
	@echo "$(CYAN)  make volume-inspect      $(WHITE)List Docker volumes$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🔬 BENCHMARKING$(NC)"
	@echo "$(CYAN)  make benchmark           $(WHITE)Run performance benchmark$(NC)"
	@echo "$(CYAN)  make stress-test         $(WHITE)Run API stress test$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)📝 EXAMPLES$(NC)"
	@echo "$(WHITE)  # Complete setup from scratch$(NC)"
	@echo "$(CYAN)  make quickstart$(NC)"
	@echo ""
	@echo "$(WHITE)  # View specific service logs$(NC)"
	@echo "$(CYAN)  make logs-api$(NC)"
	@echo ""
	@echo "$(WHITE)  # Backup and restore database$(NC)"
	@echo "$(CYAN)  make backup-db$(NC)"
	@echo "$(CYAN)  make restore-db FILE=backups/rag_db_20240315_120000.sql$(NC)"
	@echo ""
	@echo "$(WHITE)  # Debug in container$(NC)"
	@echo "$(CYAN)  make shell-api$(NC)"
	@echo ""

# ============================================
# Build & Deploy
# ============================================

build:
	@echo "$(BLUE)$(BOLD)🏗️  Building Docker images...$(NC)"
	@$(DC) build
	@echo "$(GREEN)✓ Build complete!$(NC)"

up:
	@echo "$(BLUE)$(BOLD)🚀 Starting RAG system...$(NC)"
	@$(DC) up -d
	@echo ""
	@echo "$(GREEN)$(BOLD)✓ Services started successfully!$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)📊 Service URLs:$(NC)"
	@echo "$(CYAN)  • UI:          $(WHITE)http://localhost:8501$(NC)"
	@echo "$(CYAN)  • API:         $(WHITE)http://localhost:8000$(NC)"
	@echo "$(CYAN)  • API Docs:    $(WHITE)http://localhost:8000/docs$(NC)"
	@echo "$(CYAN)  • Attu:        $(WHITE)http://localhost:3000$(NC)"
	@echo "$(CYAN)  • MinIO:       $(WHITE)http://localhost:9001$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)🛠️  Useful Commands:$(NC)"
	@echo "$(CYAN)  make logs      $(WHITE)View all logs$(NC)"
	@echo "$(CYAN)  make health    $(WHITE)Check system health$(NC)"
	@echo "$(CYAN)  make status    $(WHITE)View container status$(NC)"
	@echo ""

down:
	@echo "$(BLUE)$(BOLD)🛑 Stopping RAG system...$(NC)"
	@$(DC) down
	@echo "$(GREEN)✓ All services stopped!$(NC)"

restart:
	@echo "$(BLUE)$(BOLD)🔄 Restarting RAG system...$(NC)"
	@$(DC) restart
	@echo "$(GREEN)✓ All services restarted!$(NC)"

rebuild:
	@echo "$(BLUE)$(BOLD)🔨 Rebuilding and restarting (no cache)...$(NC)"
	@$(DC) down
	@$(DC) build --no-cache
	@$(DC) up -d
	@echo "$(GREEN)✓ Rebuild complete!$(NC)"

# ============================================
# Logs & Monitoring
# ============================================

logs:
	@echo "$(BLUE)$(BOLD)📋 Viewing logs (Ctrl+C to exit)...$(NC)"
	@$(DC) logs -f --tail=100

logs-api:
	@echo "$(BLUE)$(BOLD)📋 Viewing API logs...$(NC)"
	@$(DC) logs -f --tail=100 api

logs-ui:
	@echo "$(BLUE)$(BOLD)📋 Viewing UI logs...$(NC)"
	@$(DC) logs -f --tail=100 ui

logs-milvus:
	@echo "$(BLUE)$(BOLD)📋 Viewing Milvus logs...$(NC)"
	@$(DC) logs -f --tail=100 milvus

logs-postgres:
	@echo "$(BLUE)$(BOLD)📋 Viewing PostgreSQL logs...$(NC)"
	@$(DC) logs -f --tail=100 postgres

logs-ollama:
	@echo "$(BLUE)$(BOLD)📋 Viewing Ollama logs...$(NC)"
	@$(DC) logs -f --tail=100 ollama

watch-logs:
	@echo "$(BLUE)$(BOLD)👀 Watching logs with auto-refresh...$(NC)"
	@watch -n 2 'docker-compose logs --tail=50'

monitor-gpu:
	@echo "$(BLUE)$(BOLD)🎮 Monitoring GPU usage (Ctrl+C to exit)...$(NC)"
	@watch -n 1 nvidia-smi

# ============================================
# Health & Status
# ============================================

health:
	@echo "$(BLUE)$(BOLD)🏥 Checking system health...$(NC)"
	@echo ""
	@curl -s http://localhost:8000/health 2>/dev/null | $(PYTHON) -m json.tool || echo "$(RED)✗ API not responding$(NC)"
	@echo ""

status:
	@echo "$(BLUE)$(BOLD)📊 Container Status:$(NC)"
	@echo ""
	@$(DC) ps
	@echo ""

stats:
	@echo "$(BLUE)$(BOLD)📈 Fetching system statistics...$(NC)"
	@echo ""
	@curl -s http://localhost:8000/stats/system 2>/dev/null | $(PYTHON) -m json.tool || echo "$(RED)✗ API not responding$(NC)"
	@echo ""

services:
	@echo "$(BLUE)$(BOLD)🔗 Service URLs:$(NC)"
	@echo ""
	@curl -s http://localhost:8000/system/services 2>/dev/null | $(PYTHON) -m json.tool || echo "$(RED)✗ API not responding$(NC)"
	@echo ""

features:
	@echo "$(BLUE)$(BOLD)⚡ Enabled Features:$(NC)"
	@echo ""
	@curl -s http://localhost:8000/features 2>/dev/null | $(PYTHON) -m json.tool || echo "$(RED)✗ API not responding$(NC)"
	@echo ""

# ============================================
# Model Management
# ============================================

pull-models:
	@echo "$(BLUE)$(BOLD)📥 Pulling Ollama models...$(NC)"
	@echo "$(YELLOW)Downloading Llama 3.2 3B (this may take a few minutes)...$(NC)"
	@docker exec -it $(CONTAINER_OLLAMA) ollama pull llama3.2:3b
	@echo ""
	@echo "$(GREEN)✓ Models downloaded successfully!$(NC)"

list-models:
	@echo "$(BLUE)$(BOLD)📋 Installed Ollama models:$(NC)"
	@echo ""
	@docker exec -it $(CONTAINER_OLLAMA) ollama list

remove-models:
	@echo "$(RED)$(BOLD)⚠️  WARNING: This will remove all Ollama models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker exec -it $(CONTAINER_OLLAMA) sh -c "ollama list | tail -n +2 | awk '{print \$$1}' | xargs -I {} ollama rm {}"; \
		echo "$(GREEN)✓ All models removed!$(NC)"; \
	else \
		echo "$(YELLOW)Aborted.$(NC)"; \
	fi

# ============================================
# Database Operations
# ============================================

test-db:
	@echo "$(BLUE)$(BOLD)🔌 Testing database connection...$(NC)"
	@echo ""
	@docker exec -it $(CONTAINER_POSTGRES) psql -U rag_user -d rag_db -c "SELECT version();" && \
		echo "" && echo "$(GREEN)✓ Database connection successful!$(NC)" || \
		echo "" && echo "$(RED)✗ Database connection failed!$(NC)"

db-shell:
	@echo "$(BLUE)$(BOLD)🐚 Opening PostgreSQL shell...$(NC)"
	@docker exec -it $(CONTAINER_POSTGRES) psql -U rag_user -d rag_db

backup-db:
	@echo "$(BLUE)$(BOLD)💾 Backing up database...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@docker exec $(CONTAINER_POSTGRES) pg_dump -U rag_user rag_db > $(BACKUP_DIR)/rag_db_$(TIMESTAMP).sql
	@echo ""
	@echo "$(GREEN)✓ Backup saved: $(BACKUP_DIR)/rag_db_$(TIMESTAMP).sql$(NC)"
	@ls -lh $(BACKUP_DIR)/rag_db_$(TIMESTAMP).sql

restore-db:
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)✗ Error: Please specify FILE=path/to/backup.sql$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)$(BOLD)📂 Restoring database from $(FILE)...$(NC)"
	@cat $(FILE) | docker exec -i $(CONTAINER_POSTGRES) psql -U rag_user -d rag_db
	@echo ""
	@echo "$(GREEN)✓ Database restored successfully!$(NC)"

reset-db:
	@echo "$(RED)$(BOLD)⚠️  WARNING: This will DELETE ALL DATA in the database!$(NC)"
	@echo "$(YELLOW)This action cannot be undone!$(NC)"
	@read -p "Type 'DELETE ALL DATA' to confirm: " -r; \
	echo; \
	if [ "$$REPLY" = "DELETE ALL DATA" ]; then \
		echo "$(BLUE)Resetting database...$(NC)"; \
		docker exec -it $(CONTAINER_POSTGRES) psql -U rag_user -d rag_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		echo "$(BLUE)Restarting API to rebuild schema...$(NC)"; \
		$(DC) restart api; \
		echo "$(GREEN)✓ Database reset complete!$(NC)"; \
	else \
		echo "$(YELLOW)Aborted. Database not modified.$(NC)"; \
	fi

# ============================================
# Shell Access
# ============================================

shell-api:
	@echo "$(BLUE)$(BOLD)🐚 Opening shell in API container...$(NC)"
	@docker exec -it $(CONTAINER_API) /bin/bash

shell-ui:
	@echo "$(BLUE)$(BOLD)🐚 Opening shell in UI container...$(NC)"
	@docker exec -it $(CONTAINER_UI) /bin/bash

shell-milvus:
	@echo "$(BLUE)$(BOLD)🐚 Opening shell in Milvus container...$(NC)"
	@docker exec -it $(CONTAINER_MILVUS) /bin/bash

# ============================================
# Cleanup
# ============================================

clean:
	@echo "$(BLUE)$(BOLD)🧹 Stopping and removing containers...$(NC)"
	@$(DC) down
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

clean-all:
	@echo "$(RED)$(BOLD)⚠️  WARNING: This will PERMANENTLY DELETE all containers and volumes!$(NC)"
	@echo "$(RED)You will lose all data including:$(NC)"
	@echo "$(YELLOW)  • All uploaded documents$(NC)"
	@echo "$(YELLOW)  • All chat history$(NC)"
	@echo "$(YELLOW)  • All database records$(NC)"
	@echo "$(YELLOW)  • All vector embeddings$(NC)"
	@echo ""
	@read -p "Type 'DELETE EVERYTHING' to confirm: " -r; \
	echo; \
	if [ "$$REPLY" = "DELETE EVERYTHING" ]; then \
		$(DC) down -v; \
		echo "$(GREEN)✓ All containers and volumes removed!$(NC)"; \
	else \
		echo "$(YELLOW)Aborted. No changes made.$(NC)"; \
	fi

clean-cache:
	@echo "$(BLUE)$(BOLD)🧹 Cleaning Docker build cache...$(NC)"
	@docker builder prune -f
	@echo "$(GREEN)✓ Build cache cleaned!$(NC)"

prune:
	@echo "$(BLUE)$(BOLD)🧹 Pruning unused Docker resources...$(NC)"
	@docker system prune -af --volumes
	@echo "$(GREEN)✓ System pruned!$(NC)"

# ============================================
# Development
# ============================================

dev:
	@echo "$(BLUE)$(BOLD)🔧 Starting in development mode (hot reload enabled)...$(NC)"
	@$(DC_DEV) up

test:
	@echo "$(BLUE)$(BOLD)🧪 Running tests...$(NC)"
	@docker exec -it $(CONTAINER_API) pytest -v

lint:
	@echo "$(BLUE)$(BOLD)🔍 Running linters...$(NC)"
	@docker exec -it $(CONTAINER_API) flake8 app/
	@docker exec -it $(CONTAINER_API) black --check app/

format:
	@echo "$(BLUE)$(BOLD)✨ Formatting code with Black...$(NC)"
	@docker exec -it $(CONTAINER_API) black app/
	@echo "$(GREEN)✓ Code formatted!$(NC)"

validate-compose:
	@echo "$(BLUE)$(BOLD)✅ Validating docker-compose.yml...$(NC)"
	@$(DC) config --quiet && echo "$(GREEN)✓ Configuration is valid!$(NC)" || echo "$(RED)✗ Configuration has errors!$(NC)"

export-env:
	@echo "$(BLUE)$(BOLD)📤 Exporting resolved environment...$(NC)"
	@$(DC) config > docker-compose.resolved.yml
	@echo "$(GREEN)✓ Exported to docker-compose.resolved.yml$(NC)"

# ============================================
# System Setup
# ============================================

check-deps:
	@echo "$(BLUE)$(BOLD)🔍 Checking system dependencies...$(NC)"
	@echo ""
	@command -v docker >/dev/null 2>&1 && echo "$(GREEN)✓ Docker installed$(NC)" || echo "$(RED)✗ Docker not found$(NC)"
	@command -v docker-compose >/dev/null 2>&1 && echo "$(GREEN)✓ Docker Compose installed$(NC)" || echo "$(RED)✗ Docker Compose not found$(NC)"
	@command -v nvidia-smi >/dev/null 2>&1 && echo "$(GREEN)✓ NVIDIA drivers installed$(NC)" || echo "$(YELLOW)⚠ NVIDIA drivers not found (GPU mode disabled)$(NC)"
	@command -v curl >/dev/null 2>&1 && echo "$(GREEN)✓ curl installed$(NC)" || echo "$(RED)✗ curl not found$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 && echo "$(GREEN)✓ Python 3 installed$(NC)" || echo "$(RED)✗ Python 3 not found$(NC)"
	@command -v make >/dev/null 2>&1 && echo "$(GREEN)✓ Make installed$(NC)" || echo "$(RED)✗ Make not found$(NC)"
	@echo ""

check-gpu:
	@echo "$(BLUE)$(BOLD)🎮 Checking GPU availability...$(NC)"
	@echo ""
	@nvidia-smi 2>/dev/null || echo "$(RED)✗ No NVIDIA GPU detected or drivers not installed$(NC)"

install-nvidia:
	@echo "$(BLUE)$(BOLD)📦 Installing NVIDIA Container Toolkit...$(NC)"
	@distribution=$$(. /etc/os-release;echo $$ID$$VERSION_ID); \
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg; \
	curl -s -L https://nvidia.github.io/libnvidia-container/$$distribution/libnvidia-container.list | \
		sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
		sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list; \
	sudo apt-get update; \
	sudo apt-get install -y nvidia-container-toolkit; \
	sudo nvidia-ctk runtime configure --runtime=docker; \
	sudo systemctl restart docker; \
	echo "$(GREEN)✓ NVIDIA Container Toolkit installed!$(NC)"

# ============================================
# Maintenance
# ============================================

update-images:
	@echo "$(BLUE)$(BOLD)🔄 Updating Docker images...$(NC)"
	@$(DC) pull
	@echo "$(GREEN)✓ Images updated!$(NC)"

disk-usage:
	@echo "$(BLUE)$(BOLD)💾 Docker disk usage:$(NC)"
	@echo ""
	@docker system df -v

network-inspect:
	@echo "$(BLUE)$(BOLD)🌐 Inspecting Docker network...$(NC)"
	@echo ""
	@docker network inspect rag_network

volume-inspect:
	@echo "$(BLUE)$(BOLD)📦 Docker volumes:$(NC)"
	@echo ""
	@docker volume ls | grep rag

# ============================================
# Benchmarking
# ============================================

benchmark:
	@echo "$(BLUE)$(BOLD)🔬 Running performance benchmark...$(NC)"
	@echo "$(YELLOW)Sending 10 requests to API...$(NC)"
	@echo ""
	@for i in {1..10}; do \
		echo "$(CYAN)Request $$i:$(NC)"; \
		time curl -s -X POST "http://localhost:8000/chat" \
			-H "Content-Type: application/json" \
			-d '{"session_id":"test","message":"What is RAG?","use_rag":false}' > /dev/null 2>&1; \
		echo ""; \
	done
	@echo "$(GREEN)✓ Benchmark complete!$(NC)"

stress-test:
	@echo "$(RED)$(BOLD)⚠️  Running stress test (100 concurrent requests)...$(NC)"
	@for i in {1..100}; do \
		curl -s http://localhost:8000/health > /dev/null & \
	done; \
	wait; \
	echo "$(GREEN)✓ Stress test complete!$(NC)"

# ============================================
# Quick Start
# ============================================

quickstart: check-deps build up
	@echo ""
	@echo "$(BLUE)$(BOLD)⏳ Starting quick setup...$(NC)"
	@echo "$(YELLOW)Waiting for services to initialize (30 seconds)...$(NC)"
	@sleep 30
	@echo ""
	@echo "$(BLUE)$(BOLD)📥 Pulling AI models...$(NC)"
	@$(MAKE) pull-models
	@echo ""
	@echo "$(GREEN)$(BOLD)╔═══════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)$(BOLD)║                                                                           ║$(NC)"
	@echo "$(GREEN)$(BOLD)║                    RAG SYSTEM IS READY! 🎉                               ║$(NC)"
	@echo "$(GREEN)$(BOLD)║                                                                           ║$(NC)"
	@echo "$(GREEN)$(BOLD)╚═══════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)🌐 Access the System:$(NC)"
	@echo "$(CYAN)  • UI:          $(WHITE)http://localhost:8501$(NC)"
	@echo "$(CYAN)  • API:         $(WHITE)http://localhost:8000$(NC)"
	@echo "$(CYAN)  • API Docs:    $(WHITE)http://localhost:8000/docs$(NC)"
	@echo "$(CYAN)  • Attu:        $(WHITE)http://localhost:3000$(NC)"
	@echo "$(CYAN)  • MinIO:       $(WHITE)http://localhost:9001$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)📝 Next Steps:$(NC)"
	@echo "$(WHITE)  1. Open the UI at http://localhost:8501$(NC)"
	@echo "$(WHITE)  2. Create a new chat session$(NC)"
	@echo "$(WHITE)  3. Upload documents via File Manager$(NC)"
	@echo "$(WHITE)  4. Start asking questions!$(NC)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)🛠️  Useful Commands:$(NC)"
	@echo "$(CYAN)  make logs      $(WHITE)View system logs$(NC)"
	@echo "$(CYAN)  make health    $(WHITE)Check system health$(NC)"
	@echo "$(CYAN)  make status    $(WHITE)View container status$(NC)"
	@echo "$(CYAN)  make help      $(WHITE)Show all available commands$(NC)"
	@echo ""

# ============================================
# End of Makefile
# ============================================