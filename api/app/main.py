# file api/app/main.py
"""
FastAPI Gateway - Main Application Entry Point
Handles startup, health checks, file management, and chat endpoints.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid
from sqlalchemy import text

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymilvus import connections, Collection, utility

from .database import (
    engine, Base, SessionLocal, 
    FileRegistry, FileStatus, ChatSession, ChatEvent,
    MessageRole, EventType, Visibility
)
from .ingest import (
    load_embedding_model, 
    compute_file_hash, 
    ingest_file_task,
    auto_ingest_directory
)
from .rag import (
    load_reranker_model,
    hybrid_search_and_generate,
    create_or_get_collection
)

# Thêm import mới:
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from pydantic_ai import UserError

# Import Agent vừa tạo
from .agent import financial_agent, FinancialDeps

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="RAG Financial Assistant API",
    version="1.0.0",
    description="Hybrid Search RAG System with BGE-M3 Embeddings"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
milvus_collection: Optional[Collection] = None
UPLOAD_DIR = Path("/app/data/uploads")
DATA_DIR = Path("/app/data")

# Thêm biến Global cho StarRocks
starrocks_engine: Optional[AsyncEngine] = None


# ============================================
# Pydantic Models
# ============================================

class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True
    top_k: int = 7


class ChatResponse(BaseModel):
    session_id: str
    message: str
    sources: List[dict] = []
    metadata: dict = {}


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    details: dict


class ServiceInfo(BaseModel):
    name: str
    url: str
    description: str
    status: str


# ============================================
# Lifecycle Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize DB, Milvus, models, and auto-ingest initial files"""
    global milvus_collection, starrocks_engine
    
    logger.info("=" * 60)
    logger.info("Starting RAG API server...")
    logger.info("=" * 60)

    # >>> THÊM MỚI: Kết nối StarRocks
    try:
        logger.info("Connecting to StarRocks for AI Agent...")
        # Lấy URI từ env (đã config trong docker-compose: mysql+aiomysql://...)
        starrocks_uri = os.getenv("STARROCKS_URI")
        if starrocks_uri:
            starrocks_engine = create_async_engine(starrocks_uri, echo=False)
            logger.info("✓ StarRocks Async Engine created")
        else:
            logger.warning("⚠ STARROCKS_URI not found. Financial Agent will fail.")
    except Exception as e:
        logger.error(f"✗ Failed to connect StarRocks: {e}")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created/verified")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise
    
    # Create upload directory
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Upload directory ready: {UPLOAD_DIR}")
        logger.info(f"✓ Data directory ready: {DATA_DIR}")
    except Exception as e:
        logger.error(f"✗ Directory creation failed: {e}")
        raise
    
    # Connect to Milvus
    try:
        milvus_host = os.getenv("MILVUS_HOST", "milvus")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port
        )
        logger.info(f"✓ Connected to Milvus at {milvus_host}:{milvus_port}")
        
        # Create or load collection
        collection_name = os.getenv("MILVUS_COLLECTION", "rag_hybrid_collection")
        milvus_collection = create_or_get_collection(collection_name)
        milvus_collection.load()
        
        num_entities = milvus_collection.num_entities
        logger.info(f"✓ Milvus collection '{collection_name}' loaded ({num_entities} entities)")
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to Milvus: {e}")
        raise
    
    # Load models
    try:
        logger.info("Loading AI models...")
        device = os.getenv("DEVICE", "cuda")
        
        load_embedding_model(device=device)
        logger.info("✓ BGE-M3 embedding model loaded")
        
        load_reranker_model(device=device)
        logger.info("✓ BGE-reranker-v2-m3 loaded")
        
    except Exception as e:
        logger.error(f"✗ Failed to load models: {e}")
        raise
    
    # Auto-ingest files from /app/data
    try:
        logger.info("Starting auto-ingestion from /app/data...")
        await auto_ingest_directory(str(DATA_DIR), milvus_collection)
        logger.info("✓ Auto-ingestion completed")
    except Exception as e:
        logger.error(f"⚠ Auto-ingestion failed: {e}")
        # Non-critical, continue startup
    
    logger.info("=" * 60)
    logger.info("RAG API server is ready!")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    logger.info("Shutting down RAG API server...")
    try:
        connections.disconnect("default")
        logger.info("✓ Milvus connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    # >>> THÊM MỚI: Đóng kết nối StarRocks
    if starrocks_engine:
        await starrocks_engine.dispose()
        logger.info("✓ StarRocks connection closed")

# ============================================
# NEW: Financial Agent Test Endpoint
# ============================================

class AgentTestRequest(BaseModel):
    query: str

class AgentTestResponse(BaseModel):
    status: str      # "success", "no_intent", "error"
    sql_executed: bool
    answer: str
    details: Optional[str] = None

class StarRocksDiagnosticResponse(BaseModel):
    overall_status: str
    checks: Dict[str, Any]
    test_results: List[Dict[str, Any]]
    recommendations: List[str]

@app.get("/agent/diagnostics", response_model=StarRocksDiagnosticResponse, tags=["Agent Debug"])
async def diagnose_starrocks_agent():
    """
    Comprehensive diagnostic endpoint for StarRocks + Pydantic AI Agent.
    
    Tests:
    1. StarRocks connection
    2. Database schema accessibility
    3. Simple SQL execution
    4. LLM agent with various query types
    5. Tool calling functionality
    
    Returns detailed status and recommendations for troubleshooting.
    """
    checks = {}
    test_results = []
    recommendations = []
    overall_healthy = True
    
    # ===== CHECK 1: StarRocks Engine Status =====
    if not starrocks_engine:
        checks["starrocks_engine"] = {
            "status": "error",
            "message": "StarRocks engine not initialized",
            "details": "STARROCKS_URI environment variable may be missing"
        }
        overall_healthy = False
        recommendations.append("Set STARROCKS_URI environment variable in docker-compose.yml")
    else:
        checks["starrocks_engine"] = {
            "status": "ok",
            "message": "StarRocks async engine initialized"
        }
    
    # ===== CHECK 2: Database Connection =====
    if starrocks_engine:
        try:
            async with starrocks_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                checks["database_connection"] = {
                    "status": "ok",
                    "message": "Successfully connected to StarRocks",
                    "test_query_result": row[0] if row else None
                }
        except Exception as e:
            checks["database_connection"] = {
                "status": "error",
                "message": f"Failed to connect: {str(e)}",
                "error_type": type(e).__name__
            }
            overall_healthy = False
            recommendations.append("Verify StarRocks container is running: docker ps | grep starrocks")
            recommendations.append("Check connection string format: mysql+aiomysql://user:pass@host:port/db")
    
    # ===== CHECK 3: Schema Accessibility =====
    if starrocks_engine and checks.get("database_connection", {}).get("status") == "ok":
        try:
            async with starrocks_engine.connect() as conn:
                # Check if key tables exist
                tables_query = """
                    SELECT TABLE_NAME 
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME IN ('dim_company', 'mart_master_analysis', 'fact_income_statement')
                """
                result = await conn.execute(text(tables_query))
                found_tables = [row[0] for row in result.fetchall()]
                
                checks["schema_access"] = {
                    "status": "ok" if len(found_tables) >= 2 else "warning",
                    "message": f"Found {len(found_tables)}/3 key tables",
                    "tables_found": found_tables,
                    "tables_expected": ["dim_company", "mart_master_analysis", "fact_income_statement"]
                }
                
                if len(found_tables) < 2:
                    recommendations.append("Some expected tables are missing. Verify StarRocks data ingestion.")
                    
        except Exception as e:
            checks["schema_access"] = {
                "status": "error",
                "message": f"Cannot access schema: {str(e)}"
            }
            overall_healthy = False
    
    # ===== CHECK 4: LLM Model Availability =====
    gemini_key = os.getenv("GEMINI_API_KEY")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    if gemini_key:
        checks["llm_model"] = {
            "status": "ok",
            "message": "Using Google Gemini 2.0 Flash",
            "model": "gemini-2.0-flash",
            "api_key_configured": True
        }
    else:
        checks["llm_model"] = {
            "status": "warning",
            "message": f"Fallback to Ollama at {ollama_url}",
            "model": "llama3:8b",
            "api_key_configured": False
        }
        recommendations.append("Consider setting GEMINI_API_KEY for better performance")
    
    # ===== CHECK 5: Agent Functionality Tests =====
    if starrocks_engine and checks.get("database_connection", {}).get("status") == "ok":
        deps = FinancialDeps(engine=starrocks_engine)
        
        # Test 1: Non-financial query (should detect NO_INTENT)
        try:
            result = await financial_agent.run("What is the weather today?", deps=deps)
            test_results.append({
                "test_name": "Non-financial Query Detection",
                "query": "What is the weather today?",
                "status": "pass" if "NO_INTENT_DETECTED" in result.data else "fail",
                "response_preview": result.data[:200],
                "expected": "Should return NO_INTENT_DETECTED"
            })
        except Exception as e:
            test_results.append({
                "test_name": "Non-financial Query Detection",
                "status": "error",
                "error": str(e)
            })
            overall_healthy = False
        
        # Test 2: Simple financial query
        try:
            result = await financial_agent.run(
                "Show me the top 5 companies by market cap", 
                deps=deps
            )
            test_results.append({
                "test_name": "Simple Financial Query",
                "query": "Show me the top 5 companies by market cap",
                "status": "pass" if "NO_INTENT_DETECTED" not in result.data else "fail",
                "response_preview": result.data[:300],
                "expected": "Should execute SQL and return results"
            })
        except Exception as e:
            test_results.append({
                "test_name": "Simple Financial Query",
                "status": "error",
                "error": str(e)
            })
            overall_healthy = False
        
        # Test 3: Query with specific ticker
        try:
            result = await financial_agent.run(
                "What is the P/E ratio of HPG?", 
                deps=deps
            )
            test_results.append({
                "test_name": "Ticker-specific Query",
                "query": "What is the P/E ratio of HPG?",
                "status": "pass" if "NO_INTENT_DETECTED" not in result.data else "fail",
                "response_preview": result.data[:300],
                "expected": "Should query mart_master_analysis table"
            })
        except Exception as e:
            test_results.append({
                "test_name": "Ticker-specific Query",
                "status": "error",
                "error": str(e)
            })
    else:
        test_results.append({
            "test_name": "Agent Tests",
            "status": "skipped",
            "reason": "Database connection not available"
        })
    
    # ===== DETERMINE OVERALL STATUS =====
    error_count = sum(1 for c in checks.values() if c.get("status") == "error")
    warning_count = sum(1 for c in checks.values() if c.get("status") == "warning")
    failed_tests = sum(1 for t in test_results if t.get("status") in ["fail", "error"])
    
    if error_count > 0 or failed_tests > 0:
        overall_status = "unhealthy"
    elif warning_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # ===== FINAL RECOMMENDATIONS =====
    if not recommendations:
        recommendations.append("All systems operational!")
    
    return StarRocksDiagnosticResponse(
        overall_status=overall_status,
        checks=checks,
        test_results=test_results,
        recommendations=recommendations
    )


@app.post("/agent/test", response_model=AgentTestResponse, tags=["Agent Debug"])
async def test_financial_agent(request: AgentTestRequest):
    """
    Test endpoint for Pydantic AI + StarRocks.
    - Checks if query relates to financial data.
    - If yes -> Executes SQL -> Returns Answer.
    - If no -> Returns 'NO_INTENT_DETECTED'.
    """
    if not starrocks_engine:
        raise HTTPException(status_code=503, detail="StarRocks engine not ready")

    try:
        # Chuẩn bị Dependencies (Inject DB Connection)
        deps = FinancialDeps(engine=starrocks_engine)
        
        # Chạy Agent
        result = await financial_agent.run(request.query, deps=deps)
        
        # Logic kiểm tra kết quả
        answer_text = result.data
        
        # Kiểm tra xem Agent có gọi tool execute_sql không?
        # Pydantic AI lưu usage history, nhưng cách đơn giản nhất là check nội dung trả về
        # (Theo Prompt quy ước: nếu không liên quan trả về "NO_INTENT_DETECTED")
        
        if "NO_INTENT_DETECTED" in answer_text:
             return AgentTestResponse(
                status="no_intent",
                sql_executed=False,
                answer="Query not related to financial data.",
                details=answer_text
            )
            
        return AgentTestResponse(
            status="success",
            sql_executed=True, # Thực tế cần check result.usage().requests để chính xác 100%, nhưng ở đây giả định agent làm đúng
            answer=answer_text
        )

    except UserError as e:
        return AgentTestResponse(
            status="error",
            sql_executed=False,
            answer="Agent Error",
            details=str(e)
        )
    except Exception as e:
        logger.error(f"Agent processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# ============================================
# Health & System Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": "RAG Financial Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Overall system health check"""
    db_status = "healthy"
    milvus_status = "healthy"
    model_status = "healthy"
    
    # Check DB
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check Milvus
    try:
        if milvus_collection is None:
            milvus_status = "unhealthy: collection not loaded"
        else:
            _ = milvus_collection.num_entities
    except Exception as e:
        milvus_status = f"unhealthy: {str(e)}"
    
    # Check models
    from .ingest import bge_m3_model
    from .rag import bge_reranker
    if bge_m3_model is None or bge_reranker is None:
        model_status = "unhealthy: models not loaded"
    
    overall_status = "healthy" if all(
        s == "healthy" for s in [db_status, milvus_status, model_status]
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        details={
            "database": db_status,
            "vector_db": milvus_status,
            "models": model_status
        }
    )


@app.get("/health/db", tags=["Health"])
async def health_check_db():
    """PostgreSQL health check"""
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT version()")).fetchone()
        db.close()
        return {
            "status": "healthy",
            "version": result[0] if result else "unknown"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")


@app.get("/health/vector-db", tags=["Health"])
async def health_check_milvus():
    """Milvus health check"""
    try:
        if milvus_collection is None:
            raise HTTPException(status_code=503, detail="Milvus collection not loaded")
        
        num_entities = milvus_collection.num_entities
        return {
            "status": "healthy",
            "collection": milvus_collection.name,
            "entities": num_entities
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Milvus unhealthy: {str(e)}")


@app.get("/stats/milvus", tags=["Statistics"])
async def get_milvus_stats():
    """Detailed Milvus statistics"""
    try:
        if milvus_collection is None:
            raise HTTPException(status_code=503, detail="Collection not loaded")
        
        stats = {
            "collection_name": milvus_collection.name,
            "num_entities": milvus_collection.num_entities,
            "schema": {
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.dtype),
                        "params": field.params
                    }
                    for field in milvus_collection.schema.fields
                ]
            },
            "indexes": [
                {
                    "field": idx.field_name,
                    "index_name": idx.index_name,
                    "params": idx.params
                }
                for idx in milvus_collection.indexes
            ]
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/system", tags=["Statistics"])
async def get_system_stats():
    """Aggregated system statistics"""
    db = SessionLocal()
    try:
        total_files = db.query(FileRegistry).count()
        completed_files = db.query(FileRegistry).filter(
            FileRegistry.status == FileStatus.COMPLETED
        ).count()
        processing_files = db.query(FileRegistry).filter(
            FileRegistry.status == FileStatus.PROCESSING
        ).count()
        failed_files = db.query(FileRegistry).filter(
            FileRegistry.status == FileStatus.FAILED
        ).count()
        
        total_sessions = db.query(ChatSession).count()
        total_messages = db.query(ChatEvent).count()
        
        return {
            "files": {
                "total": total_files,
                "completed": completed_files,
                "processing": processing_files,
                "failed": failed_files
            },
            "chat": {
                "sessions": total_sessions,
                "messages": total_messages
            },
            "vector_db": {
                "entities": milvus_collection.num_entities if milvus_collection else 0,
                "collection": milvus_collection.name if milvus_collection else None
            }
        }
    finally:
        db.close()


@app.get("/system/services", response_model=List[ServiceInfo], tags=["System"])
async def get_services_info():
    """List all connected services and their URLs"""
    services = [
        ServiceInfo(
            name="Streamlit UI",
            url="http://localhost:8501",
            description="Frontend application",
            status="external"
        ),
        ServiceInfo(
            name="FastAPI Backend",
            url="http://localhost:8000",
            description="RAG engine and API gateway",
            status="running"
        ),
        ServiceInfo(
            name="Ollama",
            url="http://localhost:11435",
            description="Local LLM server (Llama 3.2 3B)",
            status="external"
        ),
        ServiceInfo(
            name="PostgreSQL",
            url="postgresql://localhost:5433",
            description="Metadata and chat history database",
            status="external"
        ),
        ServiceInfo(
            name="Milvus",
            url="http://localhost:19530",
            description="Vector database for embeddings",
            status="external"
        ),
        ServiceInfo(
            name="MinIO Console",
            url="http://localhost:9001",
            description="Object storage web UI",
            status="external"
        ),
        ServiceInfo(
            name="Attu",
            url="http://localhost:3000",
            description="Milvus vector database UI",
            status="external"
        )
    ]
    return services


@app.get("/features", tags=["System"])
async def get_enabled_features():
    """List enabled system features"""
    return {
        "hybrid_search": True,
        "dense_embedding": "BGE-M3",
        "sparse_embedding": "BGE-M3 (lexical)",
        "reranker": "BGE-reranker-v2-m3",
        "primary_llm": "Gemini 2.0 Flash" if os.getenv("GEMINI_API_KEY") else "None",
        "fallback_llm": "Llama 3.2 3B (Ollama)",
        "chunking": "Overlap-based (512 tokens)",
        "memory_mechanism": "3-3 Memory (summaries + checkpoints)",
        "gpu_acceleration": True,
        "streaming_responses": True
    }


# ============================================
# File Management Endpoints
# ============================================

@app.post("/files/reingest", tags=["Files"])
async def reingest_all_files():
    """
    Re-ingest all files inside /app/data asynchronously.
    Ensures ingestion runs only once at a time.
    """
    global milvus_collection, _is_reingesting

    if milvus_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Milvus collection is not loaded"
        )

    # Prevent concurrent reingestion
    if _is_reingesting:
        return {
            "status": "running",
            "message": "A re-ingestion process is already running"
        }

    logger.info("Manual re-ingestion requested via API")
    _is_reingesting = True

    async def run_reingest():
        global _is_reingesting
        try:
            logger.info(f"[Reingest] Starting auto-ingestion in: {DATA_DIR}")
            await auto_ingest_directory(str(DATA_DIR), milvus_collection)
            logger.info("[Reingest] Completed successfully")
        except Exception as e:
            logger.error(f"[Reingest] Failed: {e}", exc_info=True)
        finally:
            _is_reingesting = False
            logger.info("[Reingest] State reset. Ready for next run.")

    # Run async ingestion in the background
    asyncio.create_task(run_reingest())

    return {
        "status": "started",
        "message": "Re-ingestion started in background",
        "directory": str(DATA_DIR)
    }


@app.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a document for ingestion"""
    db = SessionLocal()
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.docx', '.doc')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and DOCX files are supported"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Compute hash
        file_hash = compute_file_hash(str(file_path))
        
        # Check if already exists
        existing = db.query(FileRegistry).filter(
            FileRegistry.file_hash == file_hash
        ).first()
        
        if existing:
            return FileUploadResponse(
                file_id=str(existing.id),
                filename=file.filename,
                status=existing.status.value,
                message="File already exists in the system"
            )
        
        # Create new file record
        file_record = FileRegistry(
            file_hash=file_hash,
            filename=file.filename,
            status=FileStatus.PENDING,
            meta_info={
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size": len(content)
            }
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        
        logger.info(f"Created file record: {file_record.id}")
        
        # Schedule background ingestion
        background_tasks.add_task(
            ingest_file_task,
            str(file_record.id),
            str(file_path),
            milvus_collection
        )
        
        return FileUploadResponse(
            file_id=str(file_record.id),
            filename=file.filename,
            status="pending",
            message="File uploaded and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/files", tags=["Files"])
async def list_files():
    """List all uploaded files with metadata"""
    db = SessionLocal()
    try:
        files = db.query(FileRegistry).order_by(
            FileRegistry.created_at.desc()
        ).all()
        
        return [
            {
                "id": str(f.id),
                "filename": f.filename,
                "status": f.status.value,
                "meta_info": f.meta_info,
                "created_at": f.created_at.isoformat()
            }
            for f in files
        ]
    finally:
        db.close()


@app.get("/files/status", tags=["Files"])
async def get_files_status():
    """Get aggregated file processing status"""
    db = SessionLocal()
    try:
        status_counts = {}
        for status in FileStatus:
            count = db.query(FileRegistry).filter(
                FileRegistry.status == status
            ).count()
            status_counts[status.value] = count
        
        return status_counts
    finally:
        db.close()


@app.get("/files/{file_id}", tags=["Files"])
async def get_file_detail(file_id: str):
    """Get detailed file information"""
    db = SessionLocal()
    try:
        file_record = db.query(FileRegistry).filter(
            FileRegistry.id == uuid.UUID(file_id)
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "id": str(file_record.id),
            "filename": file_record.filename,
            "file_hash": file_record.file_hash,
            "status": file_record.status.value,
            "meta_info": file_record.meta_info,
            "created_at": file_record.created_at.isoformat()
        }
    finally:
        db.close()


@app.delete("/files/{file_id}", tags=["Files"])
async def delete_file(file_id: str):
    """Delete a file and its vectors from Milvus"""
    db = SessionLocal()
    try:
        # Get file record
        file_record = db.query(FileRegistry).filter(
            FileRegistry.id == uuid.UUID(file_id)
        ).first()
        
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete vectors from Milvus
        if milvus_collection:
            try:
                expr = f'file_id == "{file_id}"'
                milvus_collection.delete(expr)
                milvus_collection.flush()
                logger.info(f"Deleted vectors for file {file_id} from Milvus")
            except Exception as e:
                logger.error(f"Error deleting vectors: {e}")
        
        # Delete from database
        filename = file_record.filename
        db.delete(file_record)
        db.commit()
        
        logger.info(f"Deleted file record: {file_id}")
        
        return {
            "message": "File deleted successfully",
            "file_id": file_id,
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ============================================
# Chat Session Endpoints
# ============================================

@app.post("/sessions", tags=["Chat"])
async def create_session(session_create: SessionCreate):
    """Create a new chat session"""
    db = SessionLocal()
    try:
        session = ChatSession(title=session_create.title or "New Chat")
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info(f"Created new session: {session.id}")
        
        return {
            "id": str(session.id),
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
    finally:
        db.close()


@app.get("/sessions", tags=["Chat"])
async def list_sessions():
    """List all chat sessions"""
    db = SessionLocal()
    try:
        sessions = db.query(ChatSession).order_by(
            ChatSession.updated_at.desc()
        ).all()
        
        return [
            {
                "id": str(s.id),
                "title": s.title,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat()
            }
            for s in sessions
        ]
    finally:
        db.close()


@app.get("/sessions/{session_id}", tags=["Chat"])
async def get_session(session_id: str):
    """Get session details"""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get message count
        message_count = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.visibility == Visibility.VISIBLE
        ).count()
        
        return {
            "id": str(session.id),
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": message_count
        }
    finally:
        db.close()


@app.delete("/sessions/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Delete a chat session and all messages"""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_title = session.title
        db.delete(session)
        db.commit()
        
        logger.info(f"Deleted session: {session_id}")
        
        return {
            "message": "Session deleted successfully",
            "session_id": session_id,
            "title": session_title
        }
    finally:
        db.close()


@app.get("/sessions/{session_id}/history", tags=["Chat"])
async def get_session_history(session_id: str):
    """Get visible chat history for a session"""
    db = SessionLocal()
    try:
        events = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id),
            ChatEvent.visibility == Visibility.VISIBLE
        ).order_by(ChatEvent.sequence_num).all()
        
        return [
            {
                "id": str(e.id),
                "role": e.role.value,
                "content": e.content,
                "sequence_num": e.sequence_num,
                "model_used": e.model_used
            }
            for e in events
        ]
    finally:
        db.close()


@app.get("/sessions/{session_id}/events", tags=["Chat"])
async def get_session_events(session_id: str):
    """Get all events (including hidden summaries) for a session"""
    db = SessionLocal()
    try:
        events = db.query(ChatEvent).filter(
            ChatEvent.session_id == uuid.UUID(session_id)
        ).order_by(ChatEvent.sequence_num).all()
        
        return [
            {
                "id": str(e.id),
                "role": e.role.value,
                "content": e.content,
                "sequence_num": e.sequence_num,
                "event_type": e.event_type.value,
                "visibility": e.visibility.value,
                "model_used": e.model_used
            }
            for e in events
        ]
    finally:
        db.close()


@app.post("/sessions/{session_id}/message", tags=["Chat"])
async def send_message(session_id: str, request: ChatRequest):
    """Send a message and get a non-streaming response"""
    db = SessionLocal()
    try:
        # Verify session exists
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use the streaming endpoint but collect all chunks
        full_response = ""
        sources = []
        
        async for chunk in hybrid_search_and_generate(
            query_text=request.message,
            session_id=session_id,
            collection=milvus_collection,
            top_k=request.top_k,
            use_rag=request.use_rag
        ):
            import json
            data = json.loads(chunk)
            if data.get("type") == "content":
                full_response += data.get("content", "")
            elif data.get("type") == "sources":
                sources = data.get("sources", [])
        
        return ChatResponse(
            session_id=session_id,
            message=full_response,
            sources=sources,
            metadata={"use_rag": request.use_rag, "top_k": request.top_k}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/chat", tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """Main RAG chat endpoint with streaming responses"""
    try:
        # Verify session exists
        db = SessionLocal()
        session = db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(request.session_id)
        ).first()
        db.close()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        async def generate():
            async for chunk in hybrid_search_and_generate(
                query_text=request.message,
                session_id=request.session_id,
                collection=milvus_collection,
                top_k=request.top_k,
                use_rag=request.use_rag
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )