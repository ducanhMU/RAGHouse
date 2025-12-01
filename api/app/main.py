"""
RAG V2 - FastAPI Gateway
"""

import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .database import DatabaseManager, FileRegistry, DocumentChunks, ChatSessions, ChatEvents
from .rag_core import RAGEngine, EmbeddingService, MilvusStore
from .ingest import IngestionPipeline

# =========================================
# LOGGING
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================
# FASTAPI APP
# =========================================

app = FastAPI(
    title="RAG V2 Ultimate API",
    description="Hybrid Search + Infinite Context + GPU-Accelerated",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# GLOBAL SERVICES
# =========================================

db_manager: Optional[DatabaseManager] = None
file_registry: Optional[FileRegistry] = None
doc_chunks: Optional[DocumentChunks] = None
chat_sessions: Optional[ChatSessions] = None
chat_events: Optional[ChatEvents] = None
rag_engine: Optional[RAGEngine] = None
ingest_pipeline: Optional[IngestionPipeline] = None

# =========================================
# STARTUP & SHUTDOWN
# =========================================

@app.on_event("startup")
async def startup_event():
    global db_manager, file_registry, doc_chunks, chat_sessions, chat_events
    global rag_engine, ingest_pipeline
    
    logger.info("🚀 Starting RAG V2 Ultimate API...")
    
    # Initialize database
    db_manager = DatabaseManager()
    file_registry = FileRegistry(db_manager)
    doc_chunks = DocumentChunks(db_manager)
    chat_sessions = ChatSessions(db_manager)
    chat_events = ChatEvents(db_manager)
    
    # Initialize RAG components
    embedder = EmbeddingService()
    vector_store = MilvusStore()
    rag_engine = RAGEngine(db_manager, doc_chunks.keyword_search)
    
    # Initialize ingestion pipeline
    ingest_pipeline = IngestionPipeline(
        file_registry, doc_chunks, embedder, vector_store
    )
    
    # Auto-process pending files
    data_path = os.getenv("DATA_PATH", "/app/data")
    ingest_pipeline.process_pending_files(data_path)
    
    logger.info("✅ All systems ready")

@app.on_event("shutdown")
async def shutdown_event():
    if db_manager:
        db_manager.close()
    logger.info("👋 API shutdown complete")

# =========================================
# PYDANTIC MODELS
# =========================================

class HealthResponse(BaseModel):
    postgres: str
    milvus: str
    models: str
    internet: str

class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"

class MessageRequest(BaseModel):
    content: str
    use_rag: bool = True

class MessageResponse(BaseModel):
    reply: str
    source_type: str
    sources: list
    model_used: str
    latency: float

# =========================================
# HEALTH CHECK ENDPOINTS
# =========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health status"""
    status = {
        "postgres": "ok",
        "milvus": "ok",
        "models": "ok",
        "internet": "ok"
    }
    
    # Check Postgres
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except:
        status["postgres"] = "down"
    
    # Check Milvus
    try:
        rag_engine.vector_store.collection.num_entities
    except:
        status["milvus"] = "down"
    
    # Check Gemini
    try:
        import requests
        requests.get("https://www.google.com", timeout=3)
    except:
        status["internet"] = "limited"
    
    return status

@app.get("/health/db")
async def health_db():
    """PostgreSQL health"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
        return {"status": "ok", "version": version}
    except Exception as e:
        raise HTTPException(500, f"DB error: {str(e)}")

@app.get("/health/vector-db")
async def health_vector_db():
    """Milvus health"""
    try:
        entities = rag_engine.vector_store.collection.num_entities
        return {"status": "ok", "total_vectors": entities}
    except Exception as e:
        raise HTTPException(500, f"Milvus error: {str(e)}")

# =========================================
# SESSION ENDPOINTS
# =========================================

@app.get("/sessions")
async def list_sessions():
    """List all chat sessions"""
    return chat_sessions.list_sessions()

@app.post("/sessions")
async def create_session(data: SessionCreate):
    """Create new chat session"""
    session_id = chat_sessions.create_session(data.title)
    return {"session_id": session_id, "title": data.title}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session metadata"""
    sessions = chat_sessions.list_sessions()
    session = next((s for s in sessions if str(s['id']) == session_id), None)
    if not session:
        raise HTTPException(404, "Session not found")
    return session

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    chat_sessions.delete_session(session_id)
    return {"status": "deleted"}

# =========================================
# CHAT ENDPOINTS
# =========================================

@app.get("/sessions/{session_id}/events")
async def get_events(session_id: str):
    """Get chat history"""
    return chat_events.get_visible_history(session_id)

@app.post("/sessions/{session_id}/message", response_model=MessageResponse)
async def send_message(session_id: str, msg: MessageRequest):
    """Send message and get AI response"""
    import time
    start_time = time.time()
    
    # Add user message
    chat_events.add_event(session_id, "USER", msg.content)
    
    # Get context for LLM
    context = chat_events.get_context_for_llm(session_id)
    
    # Perform RAG if enabled
    sources = []
    source_type = "llm_only"
    
    if msg.use_rag:
        sources = rag_engine.hybrid_search(msg.content, top_k=5)
        if sources:
            source_type = "knowledge_base"
    
    # Generate answer
    result = rag_engine.generate_answer(
        query=msg.content,
        context_docs=sources,
        chat_history=context
    )
    
    # Add assistant message
    chat_events.add_event(
        session_id, "ASSISTANT", result["answer"],
        model_used=result["model_used"]
    )
    
    # Check if summary needed
    if chat_events.should_create_summary(session_id):
        logger.info("Creating summary...")
        summary_prompt = f"Summarize the last 3 conversation turns concisely:\n{context}"
        summary = rag_engine.llm.generate(summary_prompt, temperature=0.3)
        chat_events.add_event(
            session_id, "SYSTEM", summary["text"],
            event_type="SUMMARY", visibility="HIDDEN"
        )
    
    latency = time.time() - start_time
    
    return MessageResponse(
        reply=result["answer"],
        source_type=source_type,
        sources=[{
            "text": s["text"][:200] + "...",
            "page": s.get("page_number", 0),
            "score": s.get("rerank_score", 0)
        } for s in sources],
        model_used=result["model_used"],
        latency=round(latency, 2)
    )

# =========================================
# FILE ENDPOINTS
# =========================================

@app.get("/files")
async def list_files():
    """List uploaded files"""
    return file_registry.list_files()

@app.post("/files/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files supported")
    
    # Save file
    data_path = Path(os.getenv("DATA_PATH", "/app/data"))
    data_path.mkdir(exist_ok=True)
    file_path = data_path / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process in background
    background_tasks.add_task(
        ingest_pipeline.ingest_file,
        str(file_path),
        file.filename
    )
    
    return {"status": "processing", "filename": file.filename}

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete file and associated data"""
    # Delete from Milvus
    rag_engine.vector_store.delete_by_file(file_id)
    
    # Delete from Postgres (cascades to chunks)
    file_registry.delete_file(file_id)
    
    return {"status": "deleted"}

@app.get("/files/status")
async def files_status():
    """Get processing status"""
    files = file_registry.list_files()
    return {
        "total": len(files),
        "completed": len([f for f in files if f['status'] == 'completed']),
        "pending": len([f for f in files if f['status'] in ('pending', 'processing')]),
        "failed": len([f for f in files if f['status'] == 'failed'])
    }

# =========================================
# SYSTEM ENDPOINTS
# =========================================

@app.get("/system/services")
async def system_services():
    """List running services"""
    return [
        {"name": "FastAPI", "url": "http://localhost:8000", "status": "running"},
        {"name": "Streamlit UI", "url": "http://localhost:8501", "status": "running"},
        {"name": "PostgreSQL", "url": "localhost:5433", "status": "running"},
        {"name": "Milvus", "url": "localhost:19530", "status": "running"},
        {"name": "Milvus Attu", "url": "http://localhost:3000", "status": "running"},
        {"name": "MinIO", "url": "http://localhost:9001", "status": "running"},
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)