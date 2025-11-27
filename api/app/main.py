# file: api/app/main.py

import os
import shutil
import logging
import asyncio
from datetime import datetime
from typing import Optional
import uuid
import aiofiles

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from sqlalchemy.exc import IntegrityError

# Internal Modules
from app.database import init_db, get_db, SessionLocal, FileRegistry, ChatSession, ChatEvent
from app import ingest
from app.rag_core import EnhancedRAGv2

# --- SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System V2 - Final",
    version="2.0.0-final",
    description="Production RAG with Hybrid Search, Analytics & Graceful Degradation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

DATA_PATH = os.getenv("DATA_PATH", "./data")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB

# Global RAG engine (initialized in background)
rag_engine: Optional[EnhancedRAGv2] = None


@app.on_event("startup")
async def startup():
    """
    NON-BLOCKING STARTUP
    Critical services start immediately, others in background
    """
    global rag_engine
    
    logger.info("=" * 80)
    logger.info("🚀 RAG SYSTEM V2 STARTUP")
    logger.info("=" * 80)
    
    try:
        # === CRITICAL: Database ===
        logger.info("📦 Initializing database...")
        init_db()
        logger.info("✅ Database ready")
        
        # === CRITICAL: Data directory ===
        os.makedirs(DATA_PATH, exist_ok=True)
        logger.info(f"✅ Data path: {DATA_PATH}")
        
        # === NON-BLOCKING: RAG Engine ===
        logger.info("🤖 Initializing RAG engine (background)...")
        rag_engine = EnhancedRAGv2()
        # Note: RAG engine initializes asynchronously
        
        # === NON-BLOCKING: Ingest system ===
        logger.info("📁 Initializing ingest system (background)...")
        asyncio.create_task(_background_ingest_init())
        
        logger.info("=" * 80)
        logger.info("✅ STARTUP COMPLETE - System running in graceful mode")
        logger.info("⏳ Background initialization in progress...")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical(f"❌ CRITICAL STARTUP FAILURE: {e}")
        raise


async def _background_ingest_init():
    """Background task for non-critical ingest initialization."""
    try:
        await asyncio.sleep(2)  # Let server start first
        
        logger.info("🔄 Running background ingest tasks...")
        
        # Initialize Milvus (with retry)
        success = await asyncio.to_thread(ingest.initialize_ingest_system)
        
        if success:
            logger.info("✅ Background ingest initialization complete")
        else:
            logger.warning("⚠️ Ingest initialization partial failure - system degraded")
            
    except Exception as e:
        logger.error(f"❌ Background ingest failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("👋 RAG System V2 shutting down...")


# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID or None for new")
    message: str = Field(..., min_length=1, max_length=10000)


class UploadResponse(BaseModel):
    status: str
    file_id: str
    processing_status: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    features: dict
    services: dict
    initialization: dict


# --- HELPERS ---
def validate_file_extension(filename: str) -> bool:
    """Check file extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

async def save_upload_file(upload_file: UploadFile, destination: str) -> int:
    """
    Save file asynchronously with size limit enforcement.
    Uses non-blocking I/O to prevent freezing the event loop.
    """
    size = 0
    
    # 1. Use async context manager (Non-blocking open)
    async with aiofiles.open(destination, "wb") as buffer:
        while chunk := await upload_file.read(8192): # 2. Async read from stream
            size += len(chunk)
            
            # 3. Security Check
            if size > MAX_FILE_SIZE:
                # Stop writing immediately to prevent disk filling
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {MAX_FILE_SIZE / (1024*1024):.2f}MB limit"
                )
            
            # 4. Async write to disk (Non-blocking write)
            await buffer.write(chunk)
            
    return size


# --- ENDPOINTS ---

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    COMPREHENSIVE HEALTH CHECK
    Shows initialization status and graceful degradation
    """
    # Check database
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Check RAG engine
    rag_status = "not_initialized"
    if rag_engine:
        if rag_engine.initialized:
            rag_status = "healthy"
        elif rag_engine.initialization_error:
            rag_status = "failed"
        else:
            rag_status = "initializing"
    
    # Check Milvus
    milvus_stats = ingest.get_collection_stats()
    milvus_status = "healthy" if "error" not in milvus_stats else "unhealthy"
    
    # Overall status
    if db_status == "healthy" and rag_status == "healthy":
        overall = "healthy"
    elif db_status == "healthy":
        overall = "degraded"
    else:
        overall = "unhealthy"
    
    return {
        "status": overall,
        "version": "2.0.0-final",
        "features": {
            "hybrid_search": os.getenv("ENABLE_HYBRID_SEARCH", "true") == "true",
            "reranking": os.getenv("ENABLE_RERANKING", "true") == "true",
            "text_to_sql": bool(os.getenv("CLICKHOUSE_URL")),
            "visualization": bool(os.getenv("SUPERSET_BASE_URL"))
        },
        "services": {
            "database": db_status,
            "rag_engine": rag_status,
            "vector_store": milvus_status,
            "milvus_entities": milvus_stats.get("num_entities", 0)
        },
        "initialization": {
            "rag_initialized": rag_engine.initialized if rag_engine else False,
            "error": rag_engine.initialization_error if rag_engine else None
        }
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    IDEMPOTENT FILE UPLOAD
    Checks hash before processing
    """
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(DATA_PATH, unique_filename)
    
    try:
        file_size = await save_upload_file(file, temp_path)
        logger.info(f"📥 Saved: {file.filename} ({file_size} bytes)")
        
        file_hash = ingest.calculate_md5(temp_path)
        
        # Check for duplicate
        try:
            existing = db.query(FileRegistry).filter(
                FileRegistry.file_hash == file_hash
            ).first()
            
            if existing:
                os.remove(temp_path)
                logger.info(f"⏭️ Duplicate: {file.filename} (ID: {existing.id})")
                return UploadResponse(
                    status="exists",
                    file_id=str(existing.id),
                    processing_status=existing.status,
                    message="File already exists in system"
                )
        except Exception as e:
            logger.error(f"❌ Duplicate check failed: {e}")
        
        # Register new file
        new_file = FileRegistry(
            file_hash=file_hash,
            filename=file.filename,
            file_path=temp_path,
            status="PENDING"
        )
        
        try:
            db.add(new_file)
            db.commit()
            db.refresh(new_file)
            logger.info(f"✅ Registered: {file.filename} (ID: {new_file.id})")
            
        except IntegrityError as e:
            # Race condition: file was registered by another request
            db.rollback()
            existing = db.query(FileRegistry).filter(
                FileRegistry.file_hash == file_hash
            ).first()
            
            if existing:
                os.remove(temp_path)
                return UploadResponse(
                    status="exists",
                    file_id=str(existing.id),
                    processing_status=existing.status,
                    message="File registered by concurrent request"
                )
            else:
                raise
        
        # Trigger background processing
        background_tasks.add_task(ingest.process_file_task, str(new_file.id))
        
        return UploadResponse(
            status="uploaded",
            file_id=str(new_file.id),
            processing_status="PENDING",
            message="File queued for processing"
        )
        
    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed"
        )


@app.post("/chat")
async def chat_stream(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    STREAMING CHAT WITH INTENT ROUTING
    Supports: RAG | SQL | Visualization
    """
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # Session management with auto-increment
    session_id = req.session_id
    if not session_id:
        new_sess = ChatSession(title=req.message[:50])
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        session_id = str(new_sess.id)
        logger.info(f"📝 New session: {session_id}")
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
        session.updated_at = datetime.utcnow()
        db.commit()
    
    # Auto-increment sequence_num
    last_seq = db.query(func.max(ChatEvent.sequence_num)).filter(
        ChatEvent.session_id == session_id
    ).scalar() or 0
    
    user_event = ChatEvent(
        session_id=session_id,
        sequence_num=last_seq + 1,
        role='user',
        content=req.message,
        event_type='NORMAL'
    )
    db.add(user_event)
    db.commit()

    # Stream generator
    async def response_generator():
        try:
            yield f"data: {{'type': 'session', 'session_id': '{session_id}'}}\n\n"
            
            full_response = ""
            model_used = None
            metadata = {}
            
            # Process query
            async for chunk, model, meta in rag_engine.process_query(
                db, session_id, req.message
            ):
                full_response += chunk
                model_used = model
                metadata = meta
                
                yield f"data: {{'type': 'text', 'content': {repr(chunk)}}}\n\n"
            
            # Send metadata
            if metadata:
                import json
                meta_json = json.dumps({
                    'intent': metadata.get('intent'),
                    'sql': metadata.get('sql_result', {}).get('sql'),
                    'viz_link': metadata.get('viz_link')
                }, default=str)
                yield f"data: {{'type': 'metadata', 'data': {meta_json}}}\n\n"
            
            # Save assistant message
            db_gen = SessionLocal()
            try:
                bot_seq = db_gen.query(func.max(ChatEvent.sequence_num)).filter(
                    ChatEvent.session_id == session_id
                ).scalar() + 1
                
                bot_event = ChatEvent(
                    session_id=session_id,
                    sequence_num=bot_seq,
                    role='assistant',
                    content=full_response,
                    event_type='NORMAL',
                    model_used=model_used
                )
                db_gen.add(bot_event)
                db_gen.commit()
                
                # Trigger memory consolidation
                background_tasks.add_task(
                    rag_engine.trigger_memory_consolidation,
                    session_id
                )
                
            except Exception as e:
                logger.error(f"❌ Save failed: {e}")
            finally:
                db_gen.close()
            
            yield f"data: {{'type': 'done'}}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            yield f"data: {{'type': 'error', 'message': {repr(str(e))}}}\n\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/sessions")
def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List sessions with pagination."""
    sessions = db.query(ChatSession).order_by(
        desc(ChatSession.updated_at)
    ).limit(limit).offset(offset).all()
    
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "message_count": db.query(func.count(ChatEvent.id)).filter(
                ChatEvent.session_id == s.id,
                ChatEvent.event_type == 'NORMAL'
            ).scalar()
        }
        for s in sessions
    ]


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Get session history."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    
    events = db.query(ChatEvent).filter(
        ChatEvent.session_id == session_id,
        ChatEvent.visibility == 'VISIBLE',
        ChatEvent.event_type == 'NORMAL'
    ).order_by(ChatEvent.sequence_num).all()
    
    return {
        "session_id": str(session.id),
        "title": session.title,
        "messages": [
            {
                "role": e.role,
                "content": e.content,
                "timestamp": e.created_at,
                "model": e.model_used
            }
            for e in events
        ]
    }


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    
    db.delete(session)
    db.commit()
    return None


@app.get("/files")
def list_files(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List files with optional filter."""
    query = db.query(FileRegistry)
    
    if status_filter:
        query = query.filter(FileRegistry.status == status_filter.upper())
    
    files = query.order_by(desc(FileRegistry.created_at)).limit(limit).offset(offset).all()
    
    return [
        {
            "id": str(f.id),
            "filename": f.filename,
            "status": f.status,
            "created_at": f.created_at,
            "error": f.error_log
        }
        for f in files
    ]


@app.get("/stats/milvus")
def get_milvus_stats():
    """Get Milvus collection stats."""
    return ingest.get_collection_stats()


@app.get("/stats/system")
def get_system_stats(db: Session = Depends(get_db)):
    """Get overall system statistics."""
    return {
        "total_files": db.query(func.count(FileRegistry.id)).scalar(),
        "files_completed": db.query(func.count(FileRegistry.id)).filter(
            FileRegistry.status == "COMPLETED"
        ).scalar(),
        "files_failed": db.query(func.count(FileRegistry.id)).filter(
            FileRegistry.status == "FAILED"
        ).scalar(),
        "total_sessions": db.query(func.count(ChatSession.id)).scalar(),
        "total_messages": db.query(func.count(ChatEvent.id)).filter(
            ChatEvent.event_type == 'NORMAL'
        ).scalar(),
        "vector_store": ingest.get_collection_stats()
    }