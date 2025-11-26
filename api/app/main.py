# file: api/app/main.py

import os
import shutil
import logging
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

# Internal Modules
from app.database import init_db, get_db, SessionLocal, FileRegistry, ChatSession, ChatEvent
from app import ingest
from app.rag_core import EnhancedRAGv2

# --- SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System V2",
    version="2.0.0",
    description="Enhanced RAG with Hybrid Search, Reranking & Analytics"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

DATA_PATH = os.getenv("DATA_PATH", "./data")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".doc", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Initialize RAG engine
rag_engine: Optional[EnhancedRAGv2] = None


@app.on_event("startup")
async def startup():
    """Initialize database, directories, and RAG engine."""
    global rag_engine
    
    try:
        init_db()
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Resume stuck files
        ingest.resume_stuck_files()
        
        # Initialize Milvus collection with dynamic schema
        ingest.create_collection_with_dynamic_schema()
        
        # Initialize RAG V2 engine
        rag_engine = EnhancedRAGv2()
        
        logger.info("=== RAG System V2 Started Successfully ===")
        logger.info("Features: Hybrid Search | Reranking | Text-to-SQL | Visualization")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("RAG System V2 Shutting Down")


# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session ID")
    message: str = Field(..., min_length=1, max_length=10000)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": None,
                "message": "What was the revenue in Q4?"
            }
        }


class UploadResponse(BaseModel):
    status: str
    file_id: str
    processing_status: str
    message: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    features: list
    services: dict


# --- HELPER FUNCTIONS ---
def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


async def save_upload_file(upload_file: UploadFile, destination: str) -> int:
    """Save uploaded file and return size."""
    size = 0
    with open(destination, "wb") as buffer:
        while chunk := await upload_file.read(8192):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {MAX_FILE_SIZE / (1024*1024)}MB limit"
                )
            buffer.write(chunk)
    return size


# --- ENDPOINTS ---

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check with V2 features."""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Get Milvus stats
    milvus_stats = ingest.get_collection_stats()
    milvus_status = "healthy" if "error" not in milvus_stats else "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and rag_engine else "degraded",
        "version": "2.0.0",
        "features": [
            "Hybrid Search (Vector + BM25)",
            "Cross-Encoder Reranking",
            "Text-to-SQL Analytics",
            "Visualization Integration",
            "Dynamic Schema Support"
        ],
        "services": {
            "database": db_status,
            "rag_engine": "healthy" if rag_engine else "unhealthy",
            "vector_store": milvus_status,
            "milvus_entities": milvus_stats.get("num_entities", 0)
        }
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload document with dynamic schema support."""
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(DATA_PATH, unique_filename)
    
    try:
        file_size = await save_upload_file(file, temp_path)
        logger.info(f"Saved {file.filename} ({file_size} bytes)")
        
        file_hash = ingest.calculate_md5(temp_path)
        
        # Check for duplicates
        existing = db.query(FileRegistry).filter(
            FileRegistry.file_hash == file_hash
        ).first()
        
        if existing:
            os.remove(temp_path)
            return UploadResponse(
                status="exists",
                file_id=str(existing.id),
                processing_status=existing.status,
                message="File already exists"
            )
        
        # Register new file
        new_file = FileRegistry(
            file_hash=file_hash,
            filename=file.filename,
            file_path=temp_path
        )
        db.add(new_file)
        db.commit()
        db.refresh(new_file)
        
        # Trigger async processing
        background_tasks.add_task(ingest.process_file_task, str(new_file.id))
        
        return UploadResponse(
            status="uploaded",
            file_id=str(new_file.id),
            processing_status="PENDING",
            message="File queued for processing with dynamic schema support"
        )
        
    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Upload failed: {e}")
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
    V2 ENHANCED CHAT: Supports RAG, SQL, and Visualization queries
    
    Features:
    - Intent detection (rag/sql/visualization)
    - Hybrid search with reranking
    - Text-to-SQL execution
    - Dashboard link generation
    """
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # Session management
    session_id = req.session_id
    if not session_id:
        new_sess = ChatSession(title=req.message[:50])
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        session_id = str(new_sess.id)
        logger.info(f"Created session: {session_id}")
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
        session.updated_at = datetime.utcnow()
        db.commit()
    
    # Append user message
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
            # Send session metadata
            yield f"data: {{'type': 'session', 'session_id': '{session_id}'}}\n\n"
            
            full_response = ""
            model_used = None
            intent = None
            metadata = {}
            
            # V2: Process query with unified processor
            async for chunk, model, meta in rag_engine.process_query(
                db, session_id, req.message
            ):
                full_response += chunk
                model_used = model
                intent = meta.get('intent')
                metadata = meta
                
                # Stream text
                yield f"data: {{'type': 'text', 'content': {repr(chunk)}}}\n\n"
            
            # Send metadata (intent, SQL results, viz links)
            if metadata:
                import json
                meta_json = json.dumps(metadata, default=str)
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
                
            finally:
                db_gen.close()
            
            # Send completion
            yield f"data: {{'type': 'done', 'intent': '{intent}'}}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
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


@app.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List sessions with pagination."""
    sessions = db.query(ChatSession).order_by(
        desc(ChatSession.updated_at)
    ).limit(limit).offset(offset).all()
    
    result = []
    for s in sessions:
        msg_count = db.query(func.count(ChatEvent.id)).filter(
            ChatEvent.session_id == s.id,
            ChatEvent.event_type == 'NORMAL'
        ).scalar()
        
        result.append(SessionResponse(
            id=str(s.id),
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=msg_count
        ))
    
    return result


@app.get("/sessions/{session_id}/history")
def get_session_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get session history."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
    
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
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
    
    db.delete(session)
    db.commit()
    logger.info(f"Deleted session {session_id}")
    return None


@app.get("/files")
def list_files(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List files."""
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
            "updated_at": f.updated_at,
            "error": f.error_log
        }
        for f in files
    ]


@app.get("/files/{file_id}")
def get_file_status(file_id: str, db: Session = Depends(get_db)):
    """Get file status."""
    file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
    if not file_record:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found")
    
    return {
        "id": str(file_record.id),
        "filename": file_record.filename,
        "status": file_record.status,
        "created_at": file_record.created_at,
        "updated_at": file_record.updated_at,
        "error": file_record.error_log
    }


@app.get("/stats/milvus")
def get_milvus_stats():
    """V2 FEATURE: Get Milvus collection statistics."""
    return ingest.get_collection_stats()


@app.get("/features")
def get_features():
    """V2 FEATURE: List all V2 capabilities."""
    return {
        "version": "2.0.0",
        "features": {
            "hybrid_search": {
                "enabled": True,
                "description": "Combines vector search with BM25 keyword matching",
                "components": ["Dense Vectors (EmbeddingGemma)", "Sparse BM25"]
            },
            "reranking": {
                "enabled": True,
                "model": "bge-reranker-v2-m3",
                "description": "Cross-encoder reranks candidates for better relevance"
            },
            "text_to_sql": {
                "enabled": bool(os.getenv("CLICKHOUSE_URL")),
                "database": "ClickHouse",
                "description": "Natural language to SQL query conversion"
            },
            "visualization": {
                "enabled": bool(os.getenv("SUPERSET_BASE_URL")),
                "platform": "Apache Superset",
                "description": "Dynamic dashboard linking based on query intent"
            },
            "dynamic_schema": {
                "enabled": True,
                "description": "Milvus adaptive schema handles heterogeneous metadata"
            },
            "memory_management": {
                "enabled": True,
                "strategy": "Hierarchical 3-5 Rule",
                "description": "Efficient long-term conversation memory"
            }
        }
    }