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
from app.rag_core import HybridRAG

# --- SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Version 1",
    version="1.0.0",
    description="Hybrid RAG System with Memory Management"
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

# Initialize RAG engine as singleton
rag_engine: Optional[HybridRAG] = None


@app.on_event("startup")
async def startup():
    """Initialize database, directories, and RAG engine."""
    global rag_engine
    
    try:
        init_db()
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Resume any stuck file processing
        ingest.resume_stuck_files()
        
        # Initialize RAG engine
        rag_engine = HybridRAG()
        
        logger.info("RAG System V1 Started Successfully")
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("RAG System V1 Shutting Down")


# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session ID or None for new session")
    message: str = Field(..., min_length=1, max_length=10000, description="User message")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "What is quantum computing?"
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
    services: dict


# --- HELPER FUNCTIONS ---
def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


async def save_upload_file(upload_file: UploadFile, destination: str) -> int:
    """Save uploaded file and return size in bytes."""
    size = 0
    with open(destination, "wb") as buffer:
        while chunk := await upload_file.read(8192):  # Read in chunks
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024)}MB"
                )
            buffer.write(chunk)
    return size


# --- ENDPOINTS ---

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for monitoring."""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and rag_engine else "degraded",
        "version": "1.0.0",
        "services": {
            "database": db_status,
            "rag_engine": "healthy" if rag_engine else "unhealthy",
            "vector_store": "healthy"  # TODO: Add actual Milvus health check
        }
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.
    - Validates file type and size
    - Uses MD5 hash for idempotency
    - Processes asynchronously
    """
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(DATA_PATH, unique_filename)
    
    try:
        # Save file with size validation
        file_size = await save_upload_file(file, temp_path)
        logger.info(f"Saved file {file.filename} ({file_size} bytes)")
        
        # Calculate hash for idempotency
        file_hash = ingest.calculate_md5(temp_path)
        
        # Check if file already exists
        existing = db.query(FileRegistry).filter(
            FileRegistry.file_hash == file_hash
        ).first()
        
        if existing:
            os.remove(temp_path)
            logger.info(f"File {file.filename} already exists with ID {existing.id}")
            return UploadResponse(
                status="exists",
                file_id=str(existing.id),
                processing_status=existing.status,
                message="File already exists in the system"
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
        
        logger.info(f"File {file.filename} registered with ID {new_file.id}")
        return UploadResponse(
            status="uploaded",
            file_id=str(new_file.id),
            processing_status="PENDING",
            message="File uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        # Clean up file on validation errors
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        # Clean up file on unexpected errors
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )


@app.post("/chat")
async def chat_stream(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Streaming chat endpoint with hierarchical memory and RAG.
    Returns Server-Sent Events (SSE) stream.
    """
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine not initialized"
        )
    
    # 1. Session Management
    session_id = req.session_id
    if not session_id:
        new_sess = ChatSession(title=req.message[:50])  # Use first 50 chars as title
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        session_id = str(new_sess.id)
        logger.info(f"Created new chat session: {session_id}")
    else:
        # Verify session exists
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        # Update session timestamp
        session.updated_at = datetime.utcnow()
        db.commit()
    
    # 2. Append User Message
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
    logger.info(f"User message added to session {session_id}")

    # 3. Prepare Context
    try:
        # A. Memory Context (3-5 Rule)
        memory_ctx = rag_engine.build_hierarchical_context(db, session_id)
        
        # B. RAG Context (Milvus)
        vector_db = ingest.get_vector_store()
        docs = vector_db.similarity_search(req.message, k=5)
        
        if docs:
            rag_ctx = "\n\n".join([
                f"[Source: {d.metadata.get('filename', 'Unknown')}]: {d.page_content}" 
                for d in docs
            ])
        else:
            rag_ctx = "No relevant documents found."
            
    except Exception as e:
        logger.error(f"Context preparation failed: {e}")
        rag_ctx = "Error retrieving knowledge base context."
        memory_ctx = ""

    # 4. Stream Generator
    async def response_generator():
        try:
            # Send session ID first (protocol for frontend)
            yield f"data: {{'type': 'session', 'session_id': '{session_id}'}}\n\n"
            
            full_response = ""
            model_used = None
            
            # Stream AI response
            async for chunk, model in rag_engine.generate_stream(memory_ctx, rag_ctx, req.message):
                full_response += chunk
                model_used = model
                # SSE format
                yield f"data: {{'type': 'text', 'content': {repr(chunk)}}}\n\n"
            
            # 5. Save Assistant Message
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
                
                # 6. Trigger Memory Consolidation
                background_tasks.add_task(
                    rag_engine.trigger_memory_consolidation, 
                    session_id
                )
                
                logger.info(f"Assistant response saved to session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to save assistant message: {e}")
            finally:
                db_gen.close()
            
            # Send completion signal
            yield f"data: {{'type': 'done'}}\n\n"
            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"data: {{'type': 'error', 'message': 'Stream generation failed'}}\n\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List chat sessions with pagination.
    Returns sessions ordered by most recent first.
    """
    sessions = db.query(ChatSession).order_by(
        desc(ChatSession.updated_at)
    ).limit(limit).offset(offset).all()
    
    result = []
    for s in sessions:
        # Count messages in session
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
    """Get full message history for a session."""
    # Verify session exists
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get visible messages only
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
def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a chat session and all associated events."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    db.delete(session)
    db.commit()
    logger.info(f"Deleted session {session_id}")
    
    return None


@app.get("/files", response_model=list[dict])
def list_files(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List uploaded files with optional status filter."""
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
def get_file_status(
    file_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed status of a specific file."""
    file_record = db.query(FileRegistry).filter(FileRegistry.id == file_id).first()
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    return {
        "id": str(file_record.id),
        "filename": file_record.filename,
        "status": file_record.status,
        "created_at": file_record.created_at,
        "updated_at": file_record.updated_at,
        "error": file_record.error_log
    }