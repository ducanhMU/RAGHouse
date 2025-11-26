# file: api/app/main.py

import os
import shutil
import logging
from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

# Internal Modules
from app.database import init_db, get_db, FileRegistry, ChatSession, ChatEvent
from app import ingest
from app.rag_core import HybridRAG

# --- SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Version 1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.getenv("DATA_PATH", "./data")
rag_engine = HybridRAG()

@app.on_event("startup")
def startup():
    init_db()
    os.makedirs(DATA_PATH, exist_ok=True)
    ingest.resume_stuck_files()
    logger.info("RAG System V1 Started.")

# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Idempotent Upload: Checks MD5 hash before accepting.
    """
    temp_path = os.path.join(DATA_PATH, file.filename)
    
    # 1. Save Temp
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Check Hash
    file_hash = ingest.calculate_md5(temp_path)
    existing = db.query(FileRegistry).filter(FileRegistry.file_hash == file_hash).first()
    
    if existing:
        os.remove(temp_path)
        return {
            "status": "exists", 
            "file_id": str(existing.id), 
            "processing_status": existing.status
        }
    
    # 3. Register & Trigger
    new_file = FileRegistry(
        file_hash=file_hash,
        filename=file.filename,
        file_path=temp_path
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)
    
    background_tasks.add_task(ingest.process_file_task, str(new_file.id))
    
    return {"status": "uploaded", "file_id": str(new_file.id)}

@app.post("/chat")
async def chat_stream(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Streaming Chat Endpoint with Hierarchical Memory & RAG
    """
    # 1. Session Management
    session_id = req.session_id
    if not session_id:
        new_sess = ChatSession(title="New Chat")
        db.add(new_sess)
        db.commit()
        session_id = str(new_sess.id)
    
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

    # 3. Prepare Context
    # A. Memory Context (3-5 Rule)
    memory_ctx = rag_engine.build_hierarchical_context(db, session_id)
    
    # B. RAG Context (Milvus)
    vector_db = ingest.get_vector_store()
    # Search top 5 chunks
    docs = vector_db.similarity_search(req.message, k=5)
    rag_ctx = "\n\n".join([f"[Source: {d.metadata.get('filename')}]: {d.page_content}" for d in docs])

    # 4. Stream Generator
    async def response_generator():
        yield f"meta:{session_id}\n" # Protocol for Frontend
        
        full_response = ""
        
        # Call Hybrid Inference
        async for chunk in rag_engine.generate_stream(memory_ctx, rag_ctx, req.message):
            full_response += chunk
            yield f"text:{chunk}" # Protocol for Frontend
        
        # 5. Save Assistant Message (After stream completes)
        # Re-open session since the generator runs longer
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
                event_type='NORMAL'
            )
            db_gen.add(bot_event)
            db_gen.commit()
            
            # 6. Trigger Memory Consolidation (3-5 Rule) in background
            background_tasks.add_task(rag_engine.trigger_memory_consolidation, session_id)
            
        finally:
            db_gen.close()

    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """List historical sessions for the sidebar"""
    sessions = db.query(ChatSession).order_by(desc(ChatSession.updated_at)).all()
    return [{"id": str(s.id), "title": s.title, "date": s.updated_at} for s in sessions]