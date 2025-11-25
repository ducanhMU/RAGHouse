import os
import shutil
import logging
import sqlite3
import uuid
import uvicorn
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import internal modules
from app.rag_core import ConversationalRAG
import app.ingest as ingest

# ==================== CONFIGURATION ====================
DATA_PATH = os.getenv("DATA_PATH", "./data")
DB_FILE = os.path.join(DATA_PATH, "chat_history.db")

# Summarization thresholds
BUFFER_SIZE = 6  # Trigger summarization when buffer exceeds this
KEEP_IN_BUFFER = 2  # Keep most recent messages unsummarized

# Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Financial RAG API",
    description="RAG system with streaming, citations, reranking, and session history",
    version="2.0.0"
)

# CORS Middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATABASE ====================

def init_db():
    """Initialize SQLite database with optimized schema."""
    os.makedirs(DATA_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Sessions table with summary column
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT DEFAULT '',
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP
        )
    ''')
    
    # Messages table with summarization tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            is_summarized BOOLEAN DEFAULT 0,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    ''')
    
    # Create indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_messages_summarized ON messages(session_id, is_summarized)')
    
    conn.commit()
    conn.close()
    logging.info("✓ Database initialized successfully")

# ==================== GLOBAL RAG SYSTEM ====================
rag_system: Optional[ConversationalRAG] = None

@app.on_event("startup")
def startup_event():
    """Initialize database and RAG system on startup."""
    global rag_system
    logging.info("=" * 80)
    logging.info("Starting Financial RAG API")
    logging.info("=" * 80)
    
    init_db()
    
    try:
        rag_system = ConversationalRAG()
        logging.info("✓ RAG System ready")
    except Exception as e:
        logging.critical(f"✗ Failed to initialize RAG System: {e}")
        rag_system = None

# ==================== BACKGROUND TASKS ====================

def background_summarize(session_id: str):
    """
    Background task to progressively summarize conversation.
    
    Strategy:
    - Maintains a "buffer" of recent unsummarized messages
    - When buffer exceeds threshold, summarizes oldest messages
    - Keeps most recent messages unsummarized for immediate context
    """
    if not rag_system:
        logging.warning("⚠ RAG system not available for summarization")
        return

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Get current summary
        c.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return
        current_summary = row[0] or ""

        # Get all unsummarized messages (oldest first)
        c.execute(
            "SELECT id, role, content FROM messages WHERE session_id = ? AND is_summarized = 0 ORDER BY id ASC",
            (session_id,)
        )
        rows = c.fetchall()
        
        # Check if summarization is needed
        if len(rows) > BUFFER_SIZE:
            # Summarize all except the most recent KEEP_IN_BUFFER messages
            to_summarize = rows[:-KEEP_IN_BUFFER]
            msg_ids = [r[0] for r in to_summarize]
            
            logging.info(f"Summarizing {len(to_summarize)} messages for session {session_id}")
            
            # Convert to LangChain messages
            lc_messages = []
            for msg_id, role, content in to_summarize:
                if role == 'user':
                    lc_messages.append(HumanMessage(content=content))
                else:
                    lc_messages.append(AIMessage(content=content))
            
            # Call RAG system to summarize
            new_summary = rag_system.summarize_messages(current_summary, lc_messages)
            
            # Update database
            c.execute("UPDATE sessions SET summary = ?, updated_at = ? WHERE id = ?", 
                     (new_summary, datetime.now(), session_id))
            
            # Mark messages as summarized
            placeholders = ','.join(['?'] * len(msg_ids))
            c.execute(f"UPDATE messages SET is_summarized = 1 WHERE id IN ({placeholders})", msg_ids)
            
            conn.commit()
            logging.info(f"✓ Session {session_id} summarized successfully")
            
        conn.close()
        
    except Exception as e:
        logging.error(f"✗ Background summarization failed: {e}")

# ==================== PYDANTIC MODELS ====================

class SessionItem(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: Optional[int] = 0

class MessageItem(BaseModel):
    role: str
    content: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session ID or None for new session")
    input: str = Field(..., min_length=1, description="User's question or message")

class GenericResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# ==================== HELPER FUNCTIONS ====================

def get_session_context(session_id: str) -> tuple[str, List[BaseMessage]]:
    """
    Retrieve session context: summary + unsummarized message buffer.
    
    Returns:
        (current_summary, langchain_history)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get summary
    c.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    current_summary = (row[0] if row and row[0] else "")

    # Get unsummarized messages (the buffer)
    c.execute(
        "SELECT role, content FROM messages WHERE session_id = ? AND is_summarized = 0 ORDER BY id ASC",
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    
    # Convert to LangChain messages, excluding the very last one (current input)
    langchain_history = []
    for role, content in rows[:-1] if len(rows) > 0 else []:
        if role == 'user':
            langchain_history.append(HumanMessage(content=content))
        else:
            langchain_history.append(AIMessage(content=content))
    
    return current_summary, langchain_history

def create_or_get_session(session_id: Optional[str], user_input: str) -> str:
    """Create new session or return existing session ID."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if not session_id:
        session_id = str(uuid.uuid4())
        title = user_input[:50] + "..." if len(user_input) > 50 else user_input
        now = datetime.now()
        c.execute(
            "INSERT INTO sessions (id, title, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, "", now, now)
        )
        conn.commit()
        logging.info(f"✓ Created new session: {session_id}")
    else:
        # Update session timestamp
        c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (datetime.now(), session_id))
        conn.commit()
    
    conn.close()
    return session_id

def save_message(session_id: str, role: str, content: str):
    """Save a message to the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content, is_summarized, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, 0, datetime.now())
    )
    conn.commit()
    conn.close()

# ==================== API ENDPOINTS ====================

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "status": "running",
        "service": "Financial RAG API",
        "version": "2.0.0",
        "features": [
            "✓ Milvus vector database",
            "✓ DuckDuckGo search (via tools)",
            "✓ Query rewriting with context",
            "✓ FlashRank reranking",
            "✓ Streaming responses",
            "✓ Citation tracking",
            "✓ Session history with summarization"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if rag_system else "degraded",
        "rag_system": rag_system is not None,
        "database": os.path.exists(DB_FILE),
        "timestamp": datetime.now().isoformat()
    }

# ==================== SESSION MANAGEMENT ====================

@app.get("/sessions", response_model=List[SessionItem])
def get_sessions():
    """Get all chat sessions with message counts."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get sessions with message counts
    c.execute('''
        SELECT s.id, s.title, s.created_at, COUNT(m.id) as msg_count
        FROM sessions s
        LEFT JOIN messages m ON s.id = m.session_id
        GROUP BY s.id
        ORDER BY s.updated_at DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": str(r[2]),
            "message_count": r[3]
        }
        for r in rows
    ]

@app.get("/sessions/{session_id}/messages", response_model=List[MessageItem])
def get_session_messages(session_id: str):
    """Get all messages for a specific session."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return [
        {
            "role": r[0],
            "content": r[1],
            "created_at": str(r[2])
        }
        for r in rows
    ]

@app.delete("/sessions/{session_id}", response_model=GenericResponse)
def delete_session(session_id: str):
    """Delete a session and all its messages."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Check if session exists
    c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete messages and session (CASCADE should handle messages)
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    
    logging.info(f"✓ Deleted session: {session_id}")
    return GenericResponse(status="success", message=f"Session {session_id} deleted")

# ==================== DOCUMENT INGESTION ====================

@app.post("/upload", response_model=GenericResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload documents for RAG ingestion.
    Documents are saved to DATA_PATH and ingested into Milvus in background.
    """
    try:
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        
        logging.info(f"✓ Uploaded {len(saved_files)} files: {', '.join(saved_files)}")
        
        # Trigger ingestion in background
        background_tasks.add_task(ingest.ingest_docs)
        
        return GenericResponse(
            status="success",
            message=f"Uploaded {len(saved_files)} file(s). Ingestion started in background.",
            data={"files": saved_files}
        )
    except Exception as e:
        logging.error(f"✗ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHAT ENDPOINT (STREAMING) ====================

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint with all features:
    1. Milvus retrieval
    2. Query rewriting with summary + history
    3. FlashRank reranking
    4. Streaming response
    5. Citation tracking
    6. Background summarization
    
    Returns StreamingResponse with custom protocol:
    - "meta:<session_id>" - Session identifier
    - "text:<content>" - Response tokens
    - "sources:<json>" - Citations at the end
    """
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    # A. Create or get session
    session_id = create_or_get_session(request.session_id, request.input)
    
    # B. Save user message
    save_message(session_id, "user", request.input)

    # C. Get context (summary + buffer)
    current_summary, langchain_history = get_session_context(session_id)
    
    logging.info(f"Processing chat for session {session_id}")
    logging.info(f"Summary length: {len(current_summary)} chars")
    logging.info(f"Buffer size: {len(langchain_history)} messages")

    # D. Streaming response generator
    async def response_generator():
        full_answer = ""
        
        try:
            # 1. Send session metadata
            yield f"meta:{session_id}\n"

            # 2. Stream response from RAG core
            for chunk in rag_system.stream_answer(
                user_input=request.input,
                chat_history=langchain_history,
                current_summary=current_summary
            ):
                # Accumulate text for saving
                if chunk.startswith("text:"):
                    full_answer += chunk[5:]  # Remove "text:" prefix
                
                yield chunk

            # 3. Save complete AI response
            if full_answer:
                save_message(session_id, "assistant", full_answer)
                logging.info(f"✓ Saved response ({len(full_answer)} chars)")
                
                # 4. Trigger background summarization
                background_tasks.add_task(background_summarize, session_id)
                
        except Exception as e:
            logging.error(f"✗ Chat streaming failed: {e}")
            yield f"text:Error: {str(e)}\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )

# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )