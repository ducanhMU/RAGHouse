import os
import shutil
import logging
import sqlite3
import uuid
import uvicorn
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import internal modules
from app.rag_core import ConversationalRAG
import app.ingest as ingest

# --- CONFIGURATION ---
DATA_PATH = os.getenv("DATA_PATH", "./data")
# Database file stored within the data volume to persist across restarts
DB_FILE = os.path.join(DATA_PATH, "chat_history.db")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Financial RAG API")

# --- 1. DATABASE MANAGER (SQLite) ---
def init_db():
    """Initializes the SQLite database and creates tables if they do not exist."""
    os.makedirs(DATA_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Table to store chat sessions (displayed in sidebar)
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    
    # Table to store individual messages within a session
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

# --- 2. GLOBAL RAG SYSTEM ---
rag_system = None

@app.on_event("startup")
def startup_event():
    """Runs on server startup to initialize DB and RAG engine."""
    global rag_system
    init_db()
    try:
        rag_system = ConversationalRAG()
        logging.info("RAG System initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize RAG System: {e}")

# --- 3. PYDANTIC MODELS ---
class SessionItem(BaseModel):
    id: str
    title: str
    created_at: str

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None # If None, a new session is created
    input: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str

class GenericResponse(BaseModel):
    status: str
    message: str

# --- 4. ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "RAG API with SQLite History is running."}

# --- SESSION MANAGEMENT ENDPOINTS ---

@app.get("/sessions", response_model=List[SessionItem])
def get_sessions():
    """Returns a list of all chat sessions, ordered by newest first."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": str(r[2])} for r in rows]

@app.get("/sessions/{session_id}/messages", response_model=List[MessageItem])
def get_session_messages(session_id: str):
    """Returns all messages for a specific session."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

# --- DATA INGESTION ENDPOINT ---

@app.post("/upload", response_model=GenericResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Handles file uploads and triggers background ingestion."""
    try:
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        
        logging.info(f"Uploaded {len(saved_files)} files. Triggering ingest...")
        
        # Trigger ingestion in background so the API responds immediately
        background_tasks.add_task(ingest.ingest_docs)
        
        return GenericResponse(status="success", message=f"Uploaded {len(saved_files)} files.")
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- CHAT ENDPOINT ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat logic:
    1. Manage Session ID (create new if None).
    2. Save User Message to DB.
    3. Retrieve Chat History from DB for Context.
    4. Call RAG Engine.
    5. Save AI Response to DB.
    """
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # A. Handle Session ID
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        # Use first 40 chars of input as title
        title = request.input[:40] + "..." if len(request.input) > 40 else request.input
        c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)", 
                  (session_id, title, datetime.now()))
    
    # B. Save User Message
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (session_id, "user", request.input, datetime.now()))
    conn.commit()

    # C. Fetch History for Context (Exclude the current input which was just added)
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    
    # Convert to LangChain format
    langchain_history: List[BaseMessage] = []
    # We slice [:-1] to exclude the user message we just inserted, 
    # because 'rag_system.invoke' takes the current input separately.
    for r in rows[:-1]: 
        if r[0] == 'user':
            langchain_history.append(HumanMessage(content=r[1]))
        else:
            langchain_history.append(AIMessage(content=r[1]))

    # D. Call RAG Core
    try:
        result = rag_system.invoke(request.input, langchain_history)
        answer = result.get("answer", "No answer generated.")
    except Exception as e:
        answer = f"Error processing request: {e}"

    # E. Save AI Response
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (session_id, "assistant", answer, datetime.now()))
    conn.commit()
    conn.close()

    return ChatResponse(session_id=session_id, answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)