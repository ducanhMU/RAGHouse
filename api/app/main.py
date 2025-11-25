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
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import internal modules
from app.rag_core import ConversationalRAG
import app.ingest as ingest

# --- CONFIGURATION ---
DATA_PATH = os.getenv("DATA_PATH", "./data")
DB_FILE = os.path.join(DATA_PATH, "chat_history.db")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Financial RAG API")

# --- 1. DATABASE MANAGER (SQLite) ---
def init_db():
    """Initializes SQLite tables."""
    os.makedirs(DATA_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

# --- 2. GLOBAL RAG SYSTEM ---
rag_system = None

@app.on_event("startup")
def startup_event():
    global rag_system
    init_db()
    try:
        rag_system = ConversationalRAG()
        logging.info("RAG System initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize RAG System: {e}")

# --- 3. MODELS ---
class SessionItem(BaseModel):
    id: str
    title: str
    created_at: str

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    input: str

class GenericResponse(BaseModel):
    status: str
    message: str

# --- 4. ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "RAG API with Streaming & Citations is running."}

# --- SESSION MANAGEMENT ---
@app.get("/sessions", response_model=List[SessionItem])
def get_sessions():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": str(r[2])} for r in rows]

@app.get("/sessions/{session_id}/messages", response_model=List[MessageItem])
def get_session_messages(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

# --- DATA INGESTION ---
@app.post("/upload", response_model=GenericResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    try:
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        
        logging.info(f"Uploaded {len(saved_files)} files. Triggering ingest...")
        background_tasks.add_task(ingest.ingest_docs)
        
        return GenericResponse(status="success", message=f"Uploaded {len(saved_files)} files.")
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- CHAT (STREAMING) ---
@app.post("/chat")
async def chat(request: ChatRequest):
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # A. Handle Session ID
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        # Simple title generation based on input length
        title = request.input[:40] + "..." if len(request.input) > 40 else request.input
        c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)", 
                  (session_id, title, datetime.now()))
    
    # B. Save User Message
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (session_id, "user", request.input, datetime.now()))
    conn.commit()

    # C. Fetch Context History (Limit 10 for sliding window)
    c.execute("""
        SELECT role, content 
        FROM messages 
        WHERE session_id = ? 
        ORDER BY id DESC 
        LIMIT 10
    """, (session_id,))
    rows = c.fetchall()
    conn.close() # Close connection early, generator will open its own
    
    # Reverse to restore chronological order
    rows = rows[::-1] 

    # Convert to LangChain Messages
    langchain_history: List[BaseMessage] = []
    for r in rows: 
        if r[1] == request.input: 
            continue # Skip duplicate (current input)
        if r[0] == 'user':
            langchain_history.append(HumanMessage(content=r[1]))
        else:
            langchain_history.append(AIMessage(content=r[1]))

    # D. Generator for Streaming Response
    async def response_generator():
        full_answer = ""
        
        # Send Session ID as the first event (optional metadata)
        yield f"meta:{session_id}\n"

        # Stream from RAG Core
        for chunk in rag_system.stream_answer(request.input, langchain_history):
            # Protocol: chunk starts with 'text:' or 'sources:'
            if chunk.startswith("text:"):
                # Accumulate text for DB saving
                # Note: chunk includes newline at end, remove prefix
                raw_content = chunk[5:] 
                full_answer += raw_content
            
            yield chunk

        # Save Full AI Response to DB after streaming finishes
        try:
            conn2 = sqlite3.connect(DB_FILE)
            c2 = conn2.cursor()
            c2.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (session_id, "assistant", full_answer, datetime.now()))
            conn2.commit()
            conn2.close()
        except Exception as e:
            logging.error(f"Failed to save AI response: {e}")

    return StreamingResponse(response_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)