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

from app.rag_core import ConversationalRAG
import app.ingest as ingest

# --- CONFIGURATION ---
DATA_PATH = os.getenv("DATA_PATH", "./data")
DB_FILE = os.path.join(DATA_PATH, "chat_history.db")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Financial RAG API")

# --- 1. DATABASE MANAGER ---
def init_db():
    os.makedirs(DATA_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Added 'summary' column
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, title TEXT, summary TEXT DEFAULT '', created_at TIMESTAMP)''')
    
    # Added 'is_summarized' column
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, is_summarized BOOLEAN DEFAULT 0, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

rag_system = None

@app.on_event("startup")
def startup_event():
    global rag_system
    init_db()
    try:
        rag_system = ConversationalRAG()
        logging.info("RAG System initialized.")
    except Exception:
        logging.critical("Failed to initialize RAG System.")

# --- BACKGROUND TASK: SUMMARIZATION ---
def background_summarize(session_id: str):
    """
    Checks if there are enough unsummarized messages. 
    If yes, condenses them into the session summary.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # 1. Get current summary
    c.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    if not row: 
        conn.close()
        return
    current_summary = row[0] if row[0] else ""

    # 2. Get unsummarized messages (oldest first)
    # We keep the last 2 messages unsummarized to serve as the immediate "buffer" context
    c.execute("SELECT id, role, content FROM messages WHERE session_id = ? AND is_summarized = 0 ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    
    # Threshold: Summarize if we have more than 6 unsummarized messages
    if len(rows) > 6:
        # We summarize all EXCEPT the last 2 (Buffer)
        to_summarize = rows[:-2]
        msg_ids_to_update = [r[0] for r in to_summarize]
        
        # Convert to LangChain Messages
        lc_messages = []
        for r in to_summarize:
            if r[1] == 'user':
                lc_messages.append(HumanMessage(content=r[2]))
            else:
                lc_messages.append(AIMessage(content=r[2]))
        
        # Call LLM to Summarize
        if rag_system:
            new_summary = rag_system.summarize_messages(current_summary, lc_messages)
            
            # Update DB
            c.execute("UPDATE sessions SET summary = ? WHERE id = ?", (new_summary, session_id))
            c.execute(f"UPDATE messages SET is_summarized = 1 WHERE id IN ({','.join(['?']*len(msg_ids_to_update))})", msg_ids_to_update)
            conn.commit()
            logging.info(f"Session {session_id} summarized. New summary length: {len(new_summary)}")
            
    conn.close()

# --- MODELS ---
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

class ChatResponse(BaseModel):
    session_id: str
    answer: str

class GenericResponse(BaseModel):
    status: str
    message: str

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "RAG API with Summary Memory is running."}

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

@app.post("/upload", response_model=GenericResponse)
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    try:
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
        background_tasks.add_task(ingest.ingest_docs)
        return GenericResponse(status="success", message=f"Uploaded {len(saved_files)} files.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # A. Manage Session
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        title = request.input[:40] + "..." if len(request.input) > 40 else request.input
        c.execute("INSERT INTO sessions (id, title, summary, created_at) VALUES (?, ?, ?, ?)", 
                  (session_id, title, "", datetime.now()))
    
    # B. Save User Input
    c.execute("INSERT INTO messages (session_id, role, content, is_summarized, created_at) VALUES (?, ?, ?, ?, ?)",
              (session_id, "user", request.input, 0, datetime.now()))
    conn.commit()

    # C. Get Context (Summary + Buffer)
    # 1. Get Summary
    c.execute("SELECT summary FROM sessions WHERE id = ?", (session_id,))
    row = c.fetchone()
    current_summary = row[0] if row else ""

    # 2. Get Unsummarized Messages (The Buffer)
    # We pass ALL unsummarized messages to the LLM to ensure continuity
    c.execute("SELECT role, content FROM messages WHERE session_id = ? AND is_summarized = 0 ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    
    langchain_history: List[BaseMessage] = []
    for r in rows[:-1]: # Exclude current input
        if r[0] == 'user':
            langchain_history.append(HumanMessage(content=r[1]))
        else:
            langchain_history.append(AIMessage(content=r[1]))

    # D. Invoke RAG (Pass Summary + Buffer)
    try:
        result = rag_system.invoke(request.input, langchain_history, current_summary)
        answer = result.get("answer", "No answer generated.")
    except Exception as e:
        answer = f"Error: {e}"

    # E. Save Assistant Response
    c.execute("INSERT INTO messages (session_id, role, content, is_summarized, created_at) VALUES (?, ?, ?, ?, ?)",
              (session_id, "assistant", answer, 0, datetime.now()))
    conn.commit()
    conn.close()

    # F. Trigger Background Summarization (The "Buffer" Logic)
    # This runs after the response is sent, so it doesn't slow down the user.
    background_tasks.add_task(background_summarize, session_id)

    return ChatResponse(session_id=session_id, answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)