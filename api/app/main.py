import os
import shutil
import logging
import uvicorn
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Import internal modules
from app.rag_core import ConversationalRAG
import app.ingest as ingest

# --- Configuration ---
DATA_PATH = os.getenv("DATA_PATH", "./data")

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Financial RAG API",
    description="Backend API for RAG system handling Chat and Data Ingestion."
)

# --- Global RAG System Instance ---
# We initialize this once when the server starts.
rag_system = None

@app.on_event("startup")
def startup_event():
    global rag_system
    try:
        rag_system = ConversationalRAG()
        logging.info("RAG System initialized successfully.")
    except Exception as e:
        logging.critical(f"Failed to initialize RAG System: {e}")

# --- Pydantic Models ---

class ChatHistoryItem(BaseModel):
    role: str # 'user' or 'assistant' (Matched with Streamlit)
    content: str

class ChatRequest(BaseModel):
    input: str
    chat_history: List[ChatHistoryItem] = []

class ChatResponse(BaseModel):
    answer: str

class GenericResponse(BaseModel):
    status: str
    message: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "RAG API is running."}

@app.post("/upload", response_model=GenericResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    1. Receives files from UI.
    2. Saves them to the shared volume (DATA_PATH).
    3. Triggers background ingestion.
    """
    try:
        # Ensure data directory exists
        os.makedirs(DATA_PATH, exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            logging.info(f"Saved file: {file.filename}")

        # Trigger Ingestion in Background (Non-blocking)
        # We call the 'ingest_docs' function from ingest.py
        logging.info("Triggering background ingestion task...")
        background_tasks.add_task(ingest.ingest_docs)
        
        return GenericResponse(
            status="success",
            message=f"Uploaded {len(saved_files)} files. Ingestion started in background."
        )

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main Chat Endpoint.
    Receives user input + history, invokes RAG Agent, returns answer.
    """
    global rag_system
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")

    # Convert Pydantic models to LangChain Message objects
    chat_history_messages: List[BaseMessage] = []
    for item in request.chat_history:
        if item.role == "user":
            chat_history_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            chat_history_messages.append(AIMessage(content=item.content))

    try:
        # Invoke the RAG Agent
        result = rag_system.invoke(request.input, chat_history_messages)
        answer = result.get("answer", "I couldn't generate an answer.")
        return ChatResponse(answer=answer)
    except Exception as e:
        logging.error(f"Error during chat invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)