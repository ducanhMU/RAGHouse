from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from app.rag_core import ConversationalRAG
import app.ingest as ingest
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import logging
import uvicorn

app = FastAPI(
    title="Conversational RAG API",
    description="API for a seperated RAG system (Backend)."
)

# Initialize RAG system once when starting server 
try:
    rag_system = ConversationalRAG()
except Exception as e:
    logging.critical(f"Cannot initialize RAG system: {e}. API will not work properly.")
    rag_system = None

# --- Define Models for API ---

class ChatHistoryItem(BaseModel):
    type: str # 'human' or 'ai'
    content: str

class ChatRequest(BaseModel):
    input: str
    chat_history: List[ChatHistoryItem] = []

class ChatResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    status: str
    message: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "RAG API (FastAPI) is running."}

@app.post("/ingest", response_model=IngestResponse)
async def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Trigger a background process to ingest data.
    """
    logging.info("Endpoint /ingest is called. Adding to background task...")
    
    # Use BackgroundTasks to help FastAPI response immediately, while ingest.main() run below
    background_tasks.add_task(ingest.main)
    
    return IngestResponse(
        status="success",
        message="Ingesting process is triggered to run."
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_system:
        return ChatResponse(answer="Error: RAG system is not initialized.")

    # get chat_history
    chat_history_messages: List[BaseMessage] = []
    for item in request.chat_history:
        if item.type == "human":
            chat_history_messages.append(HumanMessage(content=item.content))
        elif item.type == "ai":
            chat_history_messages.append(AIMessage(content=item.content))

    # Call RAG chain
    result = rag_system.invoke(request.input, chat_history_messages)
    answer = result.get("answer", "Cannot answer.")

    return ChatResponse(answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)