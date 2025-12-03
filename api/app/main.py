"""
FastAPI gateway implementing the RAG design spec (hybrid search + streaming chat).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import requests

from .database import (
    ChatEventDAO,
    ChatSessionDAO,
    DatabaseManager,
    DocumentChunkDAO,
    FileRegistryDAO,
    SystemStatsDAO,
)
from .ingest import IngestionJob, IngestionWorker
from .rag import RAGEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="RAG Microservice API",
    version="3.0.0",
    description="Hybrid dense+sparse retrieval with 3-3 memory and GPU acceleration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Globals initialised during startup
# ---------------------------------------------------------------------------

DB: Optional[DatabaseManager] = None
FILES: Optional[FileRegistryDAO] = None
CHUNKS: Optional[DocumentChunkDAO] = None
SESSIONS: Optional[ChatSessionDAO] = None
EVENTS: Optional[ChatEventDAO] = None
STATS: Optional[SystemStatsDAO] = None
RAG: Optional[RAGEngine] = None
INGESTION: Optional[IngestionWorker] = None
DATA_DIR: Optional[Path] = None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HealthEnvelope(BaseModel):
    postgres: str
    milvus: str
    models: str
    internet: str


class SessionCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=255)


class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(..., min_length=1, max_length=8000)
    use_rag: bool = True
    filter_expr: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: UUID
    reply: str
    citations: List[Dict]
    model_used: str
    latency_ms: int


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    global DB, FILES, CHUNKS, SESSIONS, EVENTS, STATS, RAG, INGESTION, DATA_DIR

    DATA_DIR = Path(os.getenv("DATA_PATH", "/app/data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    DB = DatabaseManager()
    FILES = FileRegistryDAO(DB)
    CHUNKS = DocumentChunkDAO(DB)
    SESSIONS = ChatSessionDAO(DB)
    EVENTS = ChatEventDAO(DB)
    STATS = SystemStatsDAO(DB)

    RAG = RAGEngine()
    INGESTION = IngestionWorker(
        files=FILES,
        chunks=CHUNKS,
        encoder=RAG.encoder,
        vector_store=RAG.vector_store,
        data_dir=DATA_DIR,
    )
    await INGESTION.start()
    await INGESTION.bootstrap_from_directory()
    logger.info("Startup complete")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if DB:
        DB.close()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _ensure_services():
    if not all([DB, FILES, CHUNKS, SESSIONS, EVENTS, RAG, INGESTION]):
        raise HTTPException(status_code=503, detail="Services not initialised")


# ---------------------------------------------------------------------------
# Health & monitoring
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthEnvelope)
async def health() -> HealthEnvelope:
    _ensure_services()
    statuses = {"postgres": "ok", "milvus": "ok", "models": "ok", "internet": "ok"}
    try:
        with DB.get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:  # pragma: no cover - infrastructure guard
        statuses["postgres"] = "down"

    try:
        RAG.vector_store.collection.num_entities
    except Exception:
        statuses["milvus"] = "down"

    try:
        requests.get("https://www.google.com", timeout=2)
    except Exception:
        statuses["internet"] = "limited"

    return HealthEnvelope(**statuses)


@app.get("/health/db")
async def health_db():
    _ensure_services()
    with DB.get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
    return {"status": "ok", "version": version}


@app.get("/health/vector-db")
async def health_vector_db():
    _ensure_services()
    stats = RAG.vector_store.stats()
    return {"status": "ok", **stats}


@app.get("/stats/system")
async def system_stats():
    _ensure_services()
    return STATS.totals()


@app.get("/stats/milvus")
async def milvus_stats():
    _ensure_services()
    return RAG.vector_store.stats()


@app.get("/features")
async def feature_flags():
    return {
        "hybrid_search": True,
        "reranking": bool(RAG.reranker),
        "importance_boost": True,
        "streaming": True,
        "infinite_memory": True,
    }


@app.get("/system/services")
async def system_services():
    base = os.getenv("BASE_HOST", "localhost")
    return [
        {"name": "API", "url": f"http://{base}:8000", "status": "running"},
        {"name": "Streamlit UI", "url": f"http://{base}:8501", "status": "running"},
        {"name": "Milvus", "url": f"http://{base}:9091/healthz", "status": "running"},
        {"name": "PostgreSQL", "url": f"{base}:5433", "status": "running"},
        {"name": "Ollama", "url": f"http://{base}:11435", "status": "running"},
    ]


# ---------------------------------------------------------------------------
# Session & event APIs
# ---------------------------------------------------------------------------


@app.get("/sessions")
async def list_sessions():
    _ensure_services()
    return SESSIONS.list()


@app.post("/sessions")
async def create_session(payload: SessionCreate):
    _ensure_services()
    session_id = SESSIONS.create(title=payload.title)
    return {"session_id": session_id, "title": payload.title}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: UUID):
    _ensure_services()
    SESSIONS.delete(str(session_id))
    return {"status": "deleted"}


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: UUID, limit: int = 50):
    _ensure_services()
    return EVENTS.visible_history(str(session_id), limit=limit)


@app.get("/sessions/{session_id}/events")
async def session_events(session_id: UUID, limit: int = 200):
    _ensure_services()
    return EVENTS.raw_events(str(session_id), limit=limit)


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------


async def _maybe_summarise(session_id: str) -> None:
    count = EVENTS.messages_since_summary(session_id)
    if count < 6:
        return
    history = EVENTS.visible_history(session_id, limit=10)
    conversation = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-6:]
    )
    prompt = (
        "Summarise the following conversation in 3 bullet points:\n"
        f"{conversation}"
    )
    loop = asyncio.get_running_loop()

    def _generate_summary():
        text = ""
        for chunk in RAG.llm.stream(prompt):
            text += chunk["token"]
        return text

    summary = await loop.run_in_executor(None, _generate_summary)
    EVENTS.append(
        session_id,
        "SYSTEM",
        summary.strip(),
        event_type="SUMMARY",
        visibility="HIDDEN",
    )


def _format_citations(chunks: List[Dict]) -> List[Dict]:
    cites = []
    for chunk in chunks:
        cites.append(
            {
                "file_id": chunk.get("file_id"),
                "page_number": chunk.get("page_number"),
                "importance": chunk.get("importance_score"),
                "excerpt": chunk.get("text", "")[:400],
            }
        )
    return cites


@app.post("/chat", response_model=None)
async def stream_chat(payload: ChatRequest):
    _ensure_services()
    session_id = str(payload.session_id)
    EVENTS.append(session_id, "USER", payload.message)
    context = EVENTS.llm_context(session_id)
    retrieved = RAG.retrieve(payload.message) if payload.use_rag else []
    citations = _format_citations(retrieved)

    async def event_generator() -> AsyncGenerator[Dict, None]:
        assistant_text = ""
        start = time.perf_counter()
        yield {"event": "context", "data": json.dumps({"citations": citations})}
        try:
            for chunk in RAG.stream(
                query=payload.message, retrieved=retrieved, conversation=context
            ):
                assistant_text += chunk["token"]
                yield {"event": "token", "data": chunk["token"]}
                await asyncio.sleep(0)
        finally:
            elapsed = int((time.perf_counter() - start) * 1000)
            EVENTS.append(
                session_id,
                "ASSISTANT",
                assistant_text.strip(),
                model_used=chunk.get("model") if "chunk" in locals() else "unknown",
            )
            await _maybe_summarise(session_id)
            yield {
                "event": "metadata",
                "data": json.dumps(
                    {
                        "latency_ms": elapsed,
                        "model_used": chunk.get("model")
                        if "chunk" in locals()
                        else "unknown",
                    }
                ),
            }

    return EventSourceResponse(event_generator())


@app.post("/sessions/{session_id}/message", response_model=ChatResponse)
async def send_message(session_id: UUID, payload: ChatRequest):
    _ensure_services()
    if str(session_id) != str(payload.session_id):
        raise HTTPException(status_code=400, detail="Session mismatch")

    start = time.perf_counter()
    EVENTS.append(str(session_id), "USER", payload.message)
    context = EVENTS.llm_context(str(session_id))
    retrieved = RAG.retrieve(payload.message) if payload.use_rag else []
    result = RAG.generate(
        query=payload.message, retrieved=retrieved, conversation=context
    )
    EVENTS.append(
        str(session_id),
        "ASSISTANT",
        result["answer"],
        model_used=result["model_used"],
    )
    await _maybe_summarise(str(session_id))
    elapsed = int((time.perf_counter() - start) * 1000)
    return ChatResponse(
        session_id=session_id,
        reply=result["answer"],
        citations=_format_citations(retrieved),
        model_used=result["model_used"],
        latency_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# File ingestion APIs
# ---------------------------------------------------------------------------


@app.get("/files")
async def list_files():
    _ensure_services()
    return FILES.list()


@app.get("/files/status")
async def file_status():
    _ensure_services()
    summary = FILES.status_counts()
    summary["total"] = sum(summary.values())
    return summary


@app.get("/files/{file_id}")
async def file_detail(file_id: UUID):
    _ensure_services()
    detail = FILES.detail(str(file_id))
    if not detail:
        raise HTTPException(status_code=404, detail="File not found")
    return detail


@app.post("/files/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_file(file: UploadFile = File(...)):
    _ensure_services()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    destination = DATA_DIR / file.filename
    destination.write_bytes(await file.read())

    await INGESTION.enqueue(
        IngestionJob(file_path=destination, display_name=file.filename)
    )
    return {"status": "queued", "filename": file.filename}


@app.delete("/files/{file_id}")
async def delete_file(file_id: UUID):
    _ensure_services()
    RAG.vector_store.delete_file(str(file_id))
    FILES.remove(str(file_id))
    return {"status": "deleted"}
