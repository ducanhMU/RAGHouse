# file: api/app/database.py

import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, UniqueConstraint, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base

# --- CONNECTION ---
# Uses the docker service name 'postgres' defined in docker-compose
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELS ---

class FileRegistry(Base):
    __tablename__ = "file_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(32), nullable=False, unique=True, index=True) # MD5 for Idempotency
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    status = Column(String(20), default="PENDING") # PENDING, PROCESSING, COMPLETED, FAILED
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatEvent(Base):
    """
    The Append-Only Log for all chat activities.
    Implements the Memory 3-5 Rule storage.
    """
    __tablename__ = "chat_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Chronological ordering within a session
    sequence_num = Column(Integer, nullable=False) 
    
    role = Column(String(20), nullable=False) # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    
    # 3-5 Rule Classifications
    event_type = Column(String(20), default="NORMAL") # 'NORMAL', 'SUMMARY_3', 'CHECKPOINT_5'
    visibility = Column(String(20), default="VISIBLE") # 'VISIBLE', 'HIDDEN' (Hidden = used for context only)
    
    # Metadata for Analytics
    model_used = Column(String(50), nullable=True) # 'gemini-flash', 'ollama-mistral'
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes for fast retrieval of history
    __table_args__ = (
        Index('idx_session_seq', 'session_id', 'sequence_num'),
    )

def init_db():
    Base.metadata.create_all(bind=engine)