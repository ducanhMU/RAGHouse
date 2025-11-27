# file: api/app/database.py

import os
import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func

# --- CONNECTION ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ENUMS ---
class FileStatus(str, enum.Enum):
    """File processing status"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class MessageRole(str, enum.Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class EventType(str, enum.Enum):
    """Chat event types for memory management"""
    NORMAL = "NORMAL"
    SUMMARY = "SUMMARY"          # Every 3 turns
    CHECKPOINT = "CHECKPOINT"    # Every 3 summaries

class Visibility(str, enum.Enum):
    """Event visibility"""
    VISIBLE = "VISIBLE"
    HIDDEN = "HIDDEN"

# --- MODELS ---
class FileRegistry(Base):
    """
    File upload and processing registry
    Enhanced with metadata tracking
    """
    __tablename__ = "file_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(32), nullable=False, unique=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)  # NEW: bytes
    
    # Status with Enum
    status = Column(SQLEnum(FileStatus), default=FileStatus.PENDING, nullable=False, index=True)
    error_log = Column(Text, nullable=True)
    
    # Processing metadata (NEW)
    meta_info = Column(JSON, default=dict)  # {pages, chunks, model, etc.}
    
    # Timestamps with timezone awareness
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<FileRegistry {self.filename} ({self.status})>"

class ChatSession(Base):
    """
    Chat session container
    Enhanced with relationship
    """
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), default="New Chat")
    
    # Timestamps with timezone
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship (NEW)
    events = relationship(
        "ChatEvent",
        back_populates="session",
        cascade="all, delete-orphan",  # Auto-delete events when session deleted
        order_by="ChatEvent.sequence_num"
    )
    
    def __repr__(self):
        return f"<ChatSession {self.id} - {self.title}>"

class ChatEvent(Base):
    """
    Append-only log for chat events
    Implements 3-3 Memory Rule
    Enhanced with enums and relationship
    """
    __tablename__ = "chat_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Auto-increment sequence (handled in application layer)
    sequence_num = Column(Integer, nullable=False)
    
    # Enums instead of strings (NEW)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    event_type = Column(SQLEnum(EventType), default=EventType.NORMAL, nullable=False, index=True)
    visibility = Column(SQLEnum(Visibility), default=Visibility.VISIBLE, nullable=False)
    
    # Metadata
    model_used = Column(String(50), nullable=True)
    
    # Timestamp with timezone
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship (NEW)
    session = relationship("ChatSession", back_populates="events")
    
    # Composite indexes for performance
    __table_args__ = (
        # For fetching history
        {'sqlite_autoincrement': True}  # For SQLite compatibility
    )
    
    def __repr__(self):
        return f"<ChatEvent {self.role} - {self.event_type}>"

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)