# file: api/app/database.py

import os
import uuid
import enum
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM as PG_ENUM
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func

# --- CONNECTION ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@postgres:5432/rag_db")
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=20,        # Increased for high concurrency
    max_overflow=40,
    pool_recycle=3600    # Recycle connections hourly
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- ENUMS (WITH EXPLICIT NAMES) ---
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
    """Chat event types for 3-3 memory rule"""
    NORMAL = "NORMAL"
    SUMMARY = "SUMMARY"          # Every 3 turns (6 messages)
    CHECKPOINT = "CHECKPOINT"    # Every 3 summaries (9 summaries)

class Visibility(str, enum.Enum):
    """Event visibility"""
    VISIBLE = "VISIBLE"
    HIDDEN = "HIDDEN"

# --- MODELS ---
class FileRegistry(Base):
    """
    OPTIMIZED: File registry with JSONB metadata
    NO FileChunk table - all chunks stored in Milvus
    """
    __tablename__ = "file_registry"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_hash = Column(String(32), nullable=False, unique=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)
    
    # Enum with explicit name for migrations
    status = Column(
        PG_ENUM(FileStatus, name='filestatus', create_type=True),
        default=FileStatus.PENDING,
        nullable=False,
        index=True
    )
    error_log = Column(Text, nullable=True)
    
    # JSONB for flexible metadata (queryable!)
    meta_info = Column(JSONB, default=dict, server_default='{}')
    
    # Timestamps with timezone
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # GIN index for JSONB queries
    __table_args__ = (
        Index('idx_file_meta_gin', 'meta_info', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<FileRegistry {self.filename} ({self.status.value})>"

class ChatSession(Base):
    """
    OPTIMIZED: Session with cascade relationship
    """
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), default="New Chat")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship with cascade delete
    events = relationship(
        "ChatEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatEvent.sequence_num",
        lazy="dynamic"  # Query only when needed
    )
    
    def __repr__(self):
        return f"<ChatSession {self.id} - {self.title}>"

class ChatEvent(Base):
    """
    OPTIMIZED: Event log with composite index
    Implements 3-3 Memory Rule
    """
    __tablename__ = "chat_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Auto-increment in application
    sequence_num = Column(Integer, nullable=False)
    
    # Enums with explicit names
    role = Column(
        PG_ENUM(MessageRole, name='messagerole', create_type=True),
        nullable=False
    )
    content = Column(Text, nullable=False)
    event_type = Column(
        PG_ENUM(EventType, name='eventtype', create_type=True),
        default=EventType.NORMAL,
        nullable=False
    )
    visibility = Column(
        PG_ENUM(Visibility, name='visibility', create_type=True),
        default=Visibility.VISIBLE,
        nullable=False
    )
    
    # Metadata
    model_used = Column(String(50), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    session = relationship("ChatSession", back_populates="events")
    
    # COMPOSITE INDEX for fast history queries
    __table_args__ = (
        Index('idx_session_sequence', 'session_id', 'sequence_num'),
        Index('idx_session_type', 'session_id', 'event_type'),
        Index('idx_session_visibility', 'session_id', 'visibility'),
    )
    
    def __repr__(self):
        return f"<ChatEvent {self.role.value} - {self.event_type.value}>"

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    
    # Create indexes if not exist
    with engine.connect() as conn:
        # Ensure GIN index exists for JSONB
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_meta_gin 
            ON file_registry USING gin(meta_info)
        """)
        conn.commit()