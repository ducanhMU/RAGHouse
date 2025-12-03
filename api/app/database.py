"""
Database Models and Configuration
PostgreSQL connection, ORM models, and database schemas.
"""

import os
import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Text,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    UniqueConstraint,
    Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid as uuid_lib

# ============================================
# Database Configuration
# ============================================

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@postgres:5432/rag_db"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


# ============================================
# Enums
# ============================================

class FileStatus(enum.Enum):
    """File processing status"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MessageRole(enum.Enum):
    """Message role in conversation"""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class EventType(enum.Enum):
    """Event type for 3-3 memory mechanism"""
    NORMAL = "NORMAL"           # Regular chat message
    SUMMARY = "SUMMARY"         # Short summary after 3 message pairs
    CHECKPOINT = "CHECKPOINT"   # Master summary after 3 summaries


class Visibility(enum.Enum):
    """Message visibility in UI"""
    VISIBLE = "VISIBLE"     # Shown in chat UI
    HIDDEN = "HIDDEN"       # Hidden from UI (used for summaries/internal state)


# ============================================
# ORM Models
# ============================================

class FileRegistry(Base):
    """
    File Registry Table
    Tracks uploaded documents and their processing status.
    """
    __tablename__ = "file_registry"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid_lib.uuid4,
        nullable=False
    )
    
    file_hash = Column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
        comment="MD5 hash for deduplication"
    )
    
    filename = Column(
        String(255),
        nullable=False,
        comment="Original filename"
    )
    
    status = Column(
        SQLEnum(FileStatus),
        default=FileStatus.PENDING,
        nullable=False,
        index=True,
        comment="Processing status"
    )
    
    meta_info = Column(
        JSONB,
        default=dict,
        nullable=False,
        comment="Flexible metadata (pages, chunks, author, etc.)"
    )
    
    created_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        nullable=False,
        index=True,
        comment="File registration timestamp"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_file_status', 'status'),
        Index('idx_file_created', 'created_at'),
        Index('idx_file_meta_gin', 'meta_info', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<FileRegistry(id={self.id}, filename={self.filename}, status={self.status.value})>"


class ChatSession(Base):
    """
    Chat Sessions Table
    Stores overarching conversation session metadata.
    """
    __tablename__ = "chat_sessions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid_lib.uuid4,
        nullable=False
    )
    
    title = Column(
        String(255),
        default="New Chat",
        nullable=True,
        comment="Session title (auto-updated from summaries)"
    )
    
    created_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        nullable=False,
        index=True,
        comment="Session creation timestamp"
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        onupdate=datetime.now(datetime.timezone.utc),
        nullable=False,
        index=True,
        comment="Last message timestamp (for sidebar sorting)"
    )
    
    # Relationship to events (one-to-many)
    events = relationship(
        "ChatEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_session_updated', 'updated_at'),
        Index('idx_session_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, title={self.title})>"


class ChatEvent(Base):
    """
    Chat Events Table
    Core memory table for AI context (messages, summaries, checkpoints).
    Implements 3-3 memory mechanism.
    """
    __tablename__ = "chat_events"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid_lib.uuid4,
        nullable=False
    )
    
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to chat_sessions"
    )
    
    sequence_num = Column(
        Integer,
        nullable=False,
        comment="Absolute message order (independent of timestamp)"
    )
    
    role = Column(
        SQLEnum(MessageRole),
        nullable=False,
        comment="Message originator (USER, ASSISTANT, SYSTEM)"
    )
    
    content = Column(
        Text,
        nullable=False,
        comment="Message or summary content"
    )
    
    event_type = Column(
        SQLEnum(EventType),
        default=EventType.NORMAL,
        nullable=False,
        index=True,
        comment="Event type for 3-3 memory mechanism"
    )
    
    visibility = Column(
        SQLEnum(Visibility),
        default=Visibility.VISIBLE,
        nullable=False,
        index=True,
        comment="Controls UI display vs AI-only context"
    )
    
    model_used = Column(
        String(50),
        nullable=True,
        comment="Model used for response (e.g., gemini-2.0-flash)"
    )
    
    created_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        nullable=False,
        comment="Event creation timestamp"
    )
    
    # Relationship to session (many-to-one)
    session = relationship("ChatSession", back_populates="events")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('session_id', 'sequence_num', name='uq_session_sequence'),
        Index('idx_session_sequence', 'session_id', 'sequence_num'),
        Index('idx_event_type', 'event_type'),
        Index('idx_visibility', 'visibility'),
    )
    
    def __repr__(self):
        return (
            f"<ChatEvent(id={self.id}, session_id={self.session_id}, "
            f"seq={self.sequence_num}, role={self.role.value}, "
            f"type={self.event_type.value})>"
        )


# ============================================
# Database Utilities
# ============================================

def get_db():
    """
    Dependency for FastAPI endpoints.
    Provides a database session and ensures cleanup.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    Safe to call multiple times (won't recreate existing tables).
    """
    Base.metadata.create_all(bind=engine)


def drop_all_tables():
    """
    Drop all tables from the database.
    WARNING: This deletes all data!
    Use only for testing or complete reset.
    """
    Base.metadata.drop_all(bind=engine)


def reset_db():
    """
    Drop and recreate all tables.
    WARNING: This deletes all data!
    """
    drop_all_tables()
    init_db()


# ============================================
# SQL Script Generation (for reference)
# ============================================

def generate_sql_script():
    """
    Generate SQL DDL script for manual database setup.
    Returns the SQL as a string.
    """
    from sqlalchemy.schema import CreateTable
    
    sql_script = []
    
    # Add UUID extension
    sql_script.append("-- Enable UUID extension")
    sql_script.append("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
    sql_script.append("")
    
    # Add enum definitions
    sql_script.append("-- Define Enums")
    sql_script.append("CREATE TYPE filestatus AS ENUM ('PENDING','PROCESSING','COMPLETED','FAILED');")
    sql_script.append("CREATE TYPE messagerole AS ENUM ('USER','ASSISTANT','SYSTEM');")
    sql_script.append("CREATE TYPE eventtype AS ENUM ('NORMAL','SUMMARY','CHECKPOINT');")
    sql_script.append("CREATE TYPE visibility AS ENUM ('VISIBLE','HIDDEN');")
    sql_script.append("")
    
    # Generate CREATE TABLE statements
    for table in Base.metadata.sorted_tables:
        sql_script.append(f"-- Table: {table.name}")
        sql_script.append(str(CreateTable(table).compile(engine)).strip() + ";")
        sql_script.append("")
    
    return "\n".join(sql_script)


# ============================================
# Example Usage & Testing
# ============================================

if __name__ == "__main__":
    """
    Example usage and testing of database models.
    Run this script directly to test database connection and models.
    """
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test database connection
    try:
        logger.info("Testing database connection...")
        init_db()
        logger.info("✓ Database initialized successfully")
        
        # Test session creation
        db = SessionLocal()
        result = db.execute("SELECT version()").fetchone()
        logger.info(f"✓ PostgreSQL version: {result[0]}")
        
        # Create a test file record
        test_file = FileRegistry(
            file_hash="test_hash_123",
            filename="test_document.pdf",
            status=FileStatus.PENDING,
            meta_info={"test": True, "pages": 10}
        )
        db.add(test_file)
        db.commit()
        logger.info(f"✓ Created test file record: {test_file.id}")
        
        # Create a test chat session
        test_session = ChatSession(title="Test Session")
        db.add(test_session)
        db.commit()
        logger.info(f"✓ Created test session: {test_session.id}")
        
        # Create a test message
        test_message = ChatEvent(
            session_id=test_session.id,
            sequence_num=1,
            role=MessageRole.USER,
            content="Test message",
            event_type=EventType.NORMAL,
            visibility=Visibility.VISIBLE
        )
        db.add(test_message)
        db.commit()
        logger.info(f"✓ Created test message: {test_message.id}")
        
        # Query test
        file_count = db.query(FileRegistry).count()
        session_count = db.query(ChatSession).count()
        event_count = db.query(ChatEvent).count()
        
        logger.info(f"✓ Database contains:")
        logger.info(f"  - {file_count} files")
        logger.info(f"  - {session_count} sessions")
        logger.info(f"  - {event_count} events")
        
        # Cleanup test data
        db.delete(test_message)
        db.delete(test_session)
        db.delete(test_file)
        db.commit()
        logger.info("✓ Cleaned up test data")
        
        db.close()
        logger.info("✓ All database tests passed!")
        
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}", exc_info=True)
        raise
    
    # Print SQL script for reference
    print("\n" + "=" * 60)
    print("Generated SQL DDL Script:")
    print("=" * 60)
    print(generate_sql_script())