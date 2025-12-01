"""
RAG V2 - Database Layer with Connection Pooling & Retry Logic
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional, Generator
import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import connection as Connection
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """PostgreSQL connection pool with auto-retry"""
    
    def __init__(self, max_retries=10, retry_delay=3):
        self.db_url = os.getenv("DATABASE_URL")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Establish connection pool with exponential backoff"""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"🔌 Connecting to PostgreSQL (Attempt {attempt}/{self.max_retries})...")
                self.pool = pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    dsn=self.db_url
                )
                
                # Test connection
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                
                logger.info("✅ PostgreSQL connection pool established")
                return
                
            except Exception as e:
                logger.warning(f"❌ Connection failed: {e}")
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * attempt
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError("Failed to connect to PostgreSQL after max retries")
    
    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Context manager for connection pool"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def close(self):
        """Close all connections"""
        if self.pool:
            self.pool.closeall()
            logger.info("🔒 Database connections closed")


# =========================================
# FILE MANAGEMENT OPERATIONS
# =========================================

class FileRegistry:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def check_duplicate(self, file_hash: str) -> Optional[dict]:
        """Check if file already exists"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, filename, status FROM file_registry WHERE file_hash = %s",
                    (file_hash,)
                )
                return cur.fetchone()
    
    def register_file(self, filename: str, file_hash: str, meta_info: dict = None) -> str:
        """Register new file"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO file_registry (filename, file_hash, meta_info, status)
                    VALUES (%s, %s, %s, 'PENDING')
                    RETURNING id
                    """,
                    (filename, file_hash, meta_info or {})
                )
                return str(cur.fetchone()[0])
    
    def update_status(self, file_id: str, status: str):
        """Update file processing status"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE file_registry SET status = %s WHERE id = %s",
                    (status, file_id)
                )
    
    def list_files(self):
        """Get all files with statistics"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM v_file_stats 
                    ORDER BY created_at DESC
                """)
                return cur.fetchall()
    
    def delete_file(self, file_id: str):
        """Delete file and cascade to chunks"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM file_registry WHERE id = %s", (file_id,))


# =========================================
# DOCUMENT CHUNKS OPERATIONS
# =========================================

class DocumentChunks:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def insert_chunks(self, file_id: str, chunks: list[dict]):
        """Batch insert chunks"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                extras.execute_batch(
                    cur,
                    """
                    INSERT INTO document_chunks 
                    (file_id, chunk_index, content, page_number)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [(file_id, c['index'], c['text'], c.get('page', 0)) for c in chunks]
                )
    
    def keyword_search(self, query: str, limit: int = 10):
        """Full-text search using PostgreSQL FTS"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 
                        dc.id,
                        dc.content,
                        dc.page_number,
                        fr.filename,
                        ts_rank(dc.search_vector, websearch_to_tsquery('english', %s)) as rank
                    FROM document_chunks dc
                    JOIN file_registry fr ON dc.file_id = fr.id
                    WHERE dc.search_vector @@ websearch_to_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, query, limit)
                )
                return cur.fetchall()
    
    def get_chunks_by_file(self, file_id: str):
        """Get all chunks for a file"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, content, page_number, chunk_index
                    FROM document_chunks
                    WHERE file_id = %s
                    ORDER BY chunk_index
                    """,
                    (file_id,)
                )
                return cur.fetchall()


# =========================================
# CHAT SESSION OPERATIONS
# =========================================

class ChatSessions:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def create_session(self, title: str = "New Chat") -> str:
        """Create new chat session"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions (title) VALUES (%s) RETURNING id",
                    (title,)
                )
                return str(cur.fetchone()[0])
    
    def list_sessions(self):
        """Get all sessions ordered by recent activity"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM v_session_stats 
                    ORDER BY updated_at DESC
                    LIMIT 50
                """)
                return cur.fetchall()
    
    def update_title(self, session_id: str, title: str):
        """Update session title"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET title = %s WHERE id = %s",
                    (title, session_id)
                )
    
    def delete_session(self, session_id: str):
        """Delete session and cascade to events"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))


# =========================================
# CHAT EVENTS (INFINITE CONTEXT)
# =========================================

class ChatEvents:
    def __init__(self, db: DatabaseManager):
        self.db = db
    
    def add_event(self, session_id: str, role: str, content: str, 
                  event_type: str = 'NORMAL', visibility: str = 'VISIBLE',
                  model_used: str = None):
        """Add new chat event"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                # Get next sequence number
                cur.execute(
                    "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM chat_events WHERE session_id = %s",
                    (session_id,)
                )
                sequence_num = cur.fetchone()[0]
                
                cur.execute(
                    """
                    INSERT INTO chat_events 
                    (session_id, sequence_num, role, content, event_type, visibility, model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (session_id, sequence_num, role, content, event_type, visibility, model_used)
                )
                return str(cur.fetchone()[0])
    
    def get_visible_history(self, session_id: str, limit: int = 50):
        """Get visible chat history"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at, model_used
                    FROM chat_events
                    WHERE session_id = %s AND visibility = 'VISIBLE'
                    ORDER BY sequence_num DESC
                    LIMIT %s
                    """,
                    (session_id, limit)
                )
                return list(reversed(cur.fetchall()))
    
    def get_context_for_llm(self, session_id: str):
        """Get smart context (including hidden summaries)"""
        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    WITH recent_normal AS (
                        SELECT role, content, sequence_num
                        FROM chat_events
                        WHERE session_id = %s AND event_type = 'NORMAL'
                        ORDER BY sequence_num DESC
                        LIMIT 10
                    ),
                    recent_summaries AS (
                        SELECT role, content, sequence_num
                        FROM chat_events
                        WHERE session_id = %s AND event_type = 'SUMMARY'
                        ORDER BY sequence_num DESC
                        LIMIT 3
                    ),
                    latest_checkpoint AS (
                        SELECT role, content, sequence_num
                        FROM chat_events
                        WHERE session_id = %s AND event_type = 'CHECKPOINT'
                        ORDER BY sequence_num DESC
                        LIMIT 1
                    )
                    SELECT * FROM (
                        SELECT * FROM latest_checkpoint
                        UNION ALL
                        SELECT * FROM recent_summaries
                        UNION ALL
                        SELECT * FROM recent_normal
                    ) combined
                    ORDER BY sequence_num
                    """,
                    (session_id, session_id, session_id)
                )
                return cur.fetchall()
    
    def should_create_summary(self, session_id: str) -> bool:
        """Check if we need summary (every 6 messages = 3 turns)"""
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) 
                    FROM chat_events 
                    WHERE session_id = %s 
                    AND event_type = 'NORMAL'
                    AND sequence_num > (
                        SELECT COALESCE(MAX(sequence_num), 0) 
                        FROM chat_events 
                        WHERE session_id = %s AND event_type IN ('SUMMARY', 'CHECKPOINT')
                    )
                    """,
                    (session_id, session_id)
                )
                count = cur.fetchone()[0]
                return count >= 6  # 3 turns = 6 messages