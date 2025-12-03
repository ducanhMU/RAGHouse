"""
Database layer for the RAG microservice stack.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from psycopg2 import pool
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import RealDictCursor, execute_batch

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe PostgreSQL connection pool with retry logic."""

    def __init__(self, max_retries: int = 10, retry_delay: int = 3) -> None:
        self.dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://rag_user:rag_password@postgres:5432/rag_db",
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pool: Optional[pool.ThreadedConnectionPool] = None
        self._connect_with_retry()

    def _connect_with_retry(self) -> None:
        for attempt in range(1, self.max_retries + 1):
            try:
                self.pool = pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=int(os.getenv("DB_MAX_CONNECTIONS", "20")),
                    dsn=self.dsn,
                )
                with self.get_connection() as conn, conn.cursor() as cur:
                    cur.execute("SELECT 1")
                logger.info("PostgreSQL connection pool ready")
                return
            except Exception as exc:  # pragma: no cover - infra safety net
                logger.warning("PostgreSQL connection failed (%s)", exc)
                if attempt == self.max_retries:
                    raise ConnectionError("Unable to reach PostgreSQL") from exc
                time.sleep(self.retry_delay * attempt)

    @contextmanager
    def get_connection(self) -> Generator[PGConnection, None, None]:
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def close(self) -> None:
        if self.pool:
            self.pool.closeall()
            logger.info("Closed PostgreSQL connections")


class FileRegistryDAO:
    """CRUD helpers for the file_registry table."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def lookup_by_hash(self, file_hash: str) -> Optional[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                "SELECT id, filename, status FROM file_registry WHERE file_hash = %s",
                (file_hash,),
            )
            return cur.fetchone()

    def register(self, filename: str, file_hash: str, meta: Optional[Dict] = None) -> str:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO file_registry (filename, file_hash, status, meta_info)
                VALUES (%s, %s, 'PENDING', %s)
                RETURNING id
                """,
                (filename, file_hash, json.dumps(meta or {})),
            )
            return str(cur.fetchone()[0])

    def update_status(self, file_id: str, status: str) -> None:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE file_registry SET status = %s WHERE id = %s",
                (status.upper(), file_id),
            )

    def upsert_metadata(self, file_id: str, meta: Dict) -> None:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE file_registry
                SET meta_info = COALESCE(meta_info, '{}'::jsonb) || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps(meta), file_id),
            )

    def remove(self, file_id: str) -> None:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM file_registry WHERE id = %s", (file_id,))

    def list(self) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute("SELECT * FROM v_file_stats ORDER BY created_at DESC")
            return cur.fetchall()

    def detail(self, file_id: str) -> Optional[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT fr.*, COUNT(dc.id) AS chunk_count
                FROM file_registry fr
                LEFT JOIN document_chunks dc ON fr.id = dc.file_id
                WHERE fr.id = %s
                GROUP BY fr.id
                """,
                (file_id,),
            )
            return cur.fetchone()

    def status_counts(self) -> Dict[str, int]:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, COUNT(*) FROM file_registry GROUP BY status"
            )
            rows = cur.fetchall()
            resolved = {row[0].lower(): row[1] for row in rows}
            return {
                "pending": resolved.get("pending", 0),
                "processing": resolved.get("processing", 0),
                "completed": resolved.get("completed", 0),
                "failed": resolved.get("failed", 0),
            }


class DocumentChunkDAO:
    """Chunk persistence helpers."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def insert_batch(self, file_id: str, chunks: List[Dict]) -> None:
        payload = [
            (file_id, chunk["index"], chunk["text"], chunk.get("page", 0))
            for chunk in chunks
        ]
        with self.db.get_connection() as conn, conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO document_chunks (file_id, chunk_index, content, page_number)
                VALUES (%s, %s, %s, %s)
                """,
                payload,
            )

    def keyword_search(self, query: str, limit: int) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT
                    dc.id,
                    dc.content,
                    dc.page_number,
                    fr.filename,
                    ts_rank(dc.search_vector, websearch_to_tsquery('english', %s)) AS rank
                FROM document_chunks dc
                JOIN file_registry fr ON fr.id = dc.file_id
                WHERE dc.search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            return cur.fetchall()


class ChatSessionDAO:
    """Chat session CRUD helpers."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def create(self, title: str = "New Chat") -> str:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (title) VALUES (%s) RETURNING id",
                (title,),
            )
            return str(cur.fetchone()[0])

    def list(self, limit: int = 50) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT * FROM v_session_stats
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()

    def delete(self, session_id: str) -> None:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))


class ChatEventDAO:
    """Event sourcing storage for the 3-3 memory system."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def _next_sequence(self, session_id: str) -> int:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM chat_events WHERE session_id = %s",
                (session_id,),
            )
            return cur.fetchone()[0]

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        event_type: str = "NORMAL",
        visibility: str = "VISIBLE",
        model_used: Optional[str] = None,
    ) -> str:
        sequence_num = self._next_sequence(session_id)
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_events
                    (session_id, sequence_num, role, content, event_type, visibility, model_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    session_id,
                    sequence_num,
                    role,
                    content,
                    event_type,
                    visibility,
                    model_used,
                ),
            )
            return str(cur.fetchone()[0])

    def visible_history(self, session_id: str, limit: int = 200) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT role, content, created_at, model_used
                FROM chat_events
                WHERE session_id = %s AND visibility = 'VISIBLE'
                ORDER BY sequence_num DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            return list(reversed(cur.fetchall()))

    def raw_events(self, session_id: str, limit: int = 500) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT role, content, event_type, visibility, sequence_num, created_at
                FROM chat_events
                WHERE session_id = %s
                ORDER BY sequence_num DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            return list(reversed(cur.fetchall()))

    def llm_context(self, session_id: str) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute("SELECT * FROM get_session_context(%s)", (session_id,))
            return cur.fetchall()

    def messages_since_summary(self, session_id: str) -> int:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count_messages_since_last_summary(%s)",
                (session_id,),
            )
            return cur.fetchone()[0]


class SystemStatsDAO:
    """Aggregated statistics for dashboards."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def totals(self) -> Dict[str, int]:
        with self.db.get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM file_registry")
            total_files = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            total_chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chat_events")
            total_events = cur.fetchone()[0]
            return {
                "files": total_files,
                "chunks": total_chunks,
                "messages": total_events,
            }

    def recent_activity(self, limit: int = 25) -> List[Dict]:
        with self.db.get_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                """
                SELECT * FROM v_recent_activity
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()