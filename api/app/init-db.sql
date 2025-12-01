-- =========================================
-- RAG V2 ULTIMATE - DATABASE INITIALIZATION
-- =========================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For similarity search

-- =========================================
-- ENUMS FOR TYPE SAFETY
-- =========================================

DO $ BEGIN
    CREATE TYPE filestatus AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $;

DO $ BEGIN
    CREATE TYPE messagerole AS ENUM ('USER', 'ASSISTANT', 'SYSTEM');
EXCEPTION
    WHEN duplicate_object THEN null;
END $;

DO $ BEGIN
    CREATE TYPE eventtype AS ENUM ('NORMAL', 'SUMMARY', 'CHECKPOINT');
EXCEPTION
    WHEN duplicate_object THEN null;
END $;

DO $ BEGIN
    CREATE TYPE visibility AS ENUM ('VISIBLE', 'HIDDEN');
EXCEPTION
    WHEN duplicate_object THEN null;
END $;

-- =========================================
-- CHAT MANAGEMENT CLUSTER
-- =========================================

-- Chat Sessions Table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) DEFAULT 'New Chat',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_updated ON chat_sessions(updated_at DESC);

COMMENT ON TABLE chat_sessions IS 'Conversation sessions with automatic timestamp management';
COMMENT ON COLUMN chat_sessions.title IS 'Session title, can be auto-generated from first message';
COMMENT ON COLUMN chat_sessions.updated_at IS 'Auto-updated when new messages arrive';

-- Chat Events Table (Infinite Context Memory)
CREATE TABLE IF NOT EXISTS chat_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    sequence_num INTEGER NOT NULL,
    role messagerole NOT NULL,
    content TEXT NOT NULL,
    event_type eventtype DEFAULT 'NORMAL',
    visibility visibility DEFAULT 'VISIBLE',
    model_used VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_session_sequence UNIQUE(session_id, sequence_num)
);

CREATE INDEX IF NOT EXISTS idx_session_sequence ON chat_events(session_id, sequence_num);
CREATE INDEX IF NOT EXISTS idx_event_type ON chat_events(event_type);
CREATE INDEX IF NOT EXISTS idx_event_visibility ON chat_events(visibility);

COMMENT ON TABLE chat_events IS 'Event-sourced chat history with infinite context support via 3-3 memory architecture';
COMMENT ON COLUMN chat_events.sequence_num IS 'Strictly ordered message sequence (1, 2, 3...) to guarantee chronological order';
COMMENT ON COLUMN chat_events.event_type IS 'NORMAL: regular message, SUMMARY: 3-turn summary, CHECKPOINT: 3-summary mega-summary';
COMMENT ON COLUMN chat_events.visibility IS 'VISIBLE: shown to user in UI, HIDDEN: internal memory context for AI only';
COMMENT ON COLUMN chat_events.model_used IS 'Which LLM generated this response (e.g., gemini-2.0-flash, llama3.2:3b)';

-- =========================================
-- KNOWLEDGE BASE CLUSTER
-- =========================================

-- File Registry Table
CREATE TABLE IF NOT EXISTS file_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_hash VARCHAR(32) NOT NULL UNIQUE,
    filename VARCHAR(255) NOT NULL,
    status filestatus DEFAULT 'PENDING',
    meta_info JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_file_hash ON file_registry(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_status ON file_registry(status);
CREATE INDEX IF NOT EXISTS idx_file_meta_gin ON file_registry USING GIN(meta_info);
CREATE INDEX IF NOT EXISTS idx_file_created ON file_registry(created_at DESC);

COMMENT ON TABLE file_registry IS 'Central registry for all uploaded files with MD5 deduplication';
COMMENT ON COLUMN file_registry.file_hash IS 'MD5 hash for duplicate detection - same file uploaded twice will be detected';
COMMENT ON COLUMN file_registry.status IS 'Processing status: PENDING → PROCESSING → COMPLETED/FAILED';
COMMENT ON COLUMN file_registry.meta_info IS 'Flexible JSONB field for file metadata (pages, size, author, etc.)';

-- Document Chunks Table (Full-Text Search enabled)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id UUID NOT NULL REFERENCES file_registry(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER DEFAULT 0,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_file_chunk UNIQUE(file_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_search_vector ON document_chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_chunk_file_id ON document_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunk_file_index ON document_chunks(file_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunk_page ON document_chunks(page_number);

COMMENT ON TABLE document_chunks IS 'Chunked documents with auto-generated Full-Text Search vectors';
COMMENT ON COLUMN document_chunks.chunk_index IS 'Sequential chunk number within file (0, 1, 2...) for ordering';
COMMENT ON COLUMN document_chunks.search_vector IS 'Auto-generated tsvector for PostgreSQL Full-Text Search (keyword search)';
COMMENT ON COLUMN document_chunks.page_number IS 'Original page number in PDF for citation purposes';

-- =========================================
-- TRIGGERS & AUTOMATION
-- =========================================

-- Function: Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Trigger: Update file_registry.updated_at on modification
DROP TRIGGER IF EXISTS trigger_update_file_registry_updated_at ON file_registry;
CREATE TRIGGER trigger_update_file_registry_updated_at
    BEFORE UPDATE ON file_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function: Update session timestamp when new message arrives
CREATE OR REPLACE FUNCTION update_session_timestamp()
RETURNS TRIGGER AS $
BEGIN
    UPDATE chat_sessions 
    SET updated_at = NOW() 
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Trigger: Auto-update session.updated_at on new message
DROP TRIGGER IF EXISTS trigger_update_session_on_message ON chat_events;
CREATE TRIGGER trigger_update_session_on_message
    AFTER INSERT ON chat_events
    FOR EACH ROW
    EXECUTE FUNCTION update_session_timestamp();

-- =========================================
-- ANALYTICS VIEWS
-- =========================================

-- View: File Statistics
CREATE OR REPLACE VIEW v_file_stats AS
SELECT 
    fr.id,
    fr.filename,
    fr.status,
    fr.created_at,
    fr.updated_at,
    COUNT(dc.id) as chunk_count,
    COALESCE((fr.meta_info->>'pages')::integer, 0) as pages,
    COALESCE((fr.meta_info->>'size_kb')::integer, 0) as size_kb,
    fr.meta_info->>'author' as author
FROM file_registry fr
LEFT JOIN document_chunks dc ON fr.id = dc.file_id
GROUP BY fr.id;

COMMENT ON VIEW v_file_stats IS 'Aggregated statistics for each file including chunk count and metadata';

-- View: Session Statistics
CREATE OR REPLACE VIEW v_session_stats AS
SELECT 
    cs.id,
    cs.title,
    cs.created_at,
    cs.updated_at,
    COUNT(*) FILTER (WHERE ce.role = 'USER') as user_messages,
    COUNT(*) FILTER (WHERE ce.role = 'ASSISTANT') as assistant_messages,
    COUNT(*) FILTER (WHERE ce.event_type = 'SUMMARY') as summaries,
    COUNT(*) FILTER (WHERE ce.event_type = 'CHECKPOINT') as checkpoints,
    COUNT(*) as total_events,
    MAX(ce.created_at) as last_message_at
FROM chat_sessions cs
LEFT JOIN chat_events ce ON cs.id = ce.session_id
GROUP BY cs.id;

COMMENT ON VIEW v_session_stats IS 'Session-level statistics including message counts and activity timestamps';

-- View: Recent Activity
CREATE OR REPLACE VIEW v_recent_activity AS
SELECT 
    'session' as activity_type,
    cs.id as entity_id,
    cs.title as description,
    cs.created_at as timestamp
FROM chat_sessions cs
UNION ALL
SELECT 
    'file_upload' as activity_type,
    fr.id as entity_id,
    fr.filename as description,
    fr.created_at as timestamp
FROM file_registry fr
ORDER BY timestamp DESC
LIMIT 50;

COMMENT ON VIEW v_recent_activity IS 'Combined view of recent system activity (sessions + uploads)';

-- =========================================
-- UTILITY FUNCTIONS
-- =========================================

-- Function: Get session context for LLM (with smart memory)
CREATE OR REPLACE FUNCTION get_session_context(p_session_id UUID, p_limit INTEGER DEFAULT 10)
RETURNS TABLE (
    role messagerole,
    content TEXT,
    sequence_num INTEGER,
    event_type eventtype
) AS $
BEGIN
    RETURN QUERY
    WITH recent_normal AS (
        SELECT ce.role, ce.content, ce.sequence_num, ce.event_type
        FROM chat_events ce
        WHERE ce.session_id = p_session_id 
        AND ce.event_type = 'NORMAL'
        ORDER BY ce.sequence_num DESC
        LIMIT p_limit
    ),
    recent_summaries AS (
        SELECT ce.role, ce.content, ce.sequence_num, ce.event_type
        FROM chat_events ce
        WHERE ce.session_id = p_session_id 
        AND ce.event_type = 'SUMMARY'
        ORDER BY ce.sequence_num DESC
        LIMIT 3
    ),
    latest_checkpoint AS (
        SELECT ce.role, ce.content, ce.sequence_num, ce.event_type
        FROM chat_events ce
        WHERE ce.session_id = p_session_id 
        AND ce.event_type = 'CHECKPOINT'
        ORDER BY ce.sequence_num DESC
        LIMIT 1
    )
    SELECT * FROM (
        SELECT * FROM latest_checkpoint
        UNION ALL
        SELECT * FROM recent_summaries
        UNION ALL
        SELECT * FROM recent_normal
    ) combined
    ORDER BY sequence_num;
END;
$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_session_context IS 'Smart context retrieval: returns checkpoint + summaries + recent messages for efficient LLM prompting';

-- Function: Search documents using Full-Text Search
CREATE OR REPLACE FUNCTION search_documents(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    page_number INTEGER,
    filename VARCHAR,
    rank REAL
) AS $
BEGIN
    RETURN QUERY
    SELECT 
        dc.id,
        dc.content,
        dc.page_number,
        fr.filename,
        ts_rank(dc.search_vector, websearch_to_tsquery('english', p_query)) as rank
    FROM document_chunks dc
    JOIN file_registry fr ON dc.file_id = fr.id
    WHERE dc.search_vector @@ websearch_to_tsquery('english', p_query)
    ORDER BY rank DESC
    LIMIT p_limit;
END;
$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_documents IS 'Full-Text Search across all document chunks with relevance ranking';

-- Function: Count messages since last summary
CREATE OR REPLACE FUNCTION count_messages_since_last_summary(p_session_id UUID)
RETURNS INTEGER AS $
DECLARE
    last_summary_seq INTEGER;
    message_count INTEGER;
BEGIN
    -- Get sequence number of last summary or checkpoint
    SELECT COALESCE(MAX(sequence_num), 0)
    INTO last_summary_seq
    FROM chat_events
    WHERE session_id = p_session_id
    AND event_type IN ('SUMMARY', 'CHECKPOINT');
    
    -- Count normal messages since then
    SELECT COUNT(*)
    INTO message_count
    FROM chat_events
    WHERE session_id = p_session_id
    AND event_type = 'NORMAL'
    AND sequence_num > last_summary_seq;
    
    RETURN message_count;
END;
$ LANGUAGE plpgsql;

COMMENT ON FUNCTION count_messages_since_last_summary IS 'Helper function to determine if summary is needed (3-3 memory rule)';

-- =========================================
-- INITIAL DATA
-- =========================================

-- Create a welcome session (only if no sessions exist)
DO $
BEGIN
    IF NOT EXISTS (SELECT 1 FROM chat_sessions) THEN
        INSERT INTO chat_sessions (title) VALUES ('Welcome to RAG V2 Ultimate');
    END IF;
END $;

-- =========================================
-- GRANT PERMISSIONS
-- =========================================

-- Grant all privileges to the application user
DO $
BEGIN
    -- Grant table permissions
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_user;
    
    -- Grant function execution
    GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO rag_user;
    
    -- Grant usage on types
    GRANT USAGE ON TYPE filestatus TO rag_user;
    GRANT USAGE ON TYPE messagerole TO rag_user;
    GRANT USAGE ON TYPE eventtype TO rag_user;
    GRANT USAGE ON TYPE visibility TO rag_user;
EXCEPTION
    WHEN undefined_object THEN
        RAISE NOTICE 'User rag_user does not exist yet - skipping grants';
END $;

-- =========================================
-- DATABASE METADATA
-- =========================================

COMMENT ON DATABASE rag_db IS 'RAG V2 Ultimate - Production database with hybrid search capabilities';

-- =========================================
-- COMPLETION MESSAGE
-- =========================================

DO $
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'RAG V2 ULTIMATE - Database Initialized';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created: 4 (chat_sessions, chat_events, file_registry, document_chunks)';
    RAISE NOTICE 'Views created: 3 (v_file_stats, v_session_stats, v_recent_activity)';
    RAISE NOTICE 'Functions created: 3 (context retrieval, search, message counting)';
    RAISE NOTICE 'Triggers created: 2 (auto-update timestamps)';
    RAISE NOTICE 'Indexes created: 12 (including FTS GIN indexes)';
    RAISE NOTICE '========================================';
END $;