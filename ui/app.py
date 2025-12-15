"""
Streamlit UI for RAG Financial Assistant
Provides chat interface, file management, and system monitoring.

FIXES:
1. Added Reingest button to re-scan api/data folder
2. Lazy session creation - only create session on first user message
3. Fixed duplicate session creation when selecting historical chats
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# ============================================
# Configuration
# ============================================

API_URL = os.getenv("API_URL", "http://api:8000")

# Page configuration
st.set_page_config(
    page_title="RAG Financial Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .status-degraded {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
    }
    .status-pending {
        background-color: #cce5ff;
        color: #004085;
    }
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-completed {
        background-color: #d4edda;
        color: #155724;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #43a047;
    }
    .source-citation {
        font-size: 0.85rem;
        color: #666;
        background-color: #f0f0f0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# API Helper Functions
# ============================================

def check_api_health() -> Dict[str, Any]:
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "details": {"error": f"Status code: {response.status_code}"}}
    except Exception as e:
        return {"status": "unhealthy", "details": {"error": str(e)}}


def get_system_stats() -> Optional[Dict[str, Any]]:
    """Get system statistics"""
    try:
        response = requests.get(f"{API_URL}/stats/system", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching system stats: {e}")
    return None


def get_services_info() -> Optional[List[Dict[str, Any]]]:
    """Get services information"""
    try:
        response = requests.get(f"{API_URL}/system/services", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching services info: {e}")
    return None


def get_features() -> Optional[Dict[str, Any]]:
    """Get enabled features"""
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching features: {e}")
    return None


def list_sessions() -> List[Dict[str, Any]]:
    """List all chat sessions"""
    try:
        response = requests.get(f"{API_URL}/sessions", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
    return []


def create_session(title: str = "New Chat") -> Optional[Dict[str, Any]]:
    """Create a new chat session"""
    try:
        response = requests.post(
            f"{API_URL}/sessions",
            json={"title": title},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error creating session: {e}")
    return None


def delete_session(session_id: str) -> bool:
    """Delete a chat session"""
    try:
        response = requests.delete(f"{API_URL}/sessions/{session_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False


def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a session"""
    try:
        response = requests.get(f"{API_URL}/sessions/{session_id}/history", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching history: {e}")
    return []


def send_message_streaming(session_id: str, message: str, use_rag: bool = True, top_k: int = 7):
    """Send a message and get streaming response"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "session_id": session_id,
                "message": message,
                "use_rag": use_rag,
                "top_k": top_k
            },
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data = json.loads(line_str[6:])
                        yield data
    except Exception as e:
        st.error(f"Error sending message: {e}")


def list_files() -> List[Dict[str, Any]]:
    """List all uploaded files"""
    try:
        response = requests.get(f"{API_URL}/files", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching files: {e}")
    return []


def get_files_status() -> Optional[Dict[str, int]]:
    """Get file processing status counts"""
    try:
        response = requests.get(f"{API_URL}/files/status", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching file status: {e}")
    return None


def upload_file(file) -> Optional[Dict[str, Any]]:
    """Upload a file"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/files/upload", files=files, timeout=300)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {e}")
    return None


def delete_file(file_id: str) -> bool:
    """Delete a file"""
    try:
        response = requests.delete(f"{API_URL}/files/{file_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False


def reingest_files() -> Optional[Dict[str, Any]]:
    """Trigger re-ingestion of all files in api/data folder"""
    try:
        response = requests.post(f"{API_URL}/files/reingest", timeout=300)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error triggering reingest: {e}")
    return None


# ============================================
# UI Helper Functions
# ============================================

def render_status_badge(status: str) -> str:
    """Render a status badge"""
    status_lower = status.lower()
    css_class = f"status-badge status-{status_lower}"
    return f'<span class="{css_class}">{status.upper()}</span>'


def render_chat_message(role: str, content: str):
    """Render a chat message"""
    if role == "USER":
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{content}</div>', 
                   unsafe_allow_html=True)


# ============================================
# Initialize Session State
# ============================================

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "files_refresh" not in st.session_state:
    st.session_state.files_refresh = 0

if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = False


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown('<p class="main-header">🤖 RAG Assistant</p>', unsafe_allow_html=True)
    
    # System Health
    st.subheader("System Health")
    health = check_api_health()
    status = health.get("status", "unknown")
    st.markdown(render_status_badge(status), unsafe_allow_html=True)
    
    with st.expander("Health Details", expanded=False):
        st.json(health)
    
    st.divider()
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Select Page",
        ["💬 Chat", "📁 File Manager", "📊 Dashboard", "⚙️ System Info"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Chat Sessions (only show on chat page)
    if page == "💬 Chat":
        st.subheader("Chat Sessions")
        
        # New Chat button - reset state for new conversation
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.chat_history = []
            st.session_state.session_loaded = False
            st.rerun()
        
        sessions = list_sessions()
        
        for session in sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                # Load existing session
                if st.button(
                    f"📝 {session['title'][:20]}...",
                    key=f"session_{session['id']}",
                    use_container_width=True
                ):
                    # Only load if it's a different session
                    if st.session_state.current_session_id != session['id']:
                        st.session_state.current_session_id = session['id']
                        st.session_state.chat_history = get_session_history(session['id'])
                        st.session_state.session_loaded = True
                        st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session['id']}"):
                    if delete_session(session['id']):
                        st.success("Session deleted")
                        if st.session_state.current_session_id == session['id']:
                            st.session_state.current_session_id = None
                            st.session_state.chat_history = []
                            st.session_state.session_loaded = False
                        st.rerun()


# ============================================
# Main Content Area
# ============================================

if page == "💬 Chat":
    st.title("💬 Chat Interface")
    
    # Chat settings
    with st.expander("⚙️ Chat Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            use_rag = st.checkbox("Use RAG (Document Search)", value=True)
        with col2:
            top_k = st.slider("Documents to retrieve", 1, 20, 7)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            if st.session_state.current_session_id:
                st.info("Continue this conversation by typing a message below.")
            else:
                st.info("Start a new conversation by typing a message below.")
        else:
            for msg in st.session_state.chat_history:
                render_chat_message(msg["role"], msg["content"])
    
    # Chat input
    st.divider()
    user_input = st.chat_input("Ask me anything about your documents...")
    
    if user_input:
        # LAZY SESSION CREATION: Only create session on first message
        if st.session_state.current_session_id is None:
            new_session = create_session(title="New Chat")
            if new_session:
                st.session_state.current_session_id = new_session["id"]
                st.session_state.chat_history = []
                st.session_state.session_loaded = True
            else:
                st.error("Failed to create chat session")
                st.stop()
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "USER",
            "content": user_input
        })
        
        # Display user message
        with chat_container:
            render_chat_message("USER", user_input)
        
        # Stream assistant response
        with chat_container:
            response_placeholder = st.empty()
            full_response = ""
            sources = []
            
            for chunk in send_message_streaming(
                st.session_state.current_session_id,
                user_input,
                use_rag,
                top_k
            ):
                if chunk.get("type") == "content":
                    full_response += chunk.get("content", "")
                    response_placeholder.markdown(
                        f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br>{full_response}</div>',
                        unsafe_allow_html=True
                    )
                elif chunk.get("type") == "sources":
                    sources = chunk.get("sources", [])
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "ASSISTANT",
                "content": full_response
            })
            
            # Display sources if available
            if sources:
                sources_html = "<div class='source-citation'><strong>Sources:</strong><br>"
                for i, source in enumerate(sources, 1):
                    sources_html += f"[{i}] File: {source['file_id']}, Page: {source['page_number']}<br>"
                sources_html += "</div>"
                st.markdown(sources_html, unsafe_allow_html=True)
        
        st.rerun()


elif page == "📁 File Manager":
    st.title("📁 File Manager")
    
    # File upload
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx", "doc"],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload File"):
            with st.spinner("Uploading file..."):
                result = upload_file(uploaded_file)
                if result:
                    st.success(f"✅ {result['message']}")
                    st.session_state.files_refresh += 1
                    time.sleep(1)
                    st.rerun()
    
    st.divider()
    
    # Reingest button
    st.subheader("Re-scan Knowledge Base")
    st.markdown("""
    Click the button below to re-scan the `api/data/` folder for new files.
    This will process any files that were manually added to the folder.
    """)
    
    if st.button("🔄 Reingest All Files", type="primary"):
        with st.spinner("Re-scanning and processing files from api/data/..."):
            result = reingest_files()
            if result:
                st.success(f"✅ {result.get('message', 'Reingest completed')}")
                if result.get('processed_files'):
                    st.info(f"Processed {len(result['processed_files'])} files")
                    with st.expander("View processed files"):
                        for file_info in result['processed_files']:
                            st.write(f"- {file_info}")
                st.session_state.files_refresh += 1
                time.sleep(2)
                st.rerun()
    
    st.divider()
    
    # File status summary
    st.subheader("Processing Status")
    status_data = get_files_status()
    if status_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pending", status_data.get("PENDING", 0))
        with col2:
            st.metric("Processing", status_data.get("PROCESSING", 0))
        with col3:
            st.metric("Completed", status_data.get("COMPLETED", 0))
        with col4:
            st.metric("Failed", status_data.get("FAILED", 0))
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    st.divider()
    
    # File list
    st.subheader("Uploaded Files")
    files = list_files()
    
    if not files:
        st.info("No files uploaded yet.")
    else:
        for file in files:
            with st.expander(f"📄 {file['filename']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Status:** {render_status_badge(file['status'])}", unsafe_allow_html=True)
                    st.write(f"**File ID:** `{file['id']}`")
                    st.write(f"**Uploaded:** {file['created_at']}")
                    
                    # Metadata
                    if file.get('meta_info'):
                        meta = file['meta_info']
                        if meta.get('pages'):
                            st.write(f"**Pages:** {meta['pages']}")
                        if meta.get('chunks'):
                            st.write(f"**Chunks:** {meta['chunks']}")
                        if meta.get('error'):
                            st.error(f"Error: {meta['error']}")
                
                with col2:
                    if st.button("Delete", key=f"delete_file_{file['id']}"):
                        if delete_file(file['id']):
                            st.success("File deleted")
                            time.sleep(1)
                            st.rerun()


elif page == "📊 Dashboard":
    st.title("📊 System Dashboard")
    
    # System stats
    stats = get_system_stats()
    if stats:
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Files", stats['files']['total'])
            st.metric("Completed", stats['files']['completed'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Chat Sessions", stats['chat']['sessions'])
            st.metric("Total Messages", stats['chat']['messages'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Vector Entities", stats['vector_db']['entities'])
            st.write(f"**Collection:** {stats['vector_db']['collection']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Features
    st.subheader("Enabled Features")
    features = get_features()
    if features:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Hybrid Search:** {features.get('hybrid_search')}")
            st.write(f"**Dense Embedding:** {features.get('dense_embedding')}")
            st.write(f"**Sparse Embedding:** {features.get('sparse_embedding')}")
            st.write(f"**Reranker:** {features.get('reranker')}")
        with col2:
            st.write(f"**Primary LLM:** {features.get('primary_llm')}")
            st.write(f"**Fallback LLM:** {features.get('fallback_llm')}")
            st.write(f"**GPU Acceleration:** {features.get('gpu_acceleration')}")
            st.write(f"**Streaming:** {features.get('streaming_responses')}")
    
    st.divider()
    
    # Health details
    st.subheader("Health Check")
    health = check_api_health()
    st.json(health)


elif page == "⚙️ System Info":
    st.title("⚙️ System Information")
    
    st.subheader("Connected Services")
    services = get_services_info()
    if services:
        for service in services:
            with st.expander(f"{service['name']}", expanded=False):
                st.write(f"**URL:** {service['url']}")
                st.write(f"**Description:** {service['description']}")
                st.write(f"**Status:** {render_status_badge(service['status'])}", unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("API Endpoints")
    st.markdown("""
    **Health & System:**
    - `GET /health` - Overall health check
    - `GET /health/db` - PostgreSQL health
    - `GET /health/vector-db` - Milvus health
    - `GET /stats/system` - System statistics
    - `GET /stats/milvus` - Milvus statistics
    - `GET /system/services` - Services info
    - `GET /features` - Enabled features
    
    **File Management:**
    - `POST /files/upload` - Upload document
    - `POST /files/reingest` - Re-scan api/data folder
    - `GET /files` - List all files
    - `GET /files/status` - File processing status
    - `GET /files/{id}` - Get file details
    - `DELETE /files/{id}` - Delete file
    
    **Chat:**
    - `POST /sessions` - Create session
    - `GET /sessions` - List sessions
    - `GET /sessions/{id}` - Get session
    - `DELETE /sessions/{id}` - Delete session
    - `GET /sessions/{id}/history` - Get chat history
    - `POST /chat` - Send message (streaming)
    """)
    
    st.divider()
    
    st.subheader("Documentation")
    st.markdown(f"**API Docs:** [{API_URL}/docs]({API_URL}/docs)")
    st.markdown(f"**OpenAPI Schema:** [{API_URL}/openapi.json]({API_URL}/openapi.json)")


# ============================================
# Footer
# ============================================

st.sidebar.divider()
st.sidebar.caption("RAG Financial Assistant v1.0.0")
st.sidebar.caption("RAGHouse '25")