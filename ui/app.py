"""
RAG V2 - Streamlit UI with System Health Check
"""

import streamlit as st
import requests
import time
import os
from datetime import datetime

# =========================================
# CONFIGURATION
# =========================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG V2 Ultimate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# HELPER FUNCTIONS
# =========================================

def check_system_health():
    """Check if backend is ready"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def api_request(method, endpoint, **kwargs):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# =========================================
# LOADING SCREEN
# =========================================

def show_loading_screen():
    """Show loading screen while system initializes"""
    placeholder = st.empty()
    
    with placeholder.container():
        st.title("🚀 RAG V2 Ultimate")
        st.markdown("### System Initialization")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        max_attempts = 30
        for attempt in range(max_attempts):
            is_ready, health_data = check_system_health()
            
            if is_ready:
                progress_bar.progress(100)
                status_text.success("✅ All systems ready!")
                time.sleep(1)
                placeholder.empty()
                return True
            
            progress = int((attempt + 1) / max_attempts * 100)
            progress_bar.progress(progress)
            status_text.info(f"🔄 Waiting for backend... ({attempt + 1}/{max_attempts})")
            time.sleep(2)
        
        status_text.error("❌ Failed to connect to backend. Please check if services are running.")
        st.stop()
        return False

# =========================================
# INITIALIZE SESSION STATE
# =========================================

if "initialized" not in st.session_state:
    if show_loading_screen():
        st.session_state.initialized = True
        st.session_state.current_session = None
        st.session_state.messages = []

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:
    st.title("🤖 RAG V2 Ultimate")
    st.markdown("---")
    
    # System Status
    with st.expander("📊 System Status", expanded=False):
        if st.button("Refresh Status"):
            st.rerun()
        
        is_ready, health = check_system_health()
        if health:
            st.metric("PostgreSQL", "🟢 Online" if health['postgres'] == 'ok' else "🔴 Offline")
            st.metric("Milvus", "🟢 Online" if health['milvus'] == 'ok' else "🔴 Offline")
            st.metric("AI Models", "🟢 Ready" if health['models'] == 'ok' else "🔴 Not Ready")
            st.metric("Internet", "🟢 Connected" if health['internet'] == 'ok' else "🟡 Limited")
    
    # Chat Sessions
    st.markdown("### 💬 Chat Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        result = api_request("POST", "/sessions", json={"title": "New Chat"})
        if result:
            st.session_state.current_session = result["session_id"]
            st.session_state.messages = []
            st.rerun()
    
    # List sessions
    sessions = api_request("GET", "/sessions")
    if sessions:
        for session in sessions[:10]:
            session_id = str(session['id'])
            title = session['title']
            is_active = session_id == st.session_state.current_session
            
            if st.button(
                f"{'🟢' if is_active else '⚪'} {title[:25]}",
                key=session_id,
                use_container_width=True
            ):
                st.session_state.current_session = session_id
                # Load messages
                messages = api_request("GET", f"/sessions/{session_id}/events")
                st.session_state.messages = messages or []
                st.rerun()
    
    st.markdown("---")
    
    # File Manager
    tab1, tab2 = st.tabs(["📁 Files", "⚙️ Services"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file:
            with st.spinner("Processing..."):
                files = {'file': uploaded_file}
                result = api_request("POST", "/files/upload", files=files)
                if result:
                    st.success(f"✅ {uploaded_file.name} queued for processing")
                    time.sleep(2)
                    st.rerun()
        
        # File list
        files = api_request("GET", "/files")
        if files:
            for file in files[:5]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    status_emoji = {
                        'completed': '✅',
                        'processing': '⏳',
                        'pending': '⏸️',
                        'failed': '❌'
                    }.get(file.get('status', 'unknown'), '❓')
                    st.text(f"{status_emoji} {file['filename'][:20]}")
                with col2:
                    if st.button("🗑️", key=f"del_{file['id']}"):
                        api_request("DELETE", f"/files/{file['id']}")
                        st.rerun()
    
    with tab2:
        services = api_request("GET", "/system/services")
        if services:
            for svc in services:
                st.text(f"🟢 {svc['name']}")
                st.caption(svc['url'])

# =========================================
# MAIN CHAT INTERFACE
# =========================================

st.title("💬 RAG AI Assistant")

# Create session if none exists
if not st.session_state.current_session:
    result = api_request("POST", "/sessions", json={"title": "Welcome Chat"})
    if result:
        st.session_state.current_session = result["session_id"]

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"].lower()):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "USER", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = api_request(
                "POST",
                f"/sessions/{st.session_state.current_session}/message",
                json={"content": prompt, "use_rag": True}
            )
            
            if response:
                st.markdown(response["reply"])
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    source_emoji = {
                        'knowledge_base': '📘',
                        'sql': '📊',
                        'llm_only': '🧠'
                    }.get(response['source_type'], '❓')
                    st.caption(f"{source_emoji} {response['source_type'].replace('_', ' ').title()}")
                with col2:
                    st.caption(f"⚡ {response['latency']}s")
                with col3:
                    st.caption(f"🤖 {response['model_used']}")
                
                # Show sources
                if response['sources']:
                    with st.expander(f"📚 View Sources ({len(response['sources'])})"):
                        for i, source in enumerate(response['sources'], 1):
                            st.markdown(f"**[Source {i}]** (Page {source['page']}, Score: {source['score']:.2f})")
                            st.text(source['text'])
                            st.markdown("---")
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "ASSISTANT",
                    "content": response["reply"]
                })

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.caption("🚀 RAG V2 Ultimate | Hybrid Search + Infinite Context + GPU-Accelerated")