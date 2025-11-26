# file: ui/app.py

import streamlit as st
import requests
import os
import json
import time
from typing import Optional, List, Dict
from datetime import datetime

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="RAG Financial Assistant",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# ===== API CONFIGURATION =====
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
ENDPOINTS = {
    "health": f"{API_BASE_URL}/health",
    "chat": f"{API_BASE_URL}/chat",
    "upload": f"{API_BASE_URL}/upload",
    "sessions": f"{API_BASE_URL}/sessions",
    "files": f"{API_BASE_URL}/files",
}

# ===== SESSION STATE INITIALIZATION =====
def init_session_state():
    """Initialize all session state variables."""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_files_status" not in st.session_state:
        st.session_state.uploaded_files_status = []
    
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = "unknown"

init_session_state()

# ===== HELPER FUNCTIONS =====

def check_backend_health() -> Dict:
    """Check if backend is healthy and return status."""
    try:
        response = requests.get(ENDPOINTS["health"], timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.backend_status = data.get("status", "unknown")
            return data
        else:
            st.session_state.backend_status = "unhealthy"
            return {"status": "unhealthy"}
    except Exception as e:
        st.session_state.backend_status = "offline"
        return {"status": "offline", "error": str(e)}

def load_session_history(session_id: str) -> List[Dict]:
    """Load full message history for a session."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/history",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("messages", [])
        else:
            st.error(f"Failed to load session: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error loading session: {e}")
        return []

def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %I:%M %p")
    except:
        return timestamp_str

def get_file_status_badge(status: str) -> str:
    """Return colored badge for file status."""
    status_colors = {
        "COMPLETED": "🟢",
        "PROCESSING": "🟡",
        "PENDING": "⚪",
        "FAILED": "🔴"
    }
    return status_colors.get(status, "⚫")

# ===== SIDEBAR =====
with st.sidebar:
    st.title("💼 RAG Assistant")
    
    # Health Status Indicator
    health_data = check_backend_health()
    status = health_data.get("status", "unknown")
    
    if status == "healthy":
        st.success("🟢 System Online", icon="✅")
    elif status == "degraded":
        st.warning("🟡 System Degraded", icon="⚠️")
    else:
        st.error("🔴 System Offline", icon="❌")
    
    st.divider()
    
    # ===== FILE UPLOAD SECTION =====
    st.subheader("📁 Upload Documents")
    
    with st.expander("Upload Files", expanded=False):
        uploaded_files = st.file_uploader(
            "Select PDF or TXT files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md"],
            help="Upload documents to add to the knowledge base"
        )
        
        if st.button("📤 Upload & Process", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Uploading files..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                            response = requests.post(
                                ENDPOINTS["upload"],
                                files=files,
                                timeout=300
                            )
                            
                            if response.status_code in [200, 201]:
                                data = response.json()
                                success_count += 1
                                st.session_state.uploaded_files_status.append({
                                    "name": uploaded_file.name,
                                    "status": data.get("processing_status", "PENDING"),
                                    "file_id": data.get("file_id"),
                                    "message": data.get("message", "")
                                })
                            else:
                                st.error(f"❌ {uploaded_file.name}: {response.text}")
                        
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: {str(e)}")
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully uploaded {success_count} file(s)!")
                        time.sleep(1)
                        st.rerun()
            else:
                st.warning("Please select files to upload")
    
    # Recent Upload Status
    if st.session_state.uploaded_files_status:
        with st.expander("Recent Uploads", expanded=True):
            for file_info in st.session_state.uploaded_files_status[-5:]:
                status_badge = get_file_status_badge(file_info["status"])
                st.caption(f"{status_badge} {file_info['name']}")
    
    # View All Files
    if st.button("📋 View All Files", use_container_width=True):
        try:
            response = requests.get(f"{ENDPOINTS['files']}?limit=20", timeout=10)
            if response.status_code == 200:
                files = response.json()
                with st.expander("All Uploaded Files", expanded=True):
                    if files:
                        for file in files:
                            status_badge = get_file_status_badge(file["status"])
                            st.caption(
                                f"{status_badge} **{file['filename']}**\n"
                                f"Status: {file['status']} | "
                                f"Uploaded: {format_timestamp(file['created_at'])}"
                            )
                    else:
                        st.info("No files uploaded yet")
        except Exception as e:
            st.error(f"Error loading files: {e}")
    
    st.divider()
    
    # ===== SESSION MANAGEMENT =====
    st.subheader("💬 Chat Sessions")
    
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
    
    # Load and display recent sessions
    try:
        response = requests.get(f"{ENDPOINTS['sessions']}?limit=15", timeout=10)
        if response.status_code == 200:
            sessions = response.json()
            
            if sessions:
                st.caption("Recent Conversations:")
                for sess in sessions:
                    # Highlight current session
                    is_current = sess['id'] == st.session_state.current_session_id
                    button_type = "primary" if is_current else "secondary"
                    
                    # Create button with session info
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            f"{'📍' if is_current else '💬'} {sess['title'][:30]}...",
                            key=f"session_{sess['id']}",
                            type=button_type,
                            use_container_width=True
                        ):
                            # Load session history
                            with st.spinner("Loading conversation..."):
                                messages = load_session_history(sess['id'])
                                st.session_state.current_session_id = sess['id']
                                st.session_state.messages = messages
                                st.rerun()
                    
                    with col2:
                        # Delete button
                        if st.button("🗑️", key=f"del_{sess['id']}", help="Delete session"):
                            try:
                                del_response = requests.delete(
                                    f"{API_BASE_URL}/sessions/{sess['id']}",
                                    timeout=10
                                )
                                if del_response.status_code == 204:
                                    if sess['id'] == st.session_state.current_session_id:
                                        st.session_state.current_session_id = None
                                        st.session_state.messages = []
                                    st.success("Session deleted")
                                    time.sleep(0.5)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                    
                    # Show message count
                    msg_count = sess.get('message_count', 0)
                    st.caption(f"   {msg_count} messages • {format_timestamp(sess['updated_at'])}")
            else:
                st.info("No conversations yet")
    
    except Exception as e:
        st.warning("⚠️ Cannot load sessions")
        st.caption(f"Error: {str(e)}")
    
    st.divider()
    
    # System Info
    with st.expander("ℹ️ System Info"):
        st.caption(f"**API Endpoint:** {API_BASE_URL}")
        st.caption(f"**Status:** {status}")
        if health_data.get("version"):
            st.caption(f"**Version:** {health_data['version']}")
        
        if health_data.get("services"):
            st.caption("**Services:**")
            for service, svc_status in health_data["services"].items():
                icon = "🟢" if svc_status == "healthy" else "🔴"
                st.caption(f"  {icon} {service}: {svc_status}")

# ===== MAIN CHAT INTERFACE =====
st.title("💼 Financial AI Assistant")

# Session Info
if st.session_state.current_session_id:
    st.caption(f"📍 Session: `{st.session_state.current_session_id}`")
else:
    st.caption("🆕 New conversation - your first message will create a session")

# Display Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        
        # Show metadata for assistant messages
        if msg['role'] == 'assistant' and msg.get('timestamp'):
            st.caption(
                f"🤖 {msg.get('model', 'unknown')} • "
                f"{format_timestamp(msg['timestamp'])}"
            )

# Chat Input
if prompt := st.chat_input("Ask about financial data, reports, or analysis..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        new_session_id = None
        model_used = None
        
        try:
            # Prepare request payload
            payload = {
                "session_id": st.session_state.current_session_id,
                "message": prompt
            }
            
            # Stream response from API
            with requests.post(
                ENDPOINTS["chat"],
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                
                if response.status_code == 200:
                    # Process SSE stream
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            
                            # Remove 'data: ' prefix from SSE format
                            if decoded_line.startswith('data: '):
                                json_str = decoded_line[6:]
                                
                                try:
                                    data = json.loads(json_str)
                                    event_type = data.get('type')
                                    
                                    if event_type == 'session':
                                        # Capture new session ID
                                        new_session_id = data.get('session_id')
                                    
                                    elif event_type == 'text':
                                        # Append text chunk
                                        chunk = data.get('content', '')
                                        full_response += chunk
                                        # Show typing indicator
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    elif event_type == 'done':
                                        # Stream complete
                                        break
                                    
                                    elif event_type == 'error':
                                        # Handle error
                                        error_msg = data.get('message', 'Unknown error')
                                        st.error(f"Error: {error_msg}")
                                        break
                                
                                except json.JSONDecodeError:
                                    # Skip malformed JSON
                                    continue
                    
                    # Final render without typing indicator
                    message_placeholder.markdown(full_response)
                    
                    # Update session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().isoformat(),
                        "model": model_used
                    })
                    
                    # Update session ID if new
                    if new_session_id and not st.session_state.current_session_id:
                        st.session_state.current_session_id = new_session_id
                        time.sleep(0.5)
                        st.rerun()
                
                else:
                    # Handle HTTP errors
                    error_text = response.text
                    st.error(f"❌ API Error ({response.status_code}): {error_text}")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend. Please check if the API is running.")
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.exception(e)

# ===== FOOTER =====
st.divider()
st.caption(
    "💡 **Tips:** Upload financial documents (PDFs, TXT) to build your knowledge base. "
    "The assistant will use them to answer your questions with citations."
)