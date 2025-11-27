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
# Nếu chạy Docker -> Docker, dùng tên service "http://api:8000"
# Nếu chạy Local -> Local, dùng "http://localhost:8000"
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

ENDPOINTS = {
    "health": f"{API_BASE_URL}/health",
    "chat": f"{API_BASE_URL}/chat",
    "upload": f"{API_BASE_URL}/upload",
    "sessions": f"{API_BASE_URL}/sessions",
    "files": f"{API_BASE_URL}/files",
}

# ===== SESSION STATE =====
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_status" not in st.session_state:
    st.session_state.uploaded_files_status = []
if "backend_status" not in st.session_state:
    st.session_state.backend_status = "unknown"

# ===== HELPERS =====
def check_backend_health() -> Dict:
    try:
        response = requests.get(ENDPOINTS["health"], timeout=3)
        if response.status_code == 200:
            data = response.json()
            st.session_state.backend_status = data.get("status", "unknown")
            return data
        st.session_state.backend_status = "unhealthy"
        return {"status": "unhealthy"}
    except:
        st.session_state.backend_status = "offline"
        return {"status": "offline"}

def load_session_history(session_id: str):
    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Map history format to UI format
            return [{
                "role": m["role"],
                "content": m["content"],
                "timestamp": m.get("timestamp"),
                # Backend currently doesn't store metadata in DB history endpoint fully
                # Ideally backend should return it, but for now we render text
                "metadata": {} 
            } for m in data.get("messages", [])]
        return []
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []

def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%H:%M %d/%m")
    except:
        return ""

def render_metadata(metadata: Dict):
    """Display Intent, SQL, and Viz links."""
    if not metadata:
        return

    # 1. Intent Badge
    intent = metadata.get('intent')
    if intent:
        if intent == 'sql':
            st.caption("🤖 Mode: **SQL Analyst**")
        elif intent == 'visualization':
            st.caption("🤖 Mode: **Visualization**")
        else:
            st.caption("🤖 Mode: **Knowledge Base (RAG)**")

    # 2. SQL Code Block
    sql_query = metadata.get('sql')
    if sql_query:
        with st.expander("🔍 View Generated SQL", expanded=False):
            st.code(sql_query, language="sql")

    # 3. Visualization Link
    viz_link = metadata.get('viz_link')
    if viz_link:
        st.success("📊 Dashboard Available")
        st.link_button("Open Dashboard ↗️", viz_link)

# ===== SIDEBAR =====
with st.sidebar:
    st.header("💼 RAG Assistant")
    
    # Status
    health = check_backend_health()
    status_icon = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴", "offline": "⚫"}
    st.caption(f"{status_icon.get(st.session_state.backend_status, '⚫')} System: {st.session_state.backend_status.upper()}")
    
    st.divider()
    
    # Upload
    with st.expander("📤 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader("PDF/TXT/MD", accept_multiple_files=True)
        if st.button("Process Files", type="primary", use_container_width=True):
            if uploaded_files:
                for uf in uploaded_files:
                    try:
                        files = {"file": (uf.name, uf, uf.type)}
                        res = requests.post(ENDPOINTS["upload"], files=files, timeout=60)
                        if res.status_code in [200, 201]:
                            st.success(f"✅ {uf.name} uploaded")
                        elif res.status_code == 409:
                            st.warning(f"⚠️ {uf.name} already exists")
                        else:
                            st.error(f"❌ {uf.name} failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
                time.sleep(1)
                st.rerun()

    # Session List
    st.subheader("💬 History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
        
    try:
        res = requests.get(f"{ENDPOINTS['sessions']}?limit=10", timeout=5)
        if res.status_code == 200:
            for s in res.json():
                active = s['id'] == st.session_state.current_session_id
                label = f"{'📍' if active else ''} {s['title'][:25]}..."
                
                col1, col2 = st.columns([0.8, 0.2])
                if col1.button(label, key=s['id'], use_container_width=True):
                    st.session_state.current_session_id = s['id']
                    st.session_state.messages = load_session_history(s['id'])
                    st.rerun()
                if col2.button("✕", key=f"del_{s['id']}"):
                    requests.delete(f"{API_BASE_URL}/sessions/{s['id']}")
                    if active:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()
    except:
        st.warning("Cannot load sessions")

# ===== MAIN CHAT =====
st.title("Financial Intelligence AI")

if not st.session_state.current_session_id:
    st.info("👋 Welcome! Upload documents or start asking questions about financial data.")

# Render Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render metadata if exists (for older messages in current session state)
        if msg.get("metadata"):
            render_metadata(msg["metadata"])

# Input
if prompt := st.chat_input("Ask about revenue, profit, or upload files..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        full_text = ""
        current_metadata = {}
        
        try:
            payload = {"session_id": st.session_state.current_session_id, "message": prompt}
            
            with requests.post(ENDPOINTS["chat"], json=payload, stream=True, timeout=60) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded = line.decode('utf-8')
                            if decoded.startswith("data: "):
                                try:
                                    data = json.loads(decoded[6:])
                                    type_ = data.get("type")
                                    
                                    if type_ == "session":
                                        st.session_state.current_session_id = data.get("session_id")
                                    
                                    elif type_ == "text":
                                        chunk = data.get("content", "")
                                        full_text += chunk
                                        text_placeholder.markdown(full_text + "▌")
                                    
                                    elif type_ == "metadata":
                                        # [QUAN TRỌNG] Bắt lấy metadata từ backend
                                        current_metadata = data.get("data", {})
                                    
                                    elif type_ == "error":
                                        st.error(data.get("message"))
                                        
                                except Exception:
                                    continue
                    
                    # Final Render
                    text_placeholder.markdown(full_text)
                    if current_metadata:
                        render_metadata(current_metadata)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "metadata": current_metadata
                    })
                    
                elif r.status_code == 503:
                    st.warning("🚀 Engine is warming up... Please try again in 5 seconds.")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")
                    
        except Exception as e:
            st.error(f"Connection Error: {e}")