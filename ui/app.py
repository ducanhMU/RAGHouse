import streamlit as st
import requests
import os
import time
import json

# Page Configuration
st.set_page_config(page_title="Financial Assistant", layout="wide", page_icon="📈")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
SESSIONS_ENDPOINT = f"{API_BASE_URL}/sessions"

# --- STATE MANAGEMENT ---
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗂️ Knowledge & History")
    
    # 1. File Upload
    with st.expander("Upload Documents", expanded=False):
        uploaded_files = st.file_uploader("Select PDF/TXT", accept_multiple_files=True)
        if st.button("Upload & Ingest", type="primary"):
            if uploaded_files:
                with st.spinner("Uploading..."):
                    files = [('files', (f.name, f, f.type)) for f in uploaded_files]
                    try:
                        res = requests.post(UPLOAD_ENDPOINT, files=files, timeout=300)
                        if res.status_code == 200:
                            st.success("Upload successful!")
                        else:
                            st.error(res.text)
                    except Exception as e:
                        st.error(str(e))
    
    st.divider()

    # 2. Session List
    if st.button("New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.caption("Recent Chats:")
    try:
        res = requests.get(SESSIONS_ENDPOINT)
        if res.status_code == 200:
            sessions = res.json()
            for sess in sessions:
                if st.button(f"{sess['title']}", key=sess['id'], use_container_width=True):
                    st.session_state.current_session_id = sess['id']
                    msg_res = requests.get(f"{API_BASE_URL}/sessions/{sess['id']}/messages")
                    if msg_res.status_code == 200:
                        st.session_state.messages = msg_res.json()
                    st.rerun()
    except Exception:
        st.warning("Backend offline.")

# --- MAIN INTERFACE ---
st.title("Financial AI Assistant")

if st.session_state.current_session_id is None:
    st.caption("Start a new conversation.")
else:
    st.caption(f"Session: {st.session_state.current_session_id}")

# Render History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Handle Input
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Stream Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources_data = []
        new_session_id = None
        
        try:
            payload = {"session_id": st.session_state.current_session_id, "input": prompt}
            
            # POST request with stream=True
            with requests.post(CHAT_ENDPOINT, json=payload, stream=True, timeout=120) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            
                            # Handle Metadata (Session ID)
                            if decoded_line.startswith("meta:"):
                                new_session_id = decoded_line[5:]

                            # Handle Text Chunk
                            elif decoded_line.startswith("text:"):
                                chunk_text = decoded_line[5:] 
                                full_response += chunk_text
                                placeholder.markdown(full_response + "▌")
                            
                            # Handle Sources Chunk (JSON)
                            elif decoded_line.startswith("sources:"):
                                json_str = decoded_line[8:]
                                try:
                                    sources_data = json.loads(json_str)
                                except json.JSONDecodeError:
                                    pass

                    # Final Render
                    placeholder.markdown(full_response)
                    
                    # Render Citations
                    if sources_data:
                        with st.expander("📚 Sources / Citations", expanded=True):
                            for idx, src in enumerate(sources_data):
                                st.markdown(f"**{idx+1}. {src['file']}** (Page {src['page']})")

                    # Update State
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Handle New Session Logic
                    if st.session_state.current_session_id is None and new_session_id:
                        st.session_state.current_session_id = new_session_id
                        time.sleep(0.5)
                        st.rerun()

                else:
                    st.error(f"API Error: {r.text}")
                    
        except Exception as e:
            st.error(f"Connection Error: {e}")