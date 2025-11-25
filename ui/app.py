import streamlit as st
import requests
import os
import time

# Page Configuration
st.set_page_config(page_title="Financial Assistant", layout="wide")

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
SESSIONS_ENDPOINT = f"{API_BASE_URL}/sessions"

# --- STATE MANAGEMENT ---
# current_session_id: 
#   - None: User is starting a completely new chat.
#   - String (UUID): User is viewing an existing chat history.
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: HISTORY & UPLOAD ---
with st.sidebar:
    st.title("Knowledge & History")
    
    # --- PART 1: FILE UPLOAD ---
    with st.expander("Upload Documents", expanded=False):
        uploaded_files = st.file_uploader("Select PDF/TXT files", accept_multiple_files=True)
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

    # --- PART 2: CHAT HISTORY LIST ---
    if st.button("New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.caption("Chat History:")
    
    # Fetch Sessions from API
    try:
        res = requests.get(SESSIONS_ENDPOINT)
        if res.status_code == 200:
            sessions = res.json()
            for sess in sessions:
                # Render a button for each session
                if st.button(f"{sess['title']}", key=sess['id'], use_container_width=True):
                    st.session_state.current_session_id = sess['id']
                    
                    # Fetch messages for this session immediately
                    msg_res = requests.get(f"{API_BASE_URL}/sessions/{sess['id']}/messages")
                    if msg_res.status_code == 200:
                        st.session_state.messages = msg_res.json()
                    st.rerun()
    except Exception:
        st.warning("Cannot connect to Backend.")

# --- MAIN INTERFACE ---
st.title("Financial AI Assistant")

if st.session_state.current_session_id is None:
    st.caption("You are starting a new conversation.")
else:
    st.caption(f"Session ID: {st.session_state.current_session_id}")

# 1. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about finance..."):
    # Render User Message Immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Call API
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Agent is thinking...")
        
        try:
            payload = {
                "session_id": st.session_state.current_session_id, 
                "input": prompt
            }
            res = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            
            if res.status_code == 200:
                data = res.json()
                answer = data['answer']
                
                # Update Session ID if this was a new chat
                if st.session_state.current_session_id is None:
                    st.session_state.current_session_id = data['session_id']
                    
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Rerun to update the Sidebar title with the new session
                    time.sleep(0.1)
                    st.rerun()
                else:
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                placeholder.error(f"API Error: {res.text}")
                
        except Exception as e:
            placeholder.error(f"Connection Error: {e}")