import streamlit as st
import requests
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Financial Assistant", layout="wide")

# --- API CONFIGURATION ---
# Use 'http://api:8000' inside Docker network, or 'http://localhost:8000' for local dev
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
SESSIONS_ENDPOINT = f"{API_BASE_URL}/sessions"

# --- STATE MANAGEMENT ---
# current_session_id: 
#   - None: User is in 'New Chat' mode.
#   - String: User is in an active session.
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("Knowledge & History")
    
    # --- 1. FILE UPLOAD ---
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
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
    
    st.divider()

    # --- 2. SESSION MANAGEMENT ---
    if st.button("New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
    
    st.caption("Recent Conversations:")
    
    # Fetch and display sessions
    try:
        res = requests.get(SESSIONS_ENDPOINT)
        if res.status_code == 200:
            sessions = res.json()
            for sess in sessions:
                # Button logic to switch sessions
                if st.button(f"{sess['title']}", key=sess['id'], use_container_width=True):
                    st.session_state.current_session_id = sess['id']
                    
                    # Fetch history for the selected session
                    msg_res = requests.get(f"{API_BASE_URL}/sessions/{sess['id']}/messages")
                    if msg_res.status_code == 200:
                        st.session_state.messages = msg_res.json()
                    st.rerun()
    except Exception:
        st.warning("Cannot connect to Backend API.")

# --- MAIN CHAT INTERFACE ---
st.title("Financial AI Assistant")

# Display current status
if st.session_state.current_session_id is None:
    st.caption("Start a new conversation...")
else:
    st.caption(f"Session ID: {st.session_state.current_session_id}")

# 1. Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display Assistant Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        
        try:
            # Prepare Payload
            payload = {
                "session_id": st.session_state.current_session_id, 
                "input": prompt
            }
            
            # Call API
            res = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            
            if res.status_code == 200:
                data = res.json()
                answer = data['answer']
                new_session_id = data['session_id']
                
                # Check if we just started a new session
                is_new_session = st.session_state.current_session_id is None
                
                # Update State
                st.session_state.current_session_id = new_session_id
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Render Answer
                placeholder.markdown(answer)
                
                # If it was a new session, rerun to update the Sidebar title immediately
                if is_new_session:
                    time.sleep(0.1)
                    st.rerun()
            else:
                placeholder.error(f"API Error: {res.text}")
                
        except Exception as e:
            placeholder.error(f"Connection Error: {e}")