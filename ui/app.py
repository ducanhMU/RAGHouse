import streamlit as st
import requests
import os
import time

# Page Configuration
st.set_page_config(page_title="Financial RAG Agent", layout="wide", page_icon="📈")

# --- API CONFIGURATION ---
# Use 'http://api:8000' if running via Docker Compose (service name 'api')
# Use 'http://localhost:8000' if running locally without Docker networking
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"

# --- SIDEBAR: DATA MANAGEMENT ---
with st.sidebar:
    st.title("Data Management")
    st.markdown("Upload documents (PDF, TXT) to enhance the Knowledge Base.")
    
    # 1. File Upload Widget
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    # 2. Upload & Ingest Button
    if st.button("Upload & Ingest", type="primary"):
        if not uploaded_files:
            st.warning("Please select files first.")
        else:
            with st.spinner(f"Uploading and processing {len(uploaded_files)} files..."):
                try:
                    # Prepare file list for multipart/form-data
                    files_payload = [
                        ('files', (file.name, file, file.type)) for file in uploaded_files
                    ]
                    
                    # Call API
                    response = requests.post(UPLOAD_ENDPOINT, files=files_payload, timeout=300)
                    
                    if response.status_code == 200:
                        st.success(f"Success! {response.json().get('message', '')}")
                        # Clear history to avoid stale context since data has changed
                        st.session_state.messages = [] 
                    else:
                        st.error(f"Server Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to API at {API_BASE_URL}")
                except Exception as e:
                    st.error(f"Unknown Error: {e}")

    st.markdown("---")
    st.caption(f"Connected to Backend: `{API_BASE_URL}`")

# --- MAIN CHAT INTERFACE ---
st.title("Financial AI Assistant")
st.caption("Ask questions about your data or fetch realtime info via Internet Search.")

# 1. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render Existing History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle New User Input
if prompt := st.chat_input("Ex: What is the revenue of VTS in 2025?"):
    
    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Process Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Agent is thinking...")
        
        start_time = time.time()
        
        # Prepare Payload for Backend
        # We need to send the history so the Agent can rewrite the query if needed.
        # Format: List of dicts with keys 'role' and 'content'
        chat_history_api = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in st.session_state.messages[:-1] # Exclude the current prompt
        ]
        
        request_data = {
            "input": prompt,
            "chat_history": chat_history_api
        }
        
        answer = "Error generating response."
        
        try:
            # Call Chat API
            response = requests.post(CHAT_ENDPOINT, json=request_data, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer provided by the system.")
            else:
                answer = f"API Error ({response.status_code}): {response.text}"
                
        except requests.exceptions.ConnectionError:
            answer = f"Cannot connect to Backend at {CHAT_ENDPOINT}. Is the API container running?"
        except requests.exceptions.ReadTimeout:
            answer = "Request timed out. The model is taking too long to respond."
        except Exception as e:
            answer = f"Unexpected Error: {e}"
            
        end_time = time.time()
        
        # Render Final Answer
        message_placeholder.markdown(answer)
        st.caption(f"Processed in {end_time - start_time:.2f}s")

    # Append assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": answer})