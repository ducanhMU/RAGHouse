import streamlit as st
import requests
import os
import time

st.set_page_config(page_title="RAG", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
INGEST_ENDPOINT = f"{API_BASE_URL}/ingest"

# --- SIDEBAR ---
with st.sidebar:
    st.title("RAG")
    st.markdown("UI + API")
    st.markdown("---")
    
    st.header("Manage Data")
    st.markdown("Click the button below to load data from `/api/data`.")
    
    if st.button("Ingest"):
        with st.spinner("Requesting ingest data..."):
            try:
                response = requests.post(INGEST_ENDPOINT)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to {API_BASE_URL}. Is the service 'api' running?")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.header("Config")
    st.markdown(f"**UI (Streamlit)** is running.")
    st.markdown(f"**API (FastAPI)** at `{API_BASE_URL}`")

# --- GIAO DIỆN CHAT CHÍNH ---
st.title("Chatbot RAG")
st.caption("This interface will call to a seperated Backend API (FastAPI).")

# Declare chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user's input
if prompt := st.chat_input("Ask anything about your document..."):
    
    # add user's input into history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # render user's input
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # render "answering..."
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Answering...")
        
        start_time = time.time()
        
        # Prepare data to send to API
        chat_history_api = []
        for msg in st.session_state.messages[:-1]: # get all history messages except the last one (which is the user's input)
            chat_history_api.append({
                "type": "human" if msg["role"] == "user" else "ai",
                "content": msg["content"]
            })
        
        request_data = {
            "input": prompt,
            "chat_history": chat_history_api
        }
        
        # call API
        try:
            response = requests.post(CHAT_ENDPOINT, json=request_data, timeout=120)
            
            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = f"Error: {response.text}"
                
        except requests.exceptions.ConnectionError:
            answer = f"Cannot connect to {CHAT_ENDPOINT}."
        except requests.exceptions.ReadTimeout:
            answer = "Timeout."
        except Exception as e:
            answer = f"Error: {e}"
            
        end_time = time.time()
        
        # render the answer
        message_placeholder.markdown(answer)
        st.caption(f"Process (include call API) in {end_time - start_time:.2f}s.")

    # add answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})