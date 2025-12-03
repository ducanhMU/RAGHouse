"""
Streamlit UI aligned with the DESIGN.md specification. It exposes three
sections: chat interface, knowledge-base management, and system monitoring.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="RAG Control Room",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def api_call(method: str, path: str, **kwargs) -> Optional[Dict]:
    url = f"{API_BASE}{path}"
    try:
        resp = requests.request(method, url, timeout=30, **kwargs)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------


def render_health_panel():
    st.markdown("### System Health")
    health = api_call("GET", "/health")
    if not health:
        st.warning("Unable to reach backend")
        return
    for key, label in [
        ("postgres", "PostgreSQL"),
        ("milvus", "Milvus"),
        ("models", "Models"),
        ("internet", "Internet"),
    ]:
        status = health.get(key, "unknown")
        emoji = "🟢" if status in ("ok", "connected") else "🟡" if status == "limited" else "🔴"
        st.metric(label, f"{emoji} {status.upper()}")


def ensure_session_state():
    if "active_session" not in st.session_state:
        session = api_call("POST", "/sessions", json={"title": "Primary Chat"})
        st.session_state.active_session = session["session_id"] if session else None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def refresh_messages(session_id: str):
    history = api_call("GET", f"/sessions/{session_id}/history")
    if history is not None:
        st.session_state.messages = history


def render_session_list():
    st.markdown("### Sessions")
    if st.button("➕ New Chat", use_container_width=True):
        result = api_call("POST", "/sessions", json={"title": "New Chat"})
        if result:
            st.session_state.active_session = result["session_id"]
            refresh_messages(result["session_id"])
            st.experimental_rerun()

    sessions = api_call("GET", "/sessions") or []
    for sess in sessions:
        sess_id = str(sess["id"])
        title = sess["title"] or "Untitled"
        label = "🟢" if sess_id == st.session_state.active_session else "⚪"
        if st.button(f"{label} {title[:24]}", key=f"session_{sess_id}"):
            st.session_state.active_session = sess_id
            refresh_messages(sess_id)
            st.experimental_rerun()


def render_file_panel():
    st.markdown("### Knowledge Base")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        with st.spinner("Uploading..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            if api_call("POST", "/files/upload", files=files):
                st.success(f"{uploaded.name} queued for ingestion")
                time.sleep(1)
                st.experimental_rerun()

    stats = api_call("GET", "/files/status") or {}
    st.caption("Ingestion status")
    st.progress(
        stats.get("completed", 0) / max(stats.get("total", 1), 1),
        text=f"{stats.get('completed', 0)} / {stats.get('total', 0)} completed",
    )
    files = (api_call("GET", "/files") or [])[:5]
    for file_info in files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{file_info['filename']}** — {file_info.get('status')}")
        with col2:
            if st.button("🗑", key=f"delete_{file_info['id']}"):
                api_call("DELETE", f"/files/{file_info['id']}")
                st.experimental_rerun()


with st.sidebar:
    ensure_session_state()
    render_health_panel()
    st.markdown("---")
    render_session_list()
    st.markdown("---")
    render_file_panel()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


st.title("📘 AI Financial Assistant")

tabs = st.tabs(["Chat", "Knowledge Base", "Monitoring"])


with tabs[0]:
    if not st.session_state.active_session:
        st.info("No active session. Create one from the sidebar.")
    else:
        refresh_messages(st.session_state.active_session)
        for message in st.session_state.messages:
            with st.chat_message(message["role"].lower()):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about your documents...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            payload = {
                "session_id": st.session_state.active_session,
                "message": prompt,
                "use_rag": True,
            }
            response = api_call(
                "POST",
                f"/sessions/{st.session_state.active_session}/message",
                json=payload,
            )
            if response:
                refresh_messages(st.session_state.active_session)
                with st.chat_message("assistant"):
                    st.markdown(response["reply"])
                    cols = st.columns(3)
                    cols[0].caption(f"Model: {response['model_used']}")
                    cols[1].caption(f"Latency: {response['latency_ms']} ms")
                    cols[2].caption(
                        f"Citations: {len(response.get('citations', []))}"
                    )
                    if response.get("citations"):
                        with st.expander("Citations"):
                            for idx, cite in enumerate(response["citations"], start=1):
                                st.markdown(
                                    f"**Source {idx}** — file `{cite['file_id']}` page {cite['page_number']}"
                                )
                                st.caption(cite["excerpt"])


with tabs[1]:
    st.subheader("Knowledge Base Snapshot")
    files = api_call("GET", "/files") or []
    if not files:
        st.info("No files ingested yet.")
    else:
        for file_info in files:
            with st.expander(f"{file_info['filename']} — {file_info.get('status')}"):
                st.write(file_info)


with tabs[2]:
    st.subheader("Monitoring")
    stats = api_call("GET", "/stats/system") or {}
    milvus = api_call("GET", "/stats/milvus") or {}
    cols = st.columns(3)
    cols[0].metric("Files", stats.get("files", 0))
    cols[1].metric("Chunks", stats.get("chunks", 0))
    cols[2].metric("Chat Events", stats.get("messages", 0))
    st.markdown("#### Milvus Collection")
    st.json(milvus)

