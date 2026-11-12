import sys
import traceback
import uuid

import streamlit as st

from rag.pipeline import ask_rag
from database.session import (
    create_session,
    get_session_messages,
    get_all_sessions,
    get_session_preview
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Aditya's Personal RAG",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "current_session_id" not in st.session_state:

    session_id = str(uuid.uuid4())

    create_session(session_id)

    st.session_state.current_session_id = session_id


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def create_new_chat():

    session_id = str(uuid.uuid4())

    create_session(session_id)

    st.session_state.current_session_id = session_id

    st.rerun()


def switch_session(session_id):

    st.session_state.current_session_id = session_id

    st.rerun()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("💬 Conversations")

    st.button(
        "＋ New Chat",
        use_container_width=True,
        on_click=create_new_chat
    )

    st.divider()

    sessions = get_all_sessions()

    current_session = st.session_state.current_session_id

    for session in sessions:

        session_id = session["session_id"]

        preview = get_session_preview(session)

        if session_id == current_session:

            label = f"🟢 {preview}"

        else:

            label = f"💬 {preview}"

        st.button(
            label,
            key=f"session_{session_id}",
            use_container_width=True,
            on_click=switch_session,
            args=(session_id,)
        )


# --------------------------------------------------
# Main Chat Area
# --------------------------------------------------

st.title("🤖 Aditya's Personal RAG Assistant")

st.caption(
    "Ask questions about Aditya, USBM, "
    "or Maa Manikeshwari University."
)


# --------------------------------------------------
# Load Current Session
# --------------------------------------------------

current_session_id = st.session_state.current_session_id

messages = get_session_messages(current_session_id)


# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in messages:

    role = message["role"]

    if role == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    elif role == "assistant":

        with st.chat_message("assistant"):
            st.write(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

question = st.chat_input(
    "Ask something..."
)


# --------------------------------------------------
# Process Question
# --------------------------------------------------

if question:

    # Display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Generate RAG response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            try:
                answer = ask_rag(
                    question,
                    current_session_id
                )
            except Exception as e:
                print("\n" + "=" * 50, file=sys.stderr)
                print("ERROR OCCURRED IN RAG PIPELINE:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("=" * 50 + "\n", file=sys.stderr)
                sys.stderr.flush()
                st.error(f"Error: {e}")
                answer = "An unexpected error occurred. Please check your network connection and try again."

        st.write(answer)