import os
import streamlit as st
from agentic import rag_answer
from ingestion import ingest

st.set_page_config(
    page_title="USTH RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Auto-ingestion check
if not os.path.exists("chroma_db"):
    with st.spinner("Đang khởi tạo cơ sở dữ liệu lần đầu (có thể mất vài phút)..."):
        try:
            ingest()
            st.success("Đã khởi tạo xong cơ sở dữ liệu!")
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo dữ liệu: {e}")

st.title("USTH RAG Chatbot")

# Session state

if "chat" not in st.session_state:
    st.session_state.chat = []

# Sidebar
with st.sidebar:
    st.header("Cài đặt")
    google_api_key = st.text_input("Google API Key", type="password")

# User input
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    if not google_api_key:
        st.warning("Vui lòng nhập Google API Key ở sidebar để tiếp tục.")
    else:
        with st.spinner("Vui lòng đợi trong giây lát..."):
            answer, results = rag_answer(user_input, api_key=google_api_key)

    st.session_state.chat.append({
        "question": user_input,
        "answer": answer,
        "results": results
    })

# Render history

for turn in st.session_state.chat:
    st.markdown("---")

    col_answer, col_chunks = st.columns([2, 1])

    with col_answer:
        st.subheader("Answer")
        st.markdown(f"**Câu hỏi:** {turn['question']}")
        st.markdown(turn["answer"])

    with col_chunks:
        st.subheader("Tài liệu liên quan")

        if not turn["results"]:
            st.write("Không có chunk nào được truy xuất.")
        else:
            for i, (doc, score) in enumerate(turn["results"], 1):
                with st.expander(f"Chunk {i} (Score: {score:.4f})"):
                    st.markdown(
                        f"""
- **chunk_id:** `{doc.metadata.get('chunk_id', 'N/A')}`
- **page:** `{doc.metadata.get('page', 'N/A')}`
- **source:** `{doc.metadata.get('source', 'N/A')}`

{doc.page_content}
"""
                    )

