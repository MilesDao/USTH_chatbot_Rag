import streamlit as st
from agentic import rag_answer

st.set_page_config(
    page_title="USTH RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("USTH RAG Chatbot")

# Session state

if "chat" not in st.session_state:
    st.session_state.chat = []

# User input
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    with st.spinner("Vui lòng đợi trong giây lát..."):
        answer, results = rag_answer(user_input)

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

