import streamlit as st

from retriever import get_retriever
from agentic import build_rag_chain

st.set_page_config(page_title="USTH RAG Chatbot", page_icon="🤖")
st.title("USTH RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

rag_chain = build_rag_chain()

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Chờ tý, sự thật chỉ có một... "):
            answer = rag_chain.invoke({"question": user_input})
            st.write(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
