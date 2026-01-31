import os
import streamlit as st
import importlib
import agentic
import retriever
importlib.reload(retriever)
importlib.reload(agentic)
from agentic import rag_answer, evaluate_rag_answer
from ingestion import ingest
import json

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
    
    # Pre-fill API Key if available in environment
    default_api_key = os.getenv("GOOGLE_API_KEY", "")
    google_api_key = st.text_input("Google API Key", value=default_api_key, type="password")

# User input
user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    if not google_api_key:
        st.warning("Vui lòng nhập Google API Key ở sidebar để tiếp tục.")
    else:
        with st.spinner("Chờ xíu..."):
            # Explicitly pass k=8 to match agentic/retriever logic
            answer, results = rag_answer(user_input, api_key=google_api_key, k=8)
            
            st.session_state.chat.append({
                "question": user_input,
                "answer": answer,
                "results": results,
                "evaluation": None  # Placeholder for evaluation results
            })

# Render history

for i, turn in enumerate(st.session_state.chat):
    st.markdown("---")

    col_answer, col_chunks = st.columns([2, 1])

    with col_answer:
        st.subheader("Answer")
        st.markdown(f"**Câu hỏi:** {turn['question']}")
        st.markdown(turn["answer"])
        
        # --- EVALUATION UI ---
        with st.expander("📝 Evaluate this answer"):
            with st.form(key=f"eval_form_{i}"):
                st.write("Nhập câu trả lời mẫu để chấm điểm AI:")
                expected_ans = st.text_area("Expected Answer (Golden Answer)", height=100)
                expected_ctx = st.text_area("Expected Context (Optional, separate lines)", height=100)
                
                submit_eval = st.form_submit_button("Run Evaluation")
                
                if submit_eval:
                    if not google_api_key:
                        st.error("Cần Google API Key để chấm điểm!")
                    elif not expected_ans:
                        st.error("Cần nhập Expected Answer!")
                    else:
                        with st.spinner("Đang chấm điểm (DeepEval)..."):
                            try:
                                # Prepare context list
                                exp_ctx_list = [line.strip() for line in expected_ctx.split('\n') if line.strip()]
                                
                                eval_metrics = evaluate_rag_answer(
                                    query=turn['question'],
                                    actual_answer=turn['answer'],
                                    retrieved_docs=turn['results'],
                                    expected_output=expected_ans,
                                    expected_context=exp_ctx_list,
                                    api_key=google_api_key
                                )
                                
                                # Update session state with evaluation result
                                st.session_state.chat[i]["evaluation"] = eval_metrics
                                st.rerun() # Rerun to show results immediately
                            except Exception as e:
                                st.error(f"Lỗi khi chấm điểm: {e}")
        
        # Display Evaluation Results if they exist
        if turn.get("evaluation"):
            eval_res = turn["evaluation"]
            st.success("✅ Evaluation Results")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                prec_score = eval_res['precision_score']
                st.metric("Contextual Precision", f"{prec_score:.2f}")
                if prec_score >= 0.5:
                    st.success("✅ RELEVANT")
                else:
                    st.error("❌ FAIL")
                st.info(f"**Reason:** {eval_res['precision_reason']}")
            
            with m_col2:
                corr_score = eval_res['correctness_score']
                st.metric("Answer Correctness", f"{corr_score:.2f}")
                if corr_score >= 0.5:
                    st.success("✅ RELEVANT")
                else:
                    st.error("❌ FAIL")
                st.info(f"**Reason:** {eval_res['correctness_reason']}")

    with col_chunks:
        st.subheader("Tài liệu liên quan")

        if not turn["results"]:
            st.write("Không có chunk nào được truy xuất.")
        else:
            for j, (doc, score) in enumerate(turn["results"], 1):
                with st.expander(f"Chunk {j} (Score: {score:.4f})"):
                    st.markdown(
                        f"""
- **chunk_id:** `{doc.metadata.get('chunk_id', 'N/A')}`
- **source:** `{doc.metadata.get('source', 'N/A')}`

{doc.page_content}
"""
                    )

