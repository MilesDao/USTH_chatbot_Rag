import os
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from retriever import E5Retriever


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# CONTEXT GATE (CÓ RELEVANCE)

def is_context_usable(docs_with_scores: List[Tuple[Document, float]],
    min_chunks: int = 1,
    min_chars: int = 100,
    max_score: float = 0.6,
) -> bool:
    """
    Context usable nếu:
    - Có >= min_chunks
    - Tổng ký tự >= min_chars
    - Ít nhất 1 chunk có score đủ tốt
    """
    if not docs_with_scores or len(docs_with_scores) < min_chunks:
        return False

    total_chars = sum(len(doc.page_content) for doc, _ in docs_with_scores)
    if total_chars < min_chars:
        return False

    best_score = min(score for _, score in docs_with_scores)
    return best_score <= max_score

# Build Context

def build_context(docs_with_scores: List[Tuple[Document, float]]) -> str:
    blocks = []

    for doc, score in docs_with_scores:
        blocks.append(
            f"[chunk_id={doc.metadata.get('chunk_id', 'N/A')}, "
            f"source={doc.metadata.get('source', 'N/A')}, "
            f"score={score:.4f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)


def build_rag_chain(api_key: str ):
    prompt = ChatPromptTemplate.from_template("""
Bạn là Trợ lý AI Tư vấn Tuyển sinh và Hỗ trợ Sinh viên của Trường Đại học Khoa học và Công nghệ Hà Nội (USTH - Đại học Việt Pháp).
    
    Nhiệm vụ của bạn là giải đáp thắc mắc dựa trên cơ sở dữ liệu quy chế và thông tin tuyển sinh được cung cấp.
    
    DỮ LIỆU CUNG CẤP (CONTEXT) TỪ FILE FINALDATA.TXT:
    {context}
    
    CÂU HỎI CỦA NGƯỜI DÙNG:
    {question}
    
    ---
    QUY TẮC TRẢ LỜI (BẮT BUỘC):
    
    1. **NGUYÊN TẮC TRUNG THỰC:**
       - CHỈ sử dụng thông tin có trong [CONTEXT].
       - Nếu thông tin không có trong tài liệu, hãy trả lời: "Xin lỗi, hiện tại trong tài liệu quy chế mình chưa tìm thấy thông tin cụ thể về vấn đề này. Bạn vui lòng liên hệ trực tiếp phòng Đào tạo hoặc Fanpage USTH để được hỗ trợ chính xác nhất."
       - KHÔNG tự suy đoán hoặc bịa đặt thông tin (đặc biệt là các con số, ngày tháng).

    2. **PHONG CÁCH TƯ VẤN:**
       - Dữ liệu đầu vào là các văn bản hành chính/quy chế (Quyết định, Thông tư...), nhiệm vụ của bạn là **diễn giải lại** thành ngôn ngữ tư vấn dễ hiểu, thân thiện cho học sinh/sinh viên.
       - Giọng điệu: Chuyên nghiệp, nhiệt tình, khích lệ.
       - Xưng hô: "Trợ lý" hoặc "Mình" và gọi người dùng là "Bạn".

    3. **TRÌNH BÀY:**
       - Sử dụng **in đậm** cho các thông tin quan trọng (Hạn chót, Mức học phí, Điểm số yêu cầu, Tên chứng chỉ).
       - Sử dụng gạch đầu dòng để liệt kê các bước hoặc điều kiện.
       - Nếu câu trả lời dài, hãy tóm tắt ý chính ở đầu.

Hãy trả lời bằng tiếng Việt, rõ ràng, chính xác và trung thực.
""")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        max_tokens=1200,
        google_api_key=api_key
    )

    return prompt | llm | StrOutputParser()


# Agentic RAG (GENERAL)
def rag_answer(
    question: str,
    k: int = 8,
    api_key: str = GOOGLE_API_KEY,
):
    retriever = E5Retriever(k=k)

    docs_with_scores = retriever.retrieve_with_score(question)

    if not is_context_usable(docs_with_scores):
        return (
            "Mình không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này.",
            docs_with_scores,
        )

    context = build_context(docs_with_scores)

    chain = build_rag_chain(api_key=api_key)
    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return answer, docs_with_scores


# =====================
# DEBUG
# =====================
if __name__ == "__main__":
    question = "có bao nhiêu loại học bổng USTH"

    answer, results = rag_answer(question)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== RETRIEVED CHUNKS ===")
    for doc, score in results:
        print(
            f"\nScore: {score:.4f} | "
            f"chunk_id={doc.metadata.get('chunk_id', 'N/A')} | "
            f"source={doc.metadata.get('source', 'N/A')}"
        )
        print(doc.page_content[:400], "...")