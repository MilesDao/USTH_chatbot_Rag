import os
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from retriever import E5Retriever

from deepeval.metrics import ContextualPrecisionMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Disable DeepEval telemetry to prevent "Task destroyed but pending" errors on exit
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



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
       - KHÔNG được tự ý thay đổi thuật ngữ chuyên ngành (Ví dụ: phải phân biệt rõ "Điểm thi", "Điểm học phần", "Điểm tổng kết"). Nếu văn bản ghi là "điểm thi kết thúc học phần", hãy dùng chính xác cụm từ đó.

    2. **PHONG CÁCH TƯ VẤN:**
       - Dữ liệu đầu vào là các văn bản hành chính/quy chế (Quyết định, Thông tư...), nhiệm vụ của bạn là **diễn giải lại** thành ngôn ngữ tư vấn dễ hiểu, thân thiện cho học sinh/sinh viên.
       - Giọng điệu: Chuyên nghiệp, nhiệt tình, khích lệ.
       - Xưng hô: "Trợ lý" hoặc "Mình" và gọi người dùng là "Bạn".
                                              

    3. **TRÌNH BÀY:**
       - Sử dụng **in đậm** cho các thông tin quan trọng (Hạn chót, Mức học phí, Điểm số yêu cầu, Tên chứng chỉ).
       - Sử dụng gạch đầu dòng để liệt kê các bước hoặc điều kiện.
       - Nếu câu trả lời dài, hãy tóm tắt ý chính ở đầu.
       - Tuyệt đối trích dẫn đúng cụm từ trong văn bản gốc, không tự ý rút gọn.

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




class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, api_key=None):
        # Lấy API KEY từ tham số hoặc biến môi trường
        self.api_key = api_key if api_key else os.getenv("GOOGLE_API_KEY")
        
        # Truyền key vào tham số google_api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=self.api_key 
        )
    def load_model(self): return self.model
    def generate(self, prompt): return self.model.invoke(prompt).content
    async def a_generate(self, prompt): return (await self.model.ainvoke(prompt)).content
    def get_model_name(self): return "gemini-2.5-flash"

def quick_evaluate(query, agentic_result, expected_output, expected_context, api_key=None):
    """
    query: Câu hỏi
    agentic_result: Tuple (answer_text, list_of_docs) trả về từ hàm rag_answer
    expected_output: Câu trả lời mẫu (String)
    expected_context: List các ý chính trong context mẫu (List[String])
    """
    
    
    actual_output, docs = agentic_result
    
    # Chuyển đổi docs từ agentic thành list string cho deepeval
    # Kiểm tra kỹ nếu docs rỗng để tránh lỗi
    if docs and isinstance(docs, list):
        retrieval_context = [d[0].page_content for d in docs]
    else:
        retrieval_context = []

    # Tạo test case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
        expected_context=expected_context
    )

    judge = GeminiJudge(api_key=api_key)

    # Metric 1: Contextual Precision
    precision = ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)
    
    # Metric 2: Correctness
    correctness = GEval(
        name="Correctness",
        criteria="Is the actual output factually consistent with the expected output?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.5
    )

    print(f"\n--- ĐANG CHẤM ĐIỂM: {query} ---")
    
    # Đo lường
    precision.measure(test_case)
    print(f"✅ Text Chunk Precision: {precision.score} (Reason: {precision.reason})")

    correctness.measure(test_case)
    print(f"✅ Answer Correctness:   {correctness.score} (Reason: {correctness.reason})")

def evaluate_rag_answer(query, actual_answer, retrieved_docs, expected_output, expected_context, api_key=None):
    """
    Evaluates the RAG answer and returns a dictionary of metrics.
    """
    if retrieved_docs and isinstance(retrieved_docs, list):
        retrieval_context = [d[0].page_content for d in retrieved_docs]
    else:
        retrieval_context = []

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
        expected_context=expected_context
    )

    judge = GeminiJudge(api_key=api_key)

    precision = ContextualPrecisionMetric(threshold=0.5, model=judge, include_reason=True)
    correctness = GEval(
        name="Correctness",
        criteria="Is the actual output factually consistent with the expected output?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.5
    )

    precision.measure(test_case)
    correctness.measure(test_case)

    return {
        "precision_score": precision.score,
        "precision_reason": precision.reason,
        "correctness_score": correctness.score,
        "correctness_reason": correctness.reason
    }


if __name__ == "__main__":
    # 1. Định nghĩa câu hỏi và đáp án chuẩn (Golden Data)
    question = "Điều kiện thi cải thiện là gì?"
    
    # Đáp án kỳ vọng
    expected_answer = """
Sinh viên phải có điểm thi kết thúc học phần từ 10.0/20.0 trở lên.
"""
    # Context kỳ vọng (Ý chính cần có trong tài liệu)
    expected_chunks = [
        "Students who have the overall course score of 10.0/20.0 or higher are allowed to register for a score improvement examination"
    ]

    print(f"Câu hỏi: {question}")
    print("Đang chạy Agentic RAG...")
    
    try:
        # 2. Gọi hàm RAG
        # result = (answer_string, list_of_docs)
        result = rag_answer(question) 
        
        actual_answer, retrieved_docs = result

        # --- IN KẾT QUẢ ĐỂ KIỂM TRA ---
        print("\n" + "="*50)
        print("CÂU TRẢ LỜI CỦA AI:")
        print(actual_answer)
        print("="*50)

        print(f"\nTÌM THẤY {len(retrieved_docs)} TEXT CHUNKS:")
        for i, (doc, score) in enumerate(retrieved_docs):
            print(f"   [{i+1}] Score: {score:.4f} | Content: {doc.page_content[:100]}...") 


        print("\n" + "-"*20 + " BẮT ĐẦU CHẤM ĐIỂM " + "-"*20)
        quick_evaluate(question, result, expected_answer, expected_chunks)
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()



