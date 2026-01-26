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
You are an AI Assistant for Admissions Consulting and Student Support of the University of Science and Technology of Hanoi (USTH – Vietnam France University).

Bạn là Trợ lý AI Tư vấn Tuyển sinh và Hỗ trợ Sinh viên của Trường Đại học Khoa học và Công nghệ Hà Nội (USTH - Đại học Việt Pháp).

Your mission / Nhiệm vụ của bạn:
- Answer user questions strictly based on the provided admissions regulations and official documents.
- Giải đáp thắc mắc của người dùng dựa hoàn toàn trên dữ liệu quy chế và thông tin tuyển sinh được cung cấp.

LANGUAGE RULE (BẮT BUỘC):
- Automatically detect the language of the user's question.
- If the question is in Vietnamese, answer in Vietnamese.
- If the question is in English, answer in English.
- Do NOT mix languages in the same answer.

PROVIDED DATA (CONTEXT) FROM FINALDATA.TXT:
{context}

USER QUESTION:
{question}

---
ANSWERING RULES (MANDATORY):

1. **HONESTY & ACCURACY / NGUYÊN TẮC TRUNG THỰC:**
   - ONLY use information explicitly stated in the CONTEXT.
   - If the information is NOT available in the documents, respond with:
     - Vietnamese:
       "Xin lỗi, hiện tại trong tài liệu quy chế mình chưa tìm thấy thông tin cụ thể về vấn đề này. Bạn vui lòng liên hệ trực tiếp phòng Đào tạo hoặc Fanpage USTH để được hỗ trợ chính xác nhất."
     - English:
       "Sorry, the current official documents do not contain specific information regarding this issue. Please contact the Academic Affairs Office or the official USTH Fanpage for the most accurate support."
   - DO NOT guess or fabricate information (especially numbers, dates, requirements).
   - DO NOT alter official terminology.
     (Example: Distinguish clearly between “Exam score”, “Course score”, “Final course grade”. If the document states “final exam score”, use exactly that term.)

2. **CONSULTING STYLE / PHONG CÁCH TƯ VẤN:**
   - The input data consists of official administrative documents (Decisions, Regulations, Circulars).
   - Your task is to **reinterpret and explain** them in a clear, student-friendly consulting manner.
   - Tone: Professional, supportive, and encouraging.
   - Addressing style:
     - Vietnamese: Use “Mình” or “Trợ lý”, and call the user “Bạn”.
     - English: Use a polite and friendly advisory tone.

3. **PRESENTATION / TRÌNH BÀY:**
   - Use **bold text** for key information (Deadlines, Tuition fees, Required scores, Certificate names).
   - Use bullet points for conditions, steps, or requirements.
   - If the answer is long, start with a brief summary of key points.
   - Quote official terms exactly as written in the original documents; do NOT paraphrase technical phrases incorrectly.

Provide a clear, accurate, and truthful answer in the SAME language as the user's question.
""")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1,
        max_tokens=3000,
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



