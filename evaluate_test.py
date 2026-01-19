import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.models.base_model import DeepEvalBaseLLM

from retriever import retrieve_with_score

load_dotenv()


# 1. CẤU HÌNH JUDGE

class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        result = await self.llm.ainvoke(prompt)
        return result.content

    def get_model_name(self):
        return self.model_name

gemini_judge = GeminiJudge()


# 2. HÀM CHẠY RAG

def generate_rag_response(question: str):
    results = retrieve_with_score(question, k=5)
    
    if not results:
        return "Không tìm thấy thông tin.", []

    retrieval_context = [doc.page_content for doc, score in results]
    context_str = "\n\n".join(retrieval_context)

    prompt_template = ChatPromptTemplate.from_template("""
    Dựa vào thông tin dưới đây, hãy trả lời câu hỏi.
    Context: {context}
    Question: {question}
    Answer:
    """)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    chain = prompt_template | llm | StrOutputParser()
    actual_output = chain.invoke({"context": context_str, "question": question})
    
    return actual_output, retrieval_context


# 3. GOLDEN DATASET

golden_dataset = [
    {
        "input": "Điều kiện để sinh viên được đăng ký thi cải thiện điểm là gì?",
        "expected_output": "Sinh viên phải có điểm thi kết thúc học phần từ 10.0/20.0 trở lên mới được đăng ký thi cải thiện.",
        "expected_context": [
            "Students who have the final exam score of 10.0 or higher are allowed to register for a score improvement examination"
        ]
    },
    {
        "input": "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ để thi cải thiện trong một học kỳ?",
        "expected_output": "Sinh viên được đăng ký tối đa 10 tín chỉ mỗi học kỳ cho việc thi cải thiện.",
        "expected_context": [
            "The maximum number of credits registered for the improvement examination is 10 credits/semester."
        ]
    },
    {
        "input": "Học bổng Talent loại 1 (Talent 1) có giá trị bao nhiêu và dành cho ai?",
        "expected_output": "Học bổng Talent 1 có giá trị 100% học phí. Dành cho sinh viên đạt giải Nhì kỳ thi Học sinh giỏi Quốc gia (HSG QG).",
        "expected_context": [
            "Talentl", 
            "100% học phí",
            "Đạt giải Nhì kỳ thi HSG QG"
        ]
    },
    {
        "input": "Điều kiện về điểm số để xét học bổng thực tập loại A1 tại Pháp là gì?",
        "expected_output": "Để xét học bổng thực tập loại A1 tại Pháp, sinh viên cần có GPA lớn hơn 17.0/20.",
        "expected_context": [
            "Thực tập tại Pháp",
            "A1",
            "GPA > 17.0/20"
        ]
    }
]


# 4. CHẠY ĐÁNH GIÁ (Dùng GEval cho Correctness)

def run_evaluation():
    if not os.getenv("GOOGLE_API_KEY"):
        print(" LỖI: Chưa tìm thấy GOOGLE_API_KEY.")
        return

    test_cases = []
    print(f" Đang đánh giá {len(golden_dataset)} test cases...")

    for i, record in enumerate(golden_dataset):
        print(f"\n--- Query {i+1}: {record['input']} ---")
        try:
            actual_output, retrieval_context = generate_rag_response(record['input'])
            print(f"-> Generated: {actual_output[:100]}...")

            test_case = LLMTestCase(
                input=record['input'],
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=record['expected_output'], # Cần cho Correctness
                expected_context=record['expected_context'] # Cần cho Precision
            )
            test_cases.append(test_case)
        except Exception as e:
            print(f"Lỗi: {e}")

    # --- KHAI BÁO METRICS ---
    
    # 1. Contextual Precision
    precision_metric = ContextualPrecisionMetric(
        threshold=0.7, model=gemini_judge, include_reason=True
    )
    
    # 2. Faithfulness
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7, model=gemini_judge, include_reason=True
    )

    
    correctness_metric = GEval(
        name="Answer Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=gemini_judge,
        threshold=0.7
    )

    print("\n\n=== KẾT QUẢ ĐÁNH GIÁ (RAG TRIAD với GEval) ===")
    
    evaluate(
        test_cases=test_cases,
        metrics=[precision_metric, faithfulness_metric, correctness_metric]
    )

if __name__ == "__main__":
    run_evaluation()