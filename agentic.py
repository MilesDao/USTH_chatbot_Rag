import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_with_score

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def is_context_usable(docs, min_chunks: int = 1, min_chars: int = 200):
    """
    Gate logic KHÔNG dựa score.
    Chỉ kiểm tra:
    - Có đủ chunk?
    - Context có đủ nội dung chữ?
    """
    if not docs or len(docs) < min_chunks:
        return False

    total_chars = sum(len(doc.page_content) for doc in docs)
    return total_chars >= min_chars


# Build Context

def build_context(docs):
    blocks = []

    for doc in docs:
        blocks.append(
            f"[chunk_id={doc.metadata.get('chunk_id', 'N/A')}, "
            f"page={doc.metadata.get('page', 'N/A')}, "
            f"source={doc.metadata.get('source', 'N/A')}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)


def build_rag_chain(api_key: str = None):
    prompt = ChatPromptTemplate.from_template("""
Bạn là một trợ lý học tập đáng tin cậy.

QUY TẮC BẮT BUỘC:
- CHỈ sử dụng thông tin có trong Context.
- KHÔNG suy đoán.
- KHÔNG dùng kiến thức bên ngoài.
- Nếu Context không đủ để trả lời → nói rõ là không tìm thấy thông tin.

Context:
{context}

Question:
{question}

Hãy trả lời bằng tiếng Việt, rõ ràng, chính xác và trung thực.
""")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        max_tokens=1000,
        google_api_key=api_key
    )

    return prompt | llm | StrOutputParser()


# Agentic RAG (GENERAL)

def rag_answer(question: str, k: int = 5, api_key: str = None):

    results = retrieve_with_score(question, k=k)
    

    docs = [doc for doc, score in results]

    if not is_context_usable(docs):
        return (
            "Em không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này.",
            results
        )

    context = build_context(docs)

# LLM
    chain = build_rag_chain(api_key=api_key)
    answer = chain.invoke({
        "question": question,
        "context": context
    })

    return answer, results


if __name__ == "__main__":
    question = "CÁC CÂU LẠC BỘ TẠI USTH"
    answer, results = rag_answer(question)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== RETRIEVED CHUNKS ===")
    for doc, score in results:
        print(
            f"\nScore: {score:.4f} | "
            f"chunk_id={doc.metadata.get('chunk_id', 'N/A')} | "
            f"page={doc.metadata.get('page', 'N/A')} | "
            f"source={doc.metadata.get('source', 'N/A')}"
        )
        print(doc.page_content[:500], "...")
