import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import get_retriever

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def build_rag_chain():
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template("""
Bạn là một trợ lý học tập đáng yêu theo phong cách anime, luôn nhiệt huyết giúp đỡ học sinh như một quản gia trung thành phục vụ "cậu chủ". 
Hãy trả lời dễ hiểu, thân thiện, động viên và một chút đáng yêu nhé ~ UwU

Chỉ dựa trên các tài liệu sau để trả lời. 
Nếu không đủ thông tin thì hãy thành thật nói rằng bạn không biết, đừng bịa đặt nha cậu chủ!

Context:
{context}

Question:
{question}

Hãy trả lời bằng tiếng Việt, giọng kiểu anime đáng yêu, gần gũi và lễ phép với "cậu chủ".
""")


    # llm = ChatOpenAI(
    #     base_url="https://router.huggingface.co/v1",
    #     api_key=os.getenv("HF_TOKEN"),
    #     model="SeaLLMs/SeaLLMs-v3-7B-Chat:featherless-ai",
    #     temperature=0,
    #     max_tokens=500,
    # )
    llm = ChatGoogleGenerativeAI(
        model ="gemini-2.5-flash",
        temperature=0,
        max_tokens=500

    )

    chain = (
        {
            "context": itemgetter("question")
                    | retriever
                    | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# if __name__ == "__main__":
#     rag_chain = build_rag_chain()
#     question = "Đào Chí Trung là ai?"
#     answer = rag_chain.invoke({"question": question})
#     print(answer)
