import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "langchain"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        tenant="default_tenant",
        database="default_database"
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    return retriever

# if __name__ == "__main__":
#     retriever = get_retriever()
#     docs = retriever.invoke("Đào Chí Trung")
#     for d in docs:
#         print(d.page_content)
